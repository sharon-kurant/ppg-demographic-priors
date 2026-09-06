"""Manifest-only ingestion for PulseDB VitalDB and MIMIC cohorts.

PulseDB is distributed as one MATLAB/HDF5 file per subject.  The manifest keeps
references to the source arrays instead of creating millions of tiny waveform
files.  Both dense numeric arrays and MATLAB cell/reference arrays are handled.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ppg_bp_incremental.data.benchmark import canonicalize_manifest
from ppg_bp_incremental.data.ppgbp import file_digest
from ppg_bp_incremental.models.contracts import CONTRACT_VERSION


PULSEDB_SAMPLE_RATE_HZ = 125
# PulseDB v2 explicitly distinguishes raw ``PPG_Record`` from the legacy
# normalized/filtered ``PPG_F``. Corrected runs are intentionally strict:
# silently falling back would recreate the double-filtering defect.
_PPG_KEYS = ("PPG_Record",)
_SBP_KEYS = ("SegSBP", "SBP", "ABP_SBP", "sbp")
_DBP_KEYS = ("SegDBP", "DBP", "ABP_DBP", "dbp")
_INCLUDE_KEYS = ("IncludeFlag", "include_flag", "include")


def _canonical_sex(value: Any) -> str:
    token = str(value).strip().lower()
    if token in {"m", "male", "1", "1.0"}:
        return "male"
    if token in {"f", "female", "0", "0.0"}:
        return "female"
    return ""


def _first(mapping: Any, names: Iterable[str]):
    for name in names:
        if name in mapping:
            return name, mapping[name]
    raise ValueError(f"None of the required fields are present: {tuple(names)}")


def validate_pulsedb_raw_waveform_contract(data: pd.DataFrame) -> None:
    """Reject manifests whose source waveform is not raw ``PPG_Record``."""

    if not data["dataset"].astype(str).str.startswith("PulseDB").all():
        return
    provenance_column = (
        "source_waveform_dataset"
        if "source_waveform_dataset" in data
        else "waveform_dataset"
    )
    if provenance_column not in data:
        raise ValueError("PulseDB manifest does not identify its source waveform field")
    fields = data[provenance_column].astype(str).str.rsplit("/", n=1).str[-1]
    invalid = sorted(set(fields[fields != "PPG_Record"]))
    if invalid:
        raise ValueError(
            "Corrected PulseDB runs require raw PPG_Record and reject normalized/"
            f"filtered fields such as PPG_F; observed {invalid}"
        )


def _decode_char(values: np.ndarray) -> str:
    values = np.asarray(values).reshape(-1)
    if values.dtype.kind == "O" and values.size:
        flattened = []
        for value in values:
            flattened.extend(np.asarray(value).reshape(-1).tolist())
        return _decode_char(np.asarray(flattened))
    if values.dtype.kind == "U":
        return "".join(str(x) for x in values).strip()
    if values.dtype.kind == "S":
        return b"".join(bytes(x) for x in values).decode("utf-8").strip()
    if values.dtype.kind in {"u", "i"} and values.size:
        return "".join(chr(int(x)) for x in values if int(x) > 0).strip()
    return str(values[0]) if values.size else ""


def _h5_values(handle, node) -> np.ndarray:
    values = np.asarray(node)
    if getattr(node.dtype, "kind", "") == "O":
        resolved = [np.asarray(handle[ref]).squeeze() for ref in values.reshape(-1)]
        if all(np.asarray(value).size == 1 for value in resolved):
            return np.asarray([np.asarray(value).item() for value in resolved])
        return np.asarray(resolved, dtype=object)
    return values.squeeze()


def _h5_scalar(handle, group, names: Iterable[str], default=np.nan):
    names = tuple(names)
    text_hint = any(
        token in name.lower() for name in names for token in ("subject", "gender", "sex")
    )
    for name in names:
        if name not in group:
            continue
        node = group[name]
        if getattr(node.dtype, "kind", "") == "O":
            references = np.asarray(node).reshape(-1)
            if references.size == 0:
                return default
            values = np.asarray(handle[references[0]]).squeeze()
        else:
            values = _h5_values(handle, node)
        array = np.asarray(values)
        if array.dtype.kind in {"U", "S", "O"} or (
            text_hint and array.dtype.kind in {"u", "i"} and np.any(array > 1)
        ):
            return _decode_char(array)
        return array.reshape(-1)[0] if array.size else default
    return default


def _segment_axis(shape: tuple[int, ...], n_segments: int) -> int:
    if len(shape) == 1:
        return 0
    matches = [axis for axis, length in enumerate(shape) if length == n_segments]
    if len(matches) == 1:
        return matches[0]
    if shape[0] == n_segments:
        return 0
    if shape[-1] == n_segments:
        return len(shape) - 1
    raise ValueError(f"Cannot align waveform shape {shape} with {n_segments} labels")


def _include_mask(mapping: Any, value_loader, n_segments: int) -> np.ndarray:
    for name in _INCLUDE_KEYS:
        if name not in mapping:
            continue
        values = np.asarray(value_loader(mapping[name]), dtype=float).reshape(-1)
        if len(values) != n_segments:
            raise ValueError(
                f"PulseDB {name} length {len(values)} does not match {n_segments} labels"
            )
        return np.isfinite(values) & (values > 0)
    return np.ones(n_segments, dtype=bool)


def _subject_token(value: Any, path: Path) -> str:
    token = str(value).strip()
    if not token or token.lower() == "nan":
        token = path.stem
    return token.replace(" ", "_")


def _records_from_hdf5(
    path: Path,
    cohort: str,
    sampling_rate_hz: int,
    segment_indices: Iterable[int] | None = None,
) -> list[dict]:
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Preparing PulseDB MATLAB files requires h5py") from error
    with h5py.File(path, "r") as handle:
        group = handle["Subj_Wins"] if "Subj_Wins" in handle else handle
        ppg_name, ppg_node = _first(group, _PPG_KEYS)
        _, sbp_node = _first(group, _SBP_KEYS)
        _, dbp_node = _first(group, _DBP_KEYS)
        sbp = np.asarray(_h5_values(handle, sbp_node), dtype=float).reshape(-1)
        dbp = np.asarray(_h5_values(handle, dbp_node), dtype=float).reshape(-1)
        if len(sbp) != len(dbp):
            raise ValueError(f"PulseDB SBP/DBP length mismatch in {path}")
        n_segments = len(sbp)
        included = _include_mask(group, lambda node: _h5_values(handle, node), n_segments)
        indices = (
            list(range(n_segments))
            if segment_indices is None
            else sorted({int(index) for index in segment_indices})
        )
        if any(index < 0 or index >= n_segments for index in indices):
            raise IndexError(f"Selected PulseDB segment is out of range in {path}")
        if getattr(ppg_node.dtype, "kind", "") == "O":
            axis = 0
            references = np.asarray(ppg_node).reshape(-1)
            if len(references) != n_segments:
                raise ValueError(f"PulseDB waveform/label count mismatch in {path}")
            # ``Dataset.size`` is HDF5 metadata. Converting every referenced
            # waveform with np.asarray merely to obtain its size previously
            # materialized all segments in every selected subject file and
            # exhausted the 16 GiB CPU-job limit.
            lengths = {
                index: int(handle[references[index]].size) for index in indices
            }
        else:
            axis = _segment_axis(tuple(ppg_node.shape), n_segments)
            sample_axis = next(index for index in range(ppg_node.ndim) if index != axis)
            lengths = {
                index: int(ppg_node.shape[sample_axis]) for index in indices
            }
        subject = _subject_token(
            _h5_scalar(
                handle,
                group,
                ("SubjectID", "Subject", "subject", "Subject_ID"),
                path.stem,
            ),
            path,
        )
        age = _h5_scalar(handle, group, ("Age", "age"))
        sex = _h5_scalar(handle, group, ("Gender", "Sex", "gender", "sex"))
        height = _h5_scalar(handle, group, ("Height", "height"))
        weight = _h5_scalar(handle, group, ("Weight", "weight"))
        bmi = _h5_scalar(handle, group, ("BMI", "bmi"))
        dataset_path = ppg_node.name

        waveform_valid: dict[int, bool] = {}
        # The subset path validates the actual selected PPG values.  Avoid
        # reading every waveform when preparing an unsampled legacy manifest.
        if segment_indices is not None:
            for index in indices:
                if getattr(ppg_node.dtype, "kind", "") == "O":
                    values = np.asarray(handle[references[index]]).reshape(-1)
                elif ppg_node.ndim == 1:
                    values = np.asarray(ppg_node[...]).reshape(-1)
                elif axis == 0:
                    values = np.asarray(ppg_node[index, ...]).reshape(-1)
                else:
                    values = np.asarray(ppg_node[..., index]).reshape(-1)
                waveform_valid[index] = bool(values.size >= 2 and np.isfinite(values).all())

    source_hash = file_digest(path)
    prefix = "pulsedb-vital" if cohort == "vital" else "pulsedb-mimic"
    subject_id = f"{prefix}-{subject}"
    records = []
    for index in indices:
        sys = sbp[index]
        dia = dbp[index]
        length = lengths[index]
        records.append(
            {
                "dataset": "PulseDB-Vital" if cohort == "vital" else "PulseDB-MIMIC",
                "dataset_version": "1.0",
                "source": "VitalDB" if cohort == "vital" else "MIMIC-III",
                "subject_id": subject_id,
                "measurement_id": f"{subject_id}-{index:06d}",
                "segment_id": f"{subject_id}-{index:06d}",
                "waveform_path": str(path.resolve()),
                "waveform_sha256": source_hash,
                "waveform_format": path.suffix.lower().lstrip("."),
                "waveform_dataset": dataset_path,
                "waveform_index": index,
                "waveform_axis": axis,
                "sample_rate_hz": sampling_rate_hz,
                "n_samples": length,
                "sbp": float(sys),
                "dbp": float(dia),
                "age": age,
                "sex": sex,
                "height_cm": height,
                "weight_kg": weight,
                "bmi": bmi,
                "include_flag": int(included[index]),
                "valid_for_analysis": bool(
                    np.isfinite(sys)
                    and np.isfinite(dia)
                    and sys > dia
                    and included[index]
                    and waveform_valid.get(index, True)
                ),
                "pretraining_overlap_group": "pulsedb_vital" if cohort == "vital" else "pulsedb_mimic",
            }
        )
    return records


def _npz_field(archive, names: Iterable[str], default=None):
    for name in names:
        if name in archive:
            return np.asarray(archive[name])
    return default


def _records_from_npz(
    path: Path,
    cohort: str,
    sampling_rate_hz: int,
    segment_indices: Iterable[int] | None = None,
) -> list[dict]:
    archive = np.load(path, allow_pickle=False)
    ppg_name, ppg = _first(archive, _PPG_KEYS)
    _, sbp = _first(archive, _SBP_KEYS)
    _, dbp = _first(archive, _DBP_KEYS)
    sbp = np.asarray(sbp, dtype=float).reshape(-1)
    dbp = np.asarray(dbp, dtype=float).reshape(-1)
    included = _include_mask(archive, np.asarray, len(sbp))
    axis = _segment_axis(tuple(ppg.shape), len(sbp))
    sample_axis = next(index for index in range(ppg.ndim) if index != axis)
    subject_raw = _npz_field(
        archive, ("SubjectID", "Subject", "subject", "Subject_ID"), path.stem
    )
    subject = _subject_token(np.asarray(subject_raw).reshape(-1)[0], path)
    scalar = lambda names: np.asarray(_npz_field(archive, names, [np.nan])).reshape(-1)[0]
    source_hash = file_digest(path)
    prefix = "pulsedb-vital" if cohort == "vital" else "pulsedb-mimic"
    subject_id = f"{prefix}-{subject}"
    indices = (
        list(range(len(sbp)))
        if segment_indices is None
        else sorted({int(index) for index in segment_indices})
    )
    if any(index < 0 or index >= len(sbp) for index in indices):
        raise IndexError(f"Selected PulseDB segment is out of range in {path}")
    records = []
    for index in indices:
        sys = sbp[index]
        dia = dbp[index]
        values = ppg[index] if axis == 0 else ppg[..., index]
        records.append(
            {
                "dataset": "PulseDB-Vital" if cohort == "vital" else "PulseDB-MIMIC",
                "dataset_version": "1.0",
                "source": "VitalDB" if cohort == "vital" else "MIMIC-III",
                "subject_id": subject_id,
                "measurement_id": f"{subject_id}-{index:06d}",
                "segment_id": f"{subject_id}-{index:06d}",
                "waveform_path": str(path.resolve()),
                "waveform_sha256": source_hash,
                "waveform_format": "npz",
                "waveform_dataset": ppg_name,
                "waveform_index": index,
                "waveform_axis": axis,
                "sample_rate_hz": sampling_rate_hz,
                "n_samples": int(ppg.shape[sample_axis]),
                "sbp": float(sys),
                "dbp": float(dia),
                "age": scalar(("Age", "age")),
                "sex": scalar(("Gender", "Sex", "gender", "sex")),
                "height_cm": scalar(("Height", "height")),
                "weight_kg": scalar(("Weight", "weight")),
                "bmi": scalar(("BMI", "bmi")),
                "include_flag": int(included[index]),
                "valid_for_analysis": bool(
                    np.isfinite(sys)
                    and np.isfinite(dia)
                    and sys > dia
                    and included[index]
                    and np.asarray(values).size >= 2
                    and np.isfinite(values).all()
                ),
                "pretraining_overlap_group": "pulsedb_vital" if cohort == "vital" else "pulsedb_mimic",
            }
        )
    return records


def _summary_from_hdf5(path: Path, cohort: str) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Preparing PulseDB MATLAB files requires h5py") from error
    with h5py.File(path, "r") as handle:
        group = handle["Subj_Wins"] if "Subj_Wins" in handle else handle
        _, ppg_node = _first(group, _PPG_KEYS)
        _, sbp_node = _first(group, _SBP_KEYS)
        _, dbp_node = _first(group, _DBP_KEYS)
        sbp = np.asarray(_h5_values(handle, sbp_node), dtype=float).reshape(-1)
        dbp = np.asarray(_h5_values(handle, dbp_node), dtype=float).reshape(-1)
        if len(sbp) != len(dbp):
            raise ValueError(f"PulseDB SBP/DBP length mismatch in {path}")
        included = _include_mask(group, lambda node: _h5_values(handle, node), len(sbp))
        if getattr(ppg_node.dtype, "kind", "") == "O":
            ppg_segments = np.asarray(ppg_node).size
        else:
            ppg_segments = ppg_node.shape[_segment_axis(tuple(ppg_node.shape), len(sbp))]
        if ppg_segments != len(sbp):
            raise ValueError(f"PulseDB waveform/label count mismatch in {path}")
        subject = _subject_token(
            _h5_scalar(
                handle,
                group,
                ("SubjectID", "Subject", "subject", "Subject_ID"),
                path.stem,
            ),
            path,
        )
        age = _h5_scalar(handle, group, ("Age", "age"))
        sex = _h5_scalar(handle, group, ("Gender", "Sex", "gender", "sex"))
    valid = np.flatnonzero(
        np.isfinite(sbp) & np.isfinite(dbp) & (sbp > dbp) & included
    )
    prefix = "pulsedb-vital" if cohort == "vital" else "pulsedb-mimic"
    return {
        "waveform_path": str(path.resolve()),
        "subject_id": f"{prefix}-{subject}",
        "age": pd.to_numeric(age, errors="coerce"),
        "sex": _canonical_sex(sex),
        "subject_median_sbp": float(np.median(sbp[valid])) if len(valid) else np.nan,
        "subject_median_dbp": float(np.median(dbp[valid])) if len(valid) else np.nan,
        "eligible_segments": int(len(valid)),
        "_eligible_segment_indices": valid.astype(np.int32),
    }


def _summary_from_npz(path: Path, cohort: str) -> dict[str, Any]:
    archive = np.load(path, allow_pickle=False)
    _, ppg = _first(archive, _PPG_KEYS)
    _, sbp = _first(archive, _SBP_KEYS)
    _, dbp = _first(archive, _DBP_KEYS)
    sbp = np.asarray(sbp, dtype=float).reshape(-1)
    dbp = np.asarray(dbp, dtype=float).reshape(-1)
    included = _include_mask(archive, np.asarray, len(sbp))
    axis = _segment_axis(tuple(np.asarray(ppg).shape), len(sbp))
    if np.asarray(ppg).shape[axis] != len(sbp) or len(sbp) != len(dbp):
        raise ValueError(f"PulseDB waveform/label count mismatch in {path}")
    subject_raw = _npz_field(
        archive, ("SubjectID", "Subject", "subject", "Subject_ID"), path.stem
    )
    subject = _subject_token(np.asarray(subject_raw).reshape(-1)[0], path)
    age = np.asarray(_npz_field(archive, ("Age", "age"), [np.nan])).reshape(-1)[0]
    sex = np.asarray(
        _npz_field(archive, ("Gender", "Sex", "gender", "sex"), [np.nan])
    ).reshape(-1)[0]
    valid = np.flatnonzero(
        np.isfinite(sbp) & np.isfinite(dbp) & (sbp > dbp) & included
    )
    prefix = "pulsedb-vital" if cohort == "vital" else "pulsedb-mimic"
    return {
        "waveform_path": str(path.resolve()),
        "subject_id": f"{prefix}-{subject}",
        "age": pd.to_numeric(age, errors="coerce"),
        "sex": _canonical_sex(sex),
        "subject_median_sbp": float(np.median(sbp[valid])) if len(valid) else np.nan,
        "subject_median_dbp": float(np.median(dbp[valid])) if len(valid) else np.nan,
        "eligible_segments": int(len(valid)),
        "_eligible_segment_indices": valid.astype(np.int32),
    }


def _quantile_bin(values: pd.Series, bins: int = 3) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    unique = int(numeric.nunique(dropna=True))
    if unique <= 1:
        return pd.Series(
            np.where(numeric.notna(), 0, -1), dtype=int, index=values.index
        )
    return (
        pd.qcut(numeric, q=min(bins, unique), labels=False, duplicates="drop")
        .astype("Int64")
        .fillna(-1)
        .astype(int)
    )


def _stratified_subject_sample(
    summaries: list[dict[str, Any]],
    subject_limit: int,
    segments_per_subject: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    frame = pd.DataFrame(summaries)
    eligible = frame[
        np.isfinite(frame["subject_median_sbp"])
        & np.isfinite(frame["subject_median_dbp"])
        & (frame["eligible_segments"] >= segments_per_subject)
    ].copy()
    eligible = eligible.sort_values(["subject_id", "waveform_path"]).reset_index(drop=True)
    if eligible["subject_id"].duplicated().any():
        raise ValueError("PulseDB subject identifiers must be unique across source files")
    if len(eligible) < subject_limit:
        raise ValueError(
            f"Requested {subject_limit} PulseDB subjects but only {len(eligible)} are eligible"
        )
    eligible["age"] = pd.to_numeric(eligible["age"], errors="coerce")
    eligible["sex"] = eligible["sex"].where(
        eligible["sex"].isin({"female", "male"}), "unknown"
    )
    eligible["age_bin"] = _quantile_bin(eligible["age"])
    eligible["sbp_bin"] = _quantile_bin(eligible["subject_median_sbp"])
    eligible["dbp_bin"] = _quantile_bin(eligible["subject_median_dbp"])
    eligible["stratum"] = (
        eligible["sex"].astype(str)
        + "|a" + eligible["age_bin"].astype(str)
        + "|s" + eligible["sbp_bin"].astype(str)
        + "|d" + eligible["dbp_bin"].astype(str)
    )

    counts = eligible["stratum"].value_counts().sort_index()
    ideal = counts * (subject_limit / len(eligible))
    allocation = np.floor(ideal).astype(int)
    # Preserve even rare joint demographic/BP cells when the target size allows it.
    if len(allocation) <= subject_limit:
        allocation[allocation == 0] = 1
    while int(allocation.sum()) > subject_limit:
        removable = allocation[allocation > 1]
        key = sorted(removable.index, key=lambda value: (ideal[value] - allocation[value], value))[0]
        allocation[key] -= 1
    while int(allocation.sum()) < subject_limit:
        capacity = counts - allocation
        candidates = capacity[capacity > 0].index
        key = sorted(candidates, key=lambda value: (-(ideal[value] - allocation[value]), value))[0]
        allocation[key] += 1

    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    for stratum in allocation.index:
        candidates = eligible.index[eligible["stratum"] == stratum].to_numpy()
        chosen = rng.choice(candidates, size=int(allocation[stratum]), replace=False)
        selected_indices.extend(chosen.tolist())
    selected = eligible.loc[selected_indices].sort_values("subject_id").copy()
    selected["selection_seed"] = seed
    selected["selected_segment_indices"] = selected["_eligible_segment_indices"].map(
        lambda indices: json.dumps(
            [
                int(indices[position])
                for position in np.linspace(
                    0, len(indices) - 1, segments_per_subject, dtype=int
                )
            ]
        )
    )
    selected = selected.drop(columns=["_eligible_segment_indices"])
    return selected.to_dict(orient="records"), int(len(eligible))


def _write_waveform_cache(data: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    """Materialize only selected PPG arrays while preserving raw-file provenance."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Writing a PulseDB waveform cache requires h5py") from error
    from ppg_bp_incremental.data.benchmark import load_record_waveform

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lengths = pd.to_numeric(data["n_samples"], errors="raise").astype(int)
    if lengths.nunique() != 1:
        raise ValueError("The compact PulseDB cache requires one common segment length")
    n_samples = int(lengths.iloc[0])
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, "w") as handle:
        dataset = handle.create_dataset(
            "ppg",
            shape=(len(data), n_samples),
            dtype=np.float32,
            chunks=(min(256, len(data)), n_samples),
            compression="lzf",
        )
        for output_index, (_, row) in enumerate(data.iterrows()):
            waveform = load_record_waveform(row)
            if waveform.shape != (n_samples,) or not np.isfinite(waveform).all():
                raise ValueError(f"Invalid selected waveform for {row['segment_id']}")
            dataset[output_index] = waveform.astype(np.float32, copy=False)
        handle.attrs["sample_rate_hz"] = int(data["sample_rate_hz"].iloc[0])
        handle.attrs["segments"] = len(data)
    temporary.replace(output_path)
    cache_sha256 = file_digest(output_path)

    cached = data.copy()
    for column in (
        "waveform_path",
        "waveform_sha256",
        "waveform_format",
        "waveform_dataset",
        "waveform_index",
        "waveform_axis",
    ):
        if column in cached:
            cached[f"source_{column}"] = cached[column]
    cached["waveform_path"] = str(output_path.resolve())
    cached["waveform_sha256"] = cache_sha256
    cached["waveform_format"] = "h5"
    cached["waveform_dataset"] = "/ppg"
    cached["waveform_index"] = np.arange(len(cached), dtype=int)
    cached["waveform_axis"] = 0
    return cached


def prepare_pulsedb(
    raw_root: str | Path,
    cohort: str,
    output_csv: str | Path,
    provenance_json: str | Path,
    sampling_rate_hz: int = PULSEDB_SAMPLE_RATE_HZ,
    dataset_version: str = "2.0",
    subject_limit: int | None = None,
    segments_per_subject: int | None = None,
    selection_seed: int = 42,
    subject_selection_csv: str | Path | None = None,
    reuse_subject_selection_csv: str | Path | None = None,
    waveform_cache_hdf5: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if cohort not in {"vital", "mimic"}:
        raise ValueError("PulseDB cohort must be 'vital' or 'mimic'")
    raw_root = Path(raw_root)
    paths = sorted(
        path for path in raw_root.rglob("*") if path.suffix.lower() in {".mat", ".h5", ".hdf5", ".npz"}
    )
    if not paths:
        raise FileNotFoundError(f"No PulseDB MATLAB/HDF5/NPZ files found in {raw_root}")
    if subject_limit is not None and subject_limit < 1:
        raise ValueError("subject_limit must be positive")
    if segments_per_subject is not None and segments_per_subject < 1:
        raise ValueError("segments_per_subject must be positive")
    if subject_limit is not None and segments_per_subject is None:
        raise ValueError("segments_per_subject is required when subject_limit is set")

    full_source_files = len(paths)
    selected_by_path: dict[Path, list[int]] | None = None
    eligible_subjects: int | None = None
    selection_path: Path | None = None
    if reuse_subject_selection_csv is not None:
        reuse_path = Path(reuse_subject_selection_csv)
        selected_frame = pd.read_csv(reuse_path)
        required = {"subject_id", "waveform_path", "selected_segment_indices"}
        missing = required.difference(selected_frame)
        if missing:
            raise ValueError(
                f"Reused PulseDB selection is missing columns: {sorted(missing)}"
            )
        expected_prefix = "pulsedb-vital-" if cohort == "vital" else "pulsedb-mimic-"
        if not selected_frame["subject_id"].astype(str).str.startswith(expected_prefix).all():
            raise ValueError("Reused PulseDB selection belongs to a different cohort")
        if selected_frame["subject_id"].duplicated().any():
            raise ValueError("Reused PulseDB selection contains duplicate subjects")
        selected = selected_frame.to_dict(orient="records")
        subject_limit = len(selected)
        observed_counts = selected_frame["selected_segment_indices"].map(
            lambda value: len(json.loads(value))
        )
        if observed_counts.nunique() != 1:
            raise ValueError("Reused PulseDB selection has inconsistent segment counts")
        segments_per_subject = int(observed_counts.iloc[0])
        selection_path = (
            Path(subject_selection_csv)
            if subject_selection_csv
            else Path(output_csv).with_name(f"{Path(output_csv).stem}_subject_selection.csv")
        )
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selected_frame.to_csv(selection_path, index=False)
    elif subject_limit is not None:
        summaries = [
            _summary_from_npz(path, cohort)
            if path.suffix.lower() == ".npz"
            else _summary_from_hdf5(path, cohort)
            for path in paths
        ]
        selected, eligible_subjects = _stratified_subject_sample(
            summaries, subject_limit, int(segments_per_subject), selection_seed
        )
        del summaries
        selection_path = Path(subject_selection_csv) if subject_selection_csv else Path(
            output_csv
        ).with_name(f"{Path(output_csv).stem}_subject_selection.csv")
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection_columns = [
            "subject_id",
            "waveform_path",
            "age",
            "sex",
            "subject_median_sbp",
            "subject_median_dbp",
            "eligible_segments",
            "age_bin",
            "sbp_bin",
            "dbp_bin",
            "stratum",
            "selection_seed",
            "selected_segment_indices",
        ]
        pd.DataFrame(selected)[selection_columns].to_csv(selection_path, index=False)

    if subject_limit is not None:
        selected_by_path = {
            Path(row["waveform_path"]): json.loads(row["selected_segment_indices"])
            for row in selected
        }
        paths = sorted(selected_by_path)

    records: list[dict] = []
    for path in paths:
        indices = selected_by_path[path] if selected_by_path is not None else None
        if path.suffix.lower() == ".npz":
            records.extend(_records_from_npz(path, cohort, sampling_rate_hz, indices))
        else:
            records.extend(_records_from_hdf5(path, cohort, sampling_rate_hz, indices))
    data = canonicalize_manifest(pd.DataFrame.from_records(records))
    validate_pulsedb_raw_waveform_contract(data)
    data["dataset_version"] = str(dataset_version)
    height = pd.to_numeric(data["height_cm"], errors="coerce")
    weight = pd.to_numeric(data["weight_kg"], errors="coerce")
    calculated_bmi = weight / (height / 100.0) ** 2
    data["bmi"] = pd.to_numeric(data["bmi"], errors="coerce").fillna(calculated_bmi)
    cache_path: Path | None = None
    if waveform_cache_hdf5 is not None:
        if subject_limit is None:
            raise ValueError("waveform_cache_hdf5 requires a sampled PulseDB manifest")
        cache_path = Path(waveform_cache_hdf5)
        data = _write_waveform_cache(data, cache_path)
        validate_pulsedb_raw_waveform_contract(data)
    output_csv = Path(output_csv)
    provenance_json = Path(provenance_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    provenance_json.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_csv, index=False)
    provenance = {
        "contract_version": CONTRACT_VERSION,
        "dataset": "PulseDB-Vital" if cohort == "vital" else "PulseDB-MIMIC",
        "dataset_version": str(dataset_version),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(raw_root.resolve()),
        "source_files": len(paths),
        "subjects": int(data["subject_id"].nunique()),
        "segments": len(data),
        "sample_rate_hz": sampling_rate_hz,
        "manifest_sha256": file_digest(output_csv),
        "full_source_files": full_source_files,
        "eligible_subjects": eligible_subjects,
        "subject_limit": subject_limit,
        "segments_per_subject": segments_per_subject,
        "selection_seed": selection_seed if subject_limit is not None else None,
        "sampling_strategy": (
            "preserved selected subjects and segment indices from "
            f"{Path(reuse_subject_selection_csv).resolve()}"
            if reuse_subject_selection_csv is not None
            else "proportional joint stratification by sex and age/SBP/DBP tertiles; "
            "uniformly spaced eligible segments"
            if subject_limit is not None
            else "all source subjects and segments"
        ),
        "raw_waveform_field": "PPG_Record",
        "rejected_waveform_fields": ["PPG_Record_F", "PPG_F"],
        "subject_selection_csv": str(selection_path.resolve()) if selection_path else None,
        "subject_selection_sha256": file_digest(selection_path) if selection_path else None,
        "waveform_cache_hdf5": str(cache_path.resolve()) if cache_path else None,
        "waveform_cache_sha256": file_digest(cache_path) if cache_path else None,
    }
    provenance_json.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data, provenance
