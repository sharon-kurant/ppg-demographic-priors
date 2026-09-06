"""Shared segment-level data contract for the BP benchmark."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppg_bp_incremental.data.ppgbp import load_waveform


CORE_COLUMNS = (
    "dataset",
    "source",
    "subject_id",
    "measurement_id",
    "segment_id",
    "waveform_path",
    "sample_rate_hz",
    "n_samples",
    "sbp",
    "dbp",
    "age",
    "sex",
)
OPTIONAL_DEMOGRAPHICS = ("bmi", "height_cm", "weight_kg")


def _canonical_sex(value: Any) -> str | float:
    if pd.isna(value):
        return np.nan
    token = str(value).strip().lower()
    mapping = {
        "f": "female",
        "female": "female",
        "0": "female",
        "0.0": "female",
        "m": "male",
        "male": "male",
        "1": "male",
        "1.0": "male",
    }
    return mapping.get(token, token)


def canonicalize_manifest(data: pd.DataFrame) -> pd.DataFrame:
    """Upgrade a prepared manifest to the common benchmark schema."""
    data = data.copy()
    if "measurement_id" not in data and "subject_id" in data:
        # PPG-BP's three recordings share one cuff measurement.
        data["measurement_id"] = data["subject_id"].astype(str)
    if "waveform_format" not in data and "waveform_path" in data:
        data["waveform_format"] = data["waveform_path"].map(
            lambda value: Path(str(value)).suffix.lower().lstrip(".") or "text"
        )
    for column in OPTIONAL_DEMOGRAPHICS:
        if column not in data:
            data[column] = np.nan
    height = pd.to_numeric(data["height_cm"], errors="coerce")
    weight = pd.to_numeric(data["weight_kg"], errors="coerce")
    calculated_bmi = weight / (height / 100.0) ** 2
    supplied_bmi = pd.to_numeric(data["bmi"], errors="coerce")
    data["bmi"] = supplied_bmi.fillna(calculated_bmi)
    if "valid_for_analysis" not in data:
        data["valid_for_analysis"] = True
    if "sex" in data:
        data["sex"] = data["sex"].map(_canonical_sex)
    return data


def validate_benchmark_manifest(
    data: pd.DataFrame,
    require_demographics: bool = False,
) -> None:
    missing = sorted(set(CORE_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"Benchmark manifest is missing columns: {missing}")
    if data.empty:
        raise ValueError("Benchmark manifest contains no rows")
    if data["segment_id"].duplicated().any():
        raise ValueError("Benchmark segment identifiers must be unique")
    if data[["subject_id", "segment_id", "waveform_path"]].isna().any().any():
        raise ValueError("Subject, segment, and waveform identifiers cannot be missing")
    for column in ("sample_rate_hz", "n_samples", "sbp", "dbp", "age", "bmi"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    required_numeric = ["sample_rate_hz", "n_samples", "sbp", "dbp"]
    if require_demographics:
        required_numeric.append("age")
    if data[required_numeric].isna().any().any():
        bad = data[required_numeric].columns[data[required_numeric].isna().any()].tolist()
        raise ValueError(f"Required numeric fields contain missing values: {bad}")
    if require_demographics and data["sex"].isna().any():
        raise ValueError("Primary benchmark requires age and sex for every row")
    known_sex = data["sex"].dropna()
    if not set(known_sex).issubset({"female", "male"}):
        raise ValueError("Known benchmark sex values must be female or male")
    if (data["sample_rate_hz"] <= 0).any() or (data["n_samples"] < 2).any():
        raise ValueError("Sampling rates and waveform lengths must be positive")
    if (data["sbp"] <= data["dbp"]).any():
        raise ValueError("Every SBP label must be greater than its DBP label")


def load_benchmark_manifest(
    path: str | Path,
    require_demographics: bool = False,
) -> pd.DataFrame:
    data = canonicalize_manifest(pd.read_csv(path))
    valid = data["valid_for_analysis"]
    if valid.dtype != bool:
        valid = valid.astype(str).str.lower().eq("true")
    data = data[valid].copy()
    validate_benchmark_manifest(data, require_demographics=require_demographics)
    if data["dataset"].astype(str).str.startswith("PulseDB").all():
        from ppg_bp_incremental.data.pulsedb import (
            validate_pulsedb_raw_waveform_contract,
        )

        validate_pulsedb_raw_waveform_contract(data)
    return data.sort_values(["subject_id", "segment_id"], key=lambda x: x.astype(str)).reset_index(drop=True)


def available_demographic_fields(data: pd.DataFrame) -> tuple[str, ...]:
    """Select every cohort-available field from age, sex, and BMI.

    A field is omitted only when it is entirely unavailable. Sporadic missing
    values are retained for train-fold-only imputation by the estimator.
    """

    fields = []
    for column in ("age", "sex", "bmi"):
        if column in data and data[column].notna().any():
            fields.append(column)
    return tuple(fields)


@lru_cache(maxsize=16)
def _open_hdf5(path: str):
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - dependency error is explicit
        raise ImportError("Reading PulseDB .mat files requires h5py") from error
    return h5py.File(path, "r")


def _hdf5_waveform(row: pd.Series) -> np.ndarray:
    handle = _open_hdf5(str(Path(row["waveform_path"]).resolve()))
    node = handle[str(row["waveform_dataset"])]
    index = int(row.get("waveform_index", 0))
    if getattr(node.dtype, "kind", "") == "O":
        reference = np.asarray(node).reshape(-1)[index]
        return np.asarray(handle[reference], dtype=np.float64).reshape(-1)
    axis = int(row.get("waveform_axis", 0))
    if node.ndim == 1:
        values = node[...]
    elif axis == 0:
        values = node[index, ...]
    else:
        values = node[..., index]
    return np.asarray(values, dtype=np.float64).reshape(-1)


def load_record_waveform(row: pd.Series | dict[str, Any]) -> np.ndarray:
    """Load one waveform referenced by a common manifest row."""
    row = pd.Series(row)
    waveform_format = str(row.get("waveform_format", "")).lower()
    path = Path(str(row["waveform_path"]))
    if waveform_format == "wfdb16":
        from ppg_bp_incremental.data.butppg import load_wfdb16_waveform

        signal = load_wfdb16_waveform(row)
    elif waveform_format in {"mat", "h5", "hdf5"} and "waveform_dataset" in row:
        signal = _hdf5_waveform(row)
    elif waveform_format == "npz":
        archive = np.load(path, allow_pickle=False)
        key = str(row.get("waveform_dataset", "ppg"))
        values = np.asarray(archive[key])
        index = int(row.get("waveform_index", 0))
        axis = int(row.get("waveform_axis", 0))
        signal = values[index] if axis == 0 else values[..., index]
    else:
        signal = load_waveform(path)
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if len(signal) < 2 or not np.isfinite(signal).all():
        raise ValueError(f"Invalid waveform for segment {row.get('segment_id')}")
    return signal


def capped_subject_rows(
    data: pd.DataFrame,
    maximum_segments: int,
    seed: int,
    epoch: int | None = None,
) -> pd.DataFrame:
    """Select at most N rows per subject without allowing high-volume dominance."""
    if maximum_segments < 1:
        raise ValueError("maximum_segments must be positive")
    rng = np.random.default_rng(seed if epoch is None else seed + 10_000 * epoch)
    selected: list[pd.DataFrame] = []
    for _, group in data.groupby("subject_id", sort=True):
        if len(group) <= maximum_segments:
            selected.append(group)
        else:
            positions = np.sort(rng.choice(len(group), maximum_segments, replace=False))
            selected.append(group.iloc[positions])
    return pd.concat(selected, ignore_index=True).sort_values(
        ["subject_id", "segment_id"], key=lambda x: x.astype(str)
    ).reset_index(drop=True)
