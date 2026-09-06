"""Canonical ingestion for version 5 of the PPG-BP dataset."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import md5, sha256
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from ppg_bp_incremental.models.contracts import CONTRACT_VERSION


PPGBP_SAMPLE_RATE_HZ = 1000
PPGBP_DOI = "10.6084/m9.figshare.5459299.v5"
PPGBP_ARCHIVE_MD5 = "3b8eae44f45799aeb3c5f55af8589828"
_WAVEFORM_PATTERN = re.compile(r"^(?P<subject>\d+)_(?P<recording>[123])\.txt$")

_METADATA_COLUMNS = {
    "subject_ID": "source_subject_id",
    "Sex(M/F)": "sex",
    "Age(year)": "age",
    "Height(cm)": "height_cm",
    "Weight(kg)": "weight_kg",
    "Systolic Blood Pressure(mmHg)": "sbp",
    "Diastolic Blood Pressure(mmHg)": "dbp",
    "Heart Rate(b/m)": "heart_rate_bpm",
    "BMI(kg/m^2)": "bmi_reported",
    "Hypertension": "hypertension_category",
    "Diabetes": "diabetes",
    "cerebral infarction": "cerebral_infarction",
    "cerebrovascular disease": "cerebrovascular_disease",
}


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    hasher = sha256() if algorithm == "sha256" else md5()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _locate_dataset_root(raw_root: str | Path) -> Path:
    raw_root = Path(raw_root)
    candidates = (raw_root, raw_root / "Data File")
    for candidate in candidates:
        if (
            (candidate / "PPG-BP dataset.xlsx").is_file()
            and (candidate / "0_subject").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not find 'PPG-BP dataset.xlsx' and '0_subject' beneath {raw_root}"
    )


def load_waveform(path: str | Path) -> np.ndarray:
    """Read a canonical text or NumPy waveform without enabling pickle."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        signal = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).squeeze()
        if signal.ndim != 1 or signal.size == 0:
            raise ValueError(f"Waveform is not a non-empty vector: {path}")
        return signal
    text = path.read_text(encoding="utf-8")
    signal = np.fromstring(text, sep="\t", dtype=np.float64)
    if signal.size == 0:
        raise ValueError(f"Waveform contains no numeric samples: {path}")
    return signal


def _read_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_excel(path, sheet_name="cardiovascular dataset", header=1)
    missing = sorted(set(_METADATA_COLUMNS).difference(metadata.columns))
    if missing:
        raise ValueError(f"PPG-BP workbook is missing columns: {missing}")
    metadata = metadata[list(_METADATA_COLUMNS)].rename(columns=_METADATA_COLUMNS)
    metadata["source_subject_id"] = pd.to_numeric(
        metadata["source_subject_id"], errors="raise"
    ).astype(int)
    if metadata["source_subject_id"].duplicated().any():
        raise ValueError("PPG-BP metadata contains duplicate subject identifiers")
    metadata["subject_id"] = metadata["source_subject_id"].map(
        lambda value: f"ppgbp-{value:04d}"
    )
    metadata["sex"] = metadata["sex"].astype("string").str.strip().str.lower()
    unexpected_sex = set(metadata["sex"].dropna()) - {"female", "male"}
    if unexpected_sex:
        raise ValueError(f"Unexpected PPG-BP sex values: {sorted(unexpected_sex)}")

    numeric_columns = (
        "age",
        "height_cm",
        "weight_kg",
        "sbp",
        "dbp",
        "heart_rate_bpm",
        "bmi_reported",
    )
    for column in numeric_columns:
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
    metadata["bmi_calculated"] = metadata["weight_kg"] / (
        metadata["height_cm"] / 100
    ) ** 2
    metadata["bmi"] = metadata["bmi_reported"].fillna(
        metadata["bmi_calculated"]
    )
    metadata["bmi_abs_difference"] = (
        metadata["bmi_reported"] - metadata["bmi_calculated"]
    ).abs()
    return metadata


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _waveform_record(path: Path) -> dict[str, Any]:
    match = _WAVEFORM_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected waveform filename: {path.name}")
    signal = load_waveform(path)
    finite = np.isfinite(signal)
    finite_signal = signal[finite]
    finite_fraction = float(np.mean(finite))
    signal_range = float(np.ptp(finite_signal)) if finite_signal.size else np.nan
    signal_std = float(np.std(finite_signal)) if finite_signal.size else np.nan
    if len(signal) >= 2:
        flatline_fraction = float(np.mean(np.diff(signal) == 0))
    else:
        flatline_fraction = 1.0

    reasons = []
    if len(signal) < PPGBP_SAMPLE_RATE_HZ:
        reasons.append("shorter_than_1_second")
    if finite_fraction < 1:
        reasons.append("non_finite_samples")
    if not np.isfinite(signal_range) or signal_range <= 0:
        reasons.append("constant_or_invalid_signal")

    source_subject_id = int(match.group("subject"))
    recording = int(match.group("recording"))
    return {
        "source_subject_id": source_subject_id,
        "subject_id": f"ppgbp-{source_subject_id:04d}",
        "recording_index": recording,
        "segment_id": f"ppgbp-{source_subject_id:04d}-{recording}",
        "waveform_path": _portable_path(path),
        "waveform_sha256": file_digest(path),
        "sample_rate_hz": PPGBP_SAMPLE_RATE_HZ,
        "n_samples": len(signal),
        "duration_seconds": len(signal) / PPGBP_SAMPLE_RATE_HZ,
        "finite_fraction": finite_fraction,
        "signal_mean": float(np.mean(finite_signal)) if finite_signal.size else np.nan,
        "signal_std": signal_std,
        "signal_min": float(np.min(finite_signal)) if finite_signal.size else np.nan,
        "signal_max": float(np.max(finite_signal)) if finite_signal.size else np.nan,
        "signal_range": signal_range,
        "flatline_fraction": flatline_fraction,
        "valid_for_analysis": not reasons,
        "exclusion_reason": ";".join(reasons),
    }


def prepare_ppgbp(
    raw_root: str | Path,
    output_csv: str | Path,
    qc_manifest: str | Path,
    provenance_json: str | Path,
    source_archive: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Harmonize demographics and waveform metadata into segment-level rows."""
    dataset_root = _locate_dataset_root(raw_root)
    metadata_path = dataset_root / "PPG-BP dataset.xlsx"
    waveform_root = dataset_root / "0_subject"
    metadata = _read_metadata(metadata_path)

    paths = sorted(
        waveform_root.glob("*.txt"),
        key=lambda path: tuple(int(part) for part in path.stem.split("_")),
    )
    waveform_records = pd.DataFrame.from_records(
        [_waveform_record(path) for path in paths]
    )
    if waveform_records.empty:
        raise ValueError(f"No PPG-BP waveform files found beneath {waveform_root}")
    if waveform_records["segment_id"].duplicated().any():
        raise ValueError("Duplicate PPG-BP segment identifiers were generated")

    recording_counts = waveform_records.groupby("source_subject_id").size()
    if not (recording_counts == 3).all():
        bad = recording_counts[recording_counts != 3].to_dict()
        raise ValueError(f"Expected three recordings per subject; observed {bad}")
    metadata_subjects = set(metadata["source_subject_id"])
    waveform_subjects = set(waveform_records["source_subject_id"])
    if metadata_subjects != waveform_subjects:
        raise ValueError(
            "Metadata and waveform subject identifiers do not match: "
            f"metadata_only={sorted(metadata_subjects - waveform_subjects)}, "
            f"waveform_only={sorted(waveform_subjects - metadata_subjects)}"
        )

    harmonized = waveform_records.merge(
        metadata,
        on=["source_subject_id", "subject_id"],
        how="left",
        validate="many_to_one",
    )
    harmonized.insert(0, "dataset", "PPG-BP")
    harmonized.insert(1, "dataset_version", "5")
    harmonized.insert(2, "source", "PPG-BP")
    harmonized["measurement_id"] = harmonized["subject_id"]
    harmonized["waveform_format"] = "txt"
    harmonized["pretraining_overlap_group"] = "ppgbp"

    qc_columns = [
        "dataset",
        "dataset_version",
        "subject_id",
        "segment_id",
        "waveform_path",
        "waveform_sha256",
        "sample_rate_hz",
        "n_samples",
        "duration_seconds",
        "finite_fraction",
        "signal_mean",
        "signal_std",
        "signal_min",
        "signal_max",
        "signal_range",
        "flatline_fraction",
        "valid_for_analysis",
        "exclusion_reason",
    ]
    qc = harmonized[qc_columns].copy()

    output_csv = Path(output_csv)
    qc_manifest = Path(qc_manifest)
    provenance_json = Path(provenance_json)
    for path in (output_csv, qc_manifest, provenance_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    harmonized.to_csv(output_csv, index=False)
    qc.to_csv(qc_manifest, index=False)

    sample_counts = Counter(int(value) for value in harmonized["n_samples"])
    provenance: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "dataset": "PPG-BP",
        "dataset_version": 5,
        "doi": PPGBP_DOI,
        "license": "CC0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_path": _portable_path(metadata_path),
        "metadata_sha256": file_digest(metadata_path),
        "subjects": int(harmonized["subject_id"].nunique()),
        "segments": len(harmonized),
        "valid_segments": int(harmonized["valid_for_analysis"].sum()),
        "sample_rate_hz": PPGBP_SAMPLE_RATE_HZ,
        "sample_count_distribution": {
            str(key): value for key, value in sorted(sample_counts.items())
        },
        "harmonized_csv": _portable_path(output_csv),
        "harmonized_csv_sha256": file_digest(output_csv),
        "qc_manifest": _portable_path(qc_manifest),
        "qc_manifest_sha256": file_digest(qc_manifest),
    }
    if source_archive is not None:
        source_archive = Path(source_archive)
        archive_md5 = file_digest(source_archive, algorithm="md5")
        if archive_md5 != PPGBP_ARCHIVE_MD5:
            raise ValueError(
                f"PPG-BP archive checksum mismatch: expected {PPGBP_ARCHIVE_MD5}, "
                f"received {archive_md5}"
            )
        provenance["source_archive"] = _portable_path(source_archive)
        provenance["source_archive_md5"] = archive_md5
    with provenance_json.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return harmonized, qc, provenance
