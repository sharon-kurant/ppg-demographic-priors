"""BUT PPG v2.0 ingestion for the blood-pressure benchmark."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from ppg_bp_incremental.data.benchmark import canonicalize_manifest
from ppg_bp_incremental.data.ppgbp import file_digest
from ppg_bp_incremental.models.contracts import CONTRACT_VERSION


BUTPPG_VERSION = "2.0.0"
BUTPPG_DOI = "10.13026/tn53-8153"
BUTPPG_SAMPLE_RATE_HZ = 30
BUTPPG_N_SAMPLES = 300
BUTPPG_SOURCE_URL = "https://physionet.org/content/butppg/2.0.0/"
BUTPPG_ACQUISITION_SEGMENTS = 54
# In the released acquisition protocol, the first non-rest (higher-pressure)
# pair begins after fourteen rest segments.  This marker lets us recover the
# second acquisition when an otherwise 54-segment session was truncated to 42
# segments (subjects 119, 121, and 126 in v2.0.0).
BUTPPG_PRESSURE_MOTION_POSITION = 15
_GAIN_PATTERN = re.compile(
    r"^(?P<gain>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"(?:\((?P<baseline>[+-]?\d+)\))?"
    r"(?:/(?P<units>.*))?$"
)


def _motion_token(value: Any) -> str:
    """Normalize a source Motion value without interpreting any BP label."""
    if pd.isna(value):
        return ""
    token = str(value).strip()
    return token[:-2] if token.endswith(".0") else token


def _derive_acquisition_sessions(records: pd.DataFrame) -> pd.DataFrame:
    """Derive label-free BUT PPG acquisition-session identities.

    PhysioNet defines the first three ID digits as the subject and the final
    three as that subject's measurement order.  A normal acquisition contains
    54 ten-second records.  Several released acquisitions contain only the
    first 42 records; for those cases, a fixed 54-record modulo would put the
    boundary in the wrong place.  We therefore locate protocol resets from the
    source ``Motion`` field: motion code 1 starts at within-acquisition position
    15.  SBP and DBP are deliberately neither accepted nor inspected here.
    """
    required = {"ID", "subject_id", "Motion"}
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(
            f"BUT PPG session derivation is missing source fields: {missing}"
        )

    ids = records["ID"].astype("string").str.strip().str.zfill(6)
    malformed = ~ids.str.fullmatch(r"\d{6}", na=False)
    if malformed.any():
        raise ValueError(f"Malformed BUT PPG record ID: {ids[malformed].iloc[0]!r}")
    source_subject = "butppg-" + ids.str[:3]
    supplied_subject = records["subject_id"].astype("string")
    if not source_subject.equals(supplied_subject):
        raise ValueError("BUT PPG subject_id disagrees with the source record ID")

    source_order = pd.to_numeric(ids.str[3:], errors="raise").astype(int)
    derived = pd.DataFrame(index=records.index)
    derived["source_record_order"] = source_order
    derived["acquisition_session_index"] = pd.Series(
        index=records.index, dtype="int64"
    )
    derived["acquisition_session_position"] = pd.Series(
        index=records.index, dtype="int64"
    )
    derived["acquisition_session_id"] = pd.Series(
        index=records.index, dtype="string"
    )

    working = records.assign(
        _source_record_order=source_order,
        _motion=records["Motion"].map(_motion_token),
    )
    for subject_id, subject in working.groupby("subject_id", sort=True):
        subject = subject.sort_values("_source_record_order")
        orders = subject["_source_record_order"].to_numpy(dtype=int)
        if pd.Series(orders).duplicated().any():
            raise ValueError(f"Duplicate BUT PPG source order for {subject_id}")

        motion = subject["_motion"].to_numpy(dtype=str)
        pressure_onsets = [
            int(orders[position])
            for position in range(len(subject))
            if motion[position] == "1"
            and (position == 0 or motion[position - 1] != "1")
        ]
        first_order = int(orders[0])
        protocol_starts = sorted(
            {
                onset - (BUTPPG_PRESSURE_MOTION_POSITION - 1)
                for onset in pressure_onsets
                if onset - (BUTPPG_PRESSURE_MOTION_POSITION - 1) >= first_order
            }
        )
        if first_order not in protocol_starts:
            protocol_starts.insert(0, first_order)

        # More than one protocol marker is authoritative for truncated
        # acquisitions.  With only the initial marker (or no marker), fall back
        # to the release's nominal 54-record acquisition length.
        if len(protocol_starts) > 1:
            starts = protocol_starts
        else:
            starts = list(
                range(
                    first_order,
                    int(orders[-1]) + 1,
                    BUTPPG_ACQUISITION_SEGMENTS,
                )
            )
        while int(orders[-1]) - starts[-1] >= BUTPPG_ACQUISITION_SEGMENTS:
            starts.append(starts[-1] + BUTPPG_ACQUISITION_SEGMENTS)

        session_indices = np.searchsorted(starts, orders, side="right")
        start_by_row = np.asarray(starts, dtype=int)[session_indices - 1]
        session_positions = orders - start_by_row + 1
        if np.any(session_positions < 1) or np.any(
            session_positions > BUTPPG_ACQUISITION_SEGMENTS
        ):
            raise ValueError(
                f"Could not map {subject_id} to <=54-record acquisition sessions"
            )
        for onset in pressure_onsets:
            onset_position = int(
                onset
                - np.asarray(starts, dtype=int)[
                    np.searchsorted(starts, onset, side="right") - 1
                ]
                + 1
            )
            if onset_position != BUTPPG_PRESSURE_MOTION_POSITION:
                raise ValueError(
                    f"BUT PPG motion protocol reset is inconsistent for {subject_id}"
                )

        derived.loc[subject.index, "acquisition_session_index"] = session_indices
        derived.loc[subject.index, "acquisition_session_position"] = session_positions
        derived.loc[subject.index, "acquisition_session_id"] = [
            f"{subject_id}-session-{index:02d}" for index in session_indices
        ]

    derived["acquisition_session_index"] = derived[
        "acquisition_session_index"
    ].astype(int)
    derived["acquisition_session_position"] = derived[
        "acquisition_session_position"
    ].astype(int)
    return derived


def _locate_dataset_root(raw_root: str | Path) -> Path:
    root = Path(raw_root)
    candidates = sorted(root.rglob("subject-info.csv"))
    valid = [
        path.parent
        for path in candidates
        if (path.parent / "quality-hr-ann.csv").is_file()
    ]
    if len(valid) != 1:
        raise ValueError(
            f"Expected one BUT PPG v2.0 root beneath {root}, found {len(valid)}"
        )
    return valid[0]


def _parse_wfdb_header(path: Path, channel_name: str = "PPG_R") -> dict[str, Any]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"Empty WFDB header: {path}")
    record = lines[0].split()
    if len(record) < 4:
        raise ValueError(f"Malformed WFDB record line in {path}")
    n_signals = int(record[1])
    sample_rate = float(record[2].split("/")[0])
    n_samples = int(record[3])
    if len(lines) < n_signals + 1:
        raise ValueError(f"WFDB header declares {n_signals} signals but is truncated: {path}")

    signals = []
    for index, line in enumerate(lines[1 : n_signals + 1]):
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(f"Malformed WFDB signal line in {path}: {line}")
        format_token = fields[1]
        if format_token != "16":
            raise ValueError(
                f"BUT PPG loader supports WFDB format 16, observed {format_token!r}"
            )
        gain_match = _GAIN_PATTERN.match(fields[2])
        if gain_match is None:
            raise ValueError(f"Cannot parse WFDB gain token {fields[2]!r} in {path}")
        gain = float(gain_match.group("gain"))
        baseline_token = gain_match.group("baseline")
        baseline = (
            float(baseline_token)
            if baseline_token is not None
            else float(fields[4] if len(fields) > 4 else 0)
        )
        signals.append(
            {
                "waveform_path": str((path.parent / fields[0]).resolve()),
                "waveform_n_signals": n_signals,
                "waveform_signal_index": index,
                "waveform_gain": gain,
                "waveform_baseline": baseline,
                "waveform_units": gain_match.group("units") or "",
                "signal_name": fields[-1],
            }
        )
    matching = [signal for signal in signals if signal["signal_name"] == channel_name]
    if len(matching) != 1:
        raise ValueError(
            f"Expected one {channel_name} signal in {path}, found {len(matching)}"
        )
    return {
        **matching[0],
        "sample_rate_hz": sample_rate,
        "n_samples": n_samples,
    }


def load_wfdb16_waveform(row: pd.Series | dict[str, Any]) -> np.ndarray:
    """Read one interleaved WFDB format-16 channel using manifest metadata."""
    row = pd.Series(row)
    path = Path(str(row["waveform_path"]))
    n_signals = int(row["waveform_n_signals"])
    signal_index = int(row["waveform_signal_index"])
    expected_samples = int(row["n_samples"])
    values = np.fromfile(path, dtype="<i2")
    if values.size % n_signals:
        raise ValueError(f"WFDB sample count is not divisible by {n_signals}: {path}")
    digital = values.reshape(-1, n_signals)
    if digital.shape[0] != expected_samples:
        raise ValueError(
            f"WFDB sample count {digital.shape[0]} does not match "
            f"header value {expected_samples}: {path}"
        )
    gain = float(row["waveform_gain"])
    baseline = float(row["waveform_baseline"])
    if not np.isfinite(gain) or gain == 0:
        raise ValueError(f"Invalid WFDB gain for {path}: {gain}")
    return (digital[:, signal_index].astype(np.float64) - baseline) / gain


def _balanced_subject_cap(
    data: pd.DataFrame,
    maximum_segments: int,
    seed: int,
) -> pd.DataFrame:
    """Cap each subject while retaining source acquisition sessions when possible."""
    if maximum_segments < 1:
        raise ValueError("maximum_segments must be positive")
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    for _, subject in data.groupby("subject_id", sort=True):
        if len(subject) <= maximum_segments:
            selected_indices.extend(subject.index.tolist())
            continue
        queues: list[list[int]] = []
        for _, measurement in subject.groupby("measurement_id", sort=True):
            indices = measurement.index.to_numpy()
            queues.append(rng.permutation(indices).tolist())
        chosen: list[int] = []
        while len(chosen) < maximum_segments and any(queues):
            for queue in queues:
                if queue and len(chosen) < maximum_segments:
                    chosen.append(queue.pop())
        selected_indices.extend(chosen)
    return data.loc[selected_indices].sort_values(
        ["subject_id", "segment_id"], key=lambda value: value.astype(str)
    ).reset_index(drop=True)


def _measurement_identity_sha256(data: pd.DataFrame) -> str:
    """Hash the label-free segment-to-acquisition mapping."""
    columns = [
        "subject_id",
        "measurement_id",
        "segment_id",
        "source_record_order",
        "acquisition_session_index",
        "acquisition_session_position",
    ]
    canonical = data.loc[:, columns].sort_values(
        ["subject_id", "source_record_order"], kind="stable"
    )
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return sha256(payload).hexdigest()


def prepare_butppg(
    raw_root: str | Path,
    output_csv: str | Path,
    qc_manifest: str | Path,
    provenance_json: str | Path,
    source_archive: str | Path | None = None,
    maximum_segments_per_subject: int = 20,
    selection_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Prepare quality-approved BUT PPG records with raw-mmHg SBP/DBP labels."""
    root = _locate_dataset_root(raw_root)
    subject_info_path = root / "subject-info.csv"
    quality_path = root / "quality-hr-ann.csv"
    subject_info = pd.read_csv(subject_info_path, encoding="utf-8-sig")
    quality = pd.read_csv(quality_path, encoding="utf-8-sig")
    subject_info["ID"] = subject_info["ID"].astype(str).str.zfill(6)
    quality["ID"] = quality["ID"].astype(str).str.zfill(6)
    records = subject_info.merge(
        quality[["ID", "Quality", "HR"]], on="ID", how="left", validate="one_to_one"
    )

    bp = records["Blood pressure [mmHg]"].astype("string").str.extract(
        r"^\s*(?P<sbp>\d+(?:\.\d+)?)\s*/\s*(?P<dbp>\d+(?:\.\d+)?)\s*$"
    )
    records["sbp"] = pd.to_numeric(bp["sbp"], errors="coerce")
    records["dbp"] = pd.to_numeric(bp["dbp"], errors="coerce")
    records["age"] = pd.to_numeric(records["Age [years]"], errors="coerce")
    height_cm = pd.to_numeric(records["Height [cm]"], errors="coerce")
    weight_kg = pd.to_numeric(records["Weight [kg]"], errors="coerce")
    records["bmi"] = weight_kg / np.square(height_cm / 100.0)
    records["sex"] = records["Gender"].astype("string").str.strip().str.lower().map(
        {"f": "female", "m": "male"}
    )
    records["subject_id"] = "butppg-" + records["ID"].str[:3]
    session_fields = _derive_acquisition_sessions(records)
    records = pd.concat([records, session_fields], axis=1)
    # A benchmark measurement is one source acquisition session.  The ID is
    # intentionally independent of the BP annotation.
    records["measurement_id"] = records["acquisition_session_id"]
    records["segment_id"] = "butppg-" + records["ID"]
    records["header_path"] = records["ID"].map(
        lambda record_id: root / record_id / f"{record_id}_PPG.hea"
    )
    records["quality_approved"] = pd.to_numeric(
        records["Quality"], errors="coerce"
    ).eq(1)
    records["valid_bp"] = (
        records[["sbp", "dbp"]].notna().all(axis=1) & records["sbp"].gt(records["dbp"])
    )
    records["valid_demographics"] = records["age"].notna() & records["sex"].notna()
    records["source_files_present"] = records["header_path"].map(Path.is_file)
    records["valid_for_analysis"] = (
        records["quality_approved"]
        & records["valid_bp"]
        & records["source_files_present"]
    )
    records["exclusion_reason"] = records.apply(
        lambda row: ";".join(
            reason
            for condition, reason in (
                (not row["quality_approved"], "official_quality_zero"),
                (not row["valid_bp"], "missing_or_invalid_bp"),
                (not row["source_files_present"], "missing_ppg_header"),
            )
            if condition
        ),
        axis=1,
    )

    eligible = records[records["valid_for_analysis"]].copy()
    headers = {
        path: _parse_wfdb_header(path)
        for path in sorted(set(eligible["header_path"]))
    }
    header_fields = pd.DataFrame.from_records(
        [headers[path] for path in eligible["header_path"]], index=eligible.index
    )
    eligible = pd.concat([eligible, header_fields], axis=1)
    if not eligible["sample_rate_hz"].eq(BUTPPG_SAMPLE_RATE_HZ).all():
        raise ValueError("BUT PPG PPG sampling rate is not uniformly 30 Hz")
    if not eligible["n_samples"].eq(BUTPPG_N_SAMPLES).all():
        raise ValueError("BUT PPG records are not uniformly 10 seconds")
    if not eligible["waveform_path"].map(lambda value: Path(value).is_file()).all():
        raise FileNotFoundError("One or more BUT PPG PPG data files are missing")

    eligible = _balanced_subject_cap(
        eligible, maximum_segments_per_subject, selection_seed
    )
    eligible["waveform_sha256"] = eligible["waveform_path"].map(
        lambda value: file_digest(Path(value))
    )
    eligible.insert(0, "dataset", "BUT PPG")
    eligible.insert(1, "dataset_version", BUTPPG_VERSION)
    eligible.insert(2, "source", "BUT PPG")
    eligible["waveform_format"] = "wfdb16"
    eligible["duration_seconds"] = (
        eligible["n_samples"] / eligible["sample_rate_hz"]
    )
    eligible["measurement_site"] = eligible["Ear/finger"].map({0: "ear", 1: "finger"})
    eligible["motion"] = eligible["Motion"].astype(str)
    eligible["heart_rate_bpm"] = pd.to_numeric(eligible["HR"], errors="coerce")
    eligible["signal_quality"] = "official_good_for_hr"
    eligible["pretraining_overlap_group"] = "butppg"
    # Height and weight are intentionally not carried into the benchmark manifest.
    harmonized = canonicalize_manifest(
        eligible[
            [
                "dataset",
                "dataset_version",
                "source",
                "subject_id",
                "measurement_id",
                "segment_id",
                "source_record_order",
                "acquisition_session_id",
                "acquisition_session_index",
                "acquisition_session_position",
                "waveform_path",
                "waveform_sha256",
                "waveform_format",
                "waveform_n_signals",
                "waveform_signal_index",
                "waveform_gain",
                "waveform_baseline",
                "waveform_units",
                "sample_rate_hz",
                "n_samples",
                "duration_seconds",
                "sbp",
                "dbp",
                "age",
                "sex",
                "bmi",
                "measurement_site",
                "motion",
                "heart_rate_bpm",
                "signal_quality",
                "pretraining_overlap_group",
                "valid_for_analysis",
            ]
        ]
    )
    if not harmonized["measurement_id"].astype(str).str.fullmatch(
        r"butppg-\d{3}-session-\d{2}"
    ).all():
        raise RuntimeError("BUT PPG measurement IDs violate the label-free session contract")
    if (
        harmonized.groupby("measurement_id")["subject_id"].nunique().max()
        != 1
    ):
        raise RuntimeError("A BUT PPG acquisition session spans multiple participants")
    identity_sha256 = _measurement_identity_sha256(harmonized)

    qc = records[
        [
            "ID",
            "subject_id",
            "measurement_id",
            "segment_id",
            "source_record_order",
            "acquisition_session_id",
            "acquisition_session_index",
            "acquisition_session_position",
            "sbp",
            "dbp",
            "age",
            "sex",
            "Quality",
            "quality_approved",
            "valid_bp",
            "valid_demographics",
            "source_files_present",
            "valid_for_analysis",
            "exclusion_reason",
        ]
    ].copy()
    output_csv = Path(output_csv)
    qc_manifest = Path(qc_manifest)
    provenance_json = Path(provenance_json)
    for path in (output_csv, qc_manifest, provenance_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    harmonized.to_csv(output_csv, index=False)
    qc.to_csv(qc_manifest, index=False)

    provenance: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "dataset": "BUT PPG",
        "dataset_version": BUTPPG_VERSION,
        "doi": BUTPPG_DOI,
        "source_url": BUTPPG_SOURCE_URL,
        "license": "CC BY 4.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject_info_sha256": file_digest(subject_info_path),
        "quality_annotations_sha256": file_digest(quality_path),
        "source_archive": str(Path(source_archive).resolve()) if source_archive else None,
        "source_archive_sha256": (
            file_digest(Path(source_archive)) if source_archive else None
        ),
        "ppg_channel": "PPG_R",
        "sampling_rate_hz": BUTPPG_SAMPLE_RATE_HZ,
        "record_duration_seconds": 10,
        "eligibility": {
            "raw_records": int(len(records)),
            "records_with_bp": int(records["valid_bp"].sum()),
            "official_quality_approved_records": int(records["quality_approved"].sum()),
            "eligible_before_cap": int(records["valid_for_analysis"].sum()),
            "eligible_subjects": int(
                records.loc[records["valid_for_analysis"], "subject_id"].nunique()
            ),
        },
        "selection": {
            "maximum_segments_per_subject": maximum_segments_per_subject,
            "selection_seed": selection_seed,
            "selected_records": int(len(harmonized)),
            "selected_subjects": int(harmonized["subject_id"].nunique()),
            "selected_acquisition_sessions": int(
                harmonized["measurement_id"].nunique()
            ),
            "measurement_balanced_within_subject": True,
            "source_acquisition_session_balanced_within_subject": True,
        },
        "measurement_grouping": {
            "label_independent": True,
            "measurement_id_definition": (
                "subject identifier plus chronological source acquisition-session index"
            ),
            "source_fields": [
                "ID (first three digits subject; last three measurement order)",
                "Motion (protocol-reset marker)",
            ],
            "nominal_segments_per_acquisition": BUTPPG_ACQUISITION_SEGMENTS,
            "pressure_motion_protocol_position": BUTPPG_PRESSURE_MOTION_POSITION,
            "bp_fields_used": [],
            "selected_identity_sha256": identity_sha256,
        },
        "available_demographics": ["age", "sex", "bmi"],
        "demographic_missingness_policy": (
            "train-fold median imputation plus missing indicators for age/BMI; "
            "explicit unknown category for sex"
        ),
        "bmi_formula": "weight_kg / (height_cm / 100)^2",
        "excluded_direct_inputs": ["height_cm", "weight_kg"],
        "quality_policy": (
            "Require official Quality=1; the release describes Quality=0 PPG "
            "as unsuitable for analysis."
        ),
    }
    provenance_json.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return harmonized, qc, provenance
