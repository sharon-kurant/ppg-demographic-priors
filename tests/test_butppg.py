from pathlib import Path

import numpy as np
import pandas as pd

from ppg_bp_incremental.data.benchmark import (
    load_record_waveform,
    validate_benchmark_manifest,
)
from ppg_bp_incremental.data.butppg import (
    _derive_acquisition_sessions,
    _parse_wfdb_header,
    load_wfdb16_waveform,
    prepare_butppg,
)


def _write_record(root: Path, record_id: str, values: np.ndarray) -> None:
    record_root = root / record_id
    record_root.mkdir(parents=True)
    dat = record_root / f"{record_id}_PPG.dat"
    np.asarray(values, dtype="<i2").tofile(dat)
    (record_root / f"{record_id}_PPG.hea").write_text(
        "\n".join(
            [
                f"{record_id}_PPG 3 30 {len(values)}",
                f"{dat.name} 16 2(10)/a.u. 0 0 0 0 0 PPG_R",
                f"{dat.name} 16 4(20)/a.u. 0 0 0 0 0 PPG_G",
                f"{dat.name} 16 5(30)/a.u. 0 0 0 0 0 PPG_B",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_wfdb16_red_channel_contract(tmp_path: Path):
    values = np.array(
        [[10, 20, 30], [12, 24, 35], [14, 28, 40], [16, 32, 45]], dtype=np.int16
    )
    _write_record(tmp_path, "900001", values)
    metadata = _parse_wfdb_header(tmp_path / "900001" / "900001_PPG.hea")
    metadata["n_samples"] = 4
    signal = load_wfdb16_waveform(metadata)
    np.testing.assert_allclose(signal, [0, 1, 2, 3])


def test_prepare_butppg_filters_quality_caps_sessions_and_calculates_bmi(
    tmp_path: Path,
):
    # Entirely synthetic records use subject IDs outside the released cohort.
    rows = [
        ("901001", "M", 31, 180, 81, 0, "0", "142/88"),
        ("901002", "M", 31, 180, 81, 0, "0", "142/88"),
        # The final three ID digits are the source measurement order.  A
        # regular second acquisition begins at order 55, independent of BP.
        ("901055", "M", 31, 180, 81, 1, "0", "128/76"),
        ("901056", "M", 31, 180, 81, 1, "0", "128/76"),
        ("902001", "F", 47, 160, 64, 1, "0", "116/72"),
        ("902002", "F", 47, 160, 64, 1, "0", "116/72"),
    ]
    subject_info = pd.DataFrame(
        rows,
        columns=[
            "ID",
            "Gender",
            "Age [years]",
            "Height [cm]",
            "Weight [kg]",
            "Ear/finger",
            "Motion",
            "Blood pressure [mmHg]",
        ],
    )
    subject_info["Glycaemia [mmol/l]"] = np.nan
    subject_info["SpO2 [%]"] = 98
    quality = pd.DataFrame(
        {
            "ID": [row[0] for row in rows],
            "Quality": [1, 1, 1, 1, 1, 0],
            "HR": [70, 71, 72, 73, 74, 75],
        }
    )
    subject_info.to_csv(tmp_path / "subject-info.csv", index=False)
    quality.to_csv(tmp_path / "quality-hr-ann.csv", index=False)
    values = np.tile(np.array([[10, 20, 30]], dtype=np.int16), (300, 1))
    for record_id, *_ in rows:
        _write_record(tmp_path, record_id, values)

    manifest, qc, provenance = prepare_butppg(
        tmp_path,
        tmp_path / "prepared.csv",
        tmp_path / "qc.csv",
        tmp_path / "provenance.json",
        maximum_segments_per_subject=2,
        selection_seed=42,
    )
    validate_benchmark_manifest(manifest)
    assert len(manifest) == 3
    assert manifest["subject_id"].nunique() == 2
    assert (
        manifest.loc[manifest["subject_id"] == "butppg-901", "measurement_id"].nunique()
        == 2
    )
    assert set(
        manifest.loc[manifest["subject_id"] == "butppg-901", "measurement_id"]
    ).issubset({"butppg-901-session-01", "butppg-901-session-02"})
    assert manifest["measurement_id"].str.fullmatch(
        r"butppg-\d{3}-session-\d{2}"
    ).all()
    assert not manifest.apply(
        lambda row: f"-{int(row.sbp)}-{int(row.dbp)}" in row.measurement_id,
        axis=1,
    ).any()
    np.testing.assert_allclose(manifest["bmi"], 25.0)
    assert manifest["height_cm"].isna().all()
    assert manifest["weight_kg"].isna().all()
    assert len(qc) == 6
    assert provenance["selection"]["selected_records"] == 3
    assert provenance["measurement_grouping"]["label_independent"] is True
    assert provenance["measurement_grouping"]["bp_fields_used"] == []
    assert len(provenance["measurement_grouping"]["selected_identity_sha256"]) == 64
    assert provenance["selection"]["selected_acquisition_sessions"] == 3
    signal = load_record_waveform(manifest.iloc[0])
    assert signal.shape == (300,)
    assert np.isfinite(signal).all()

    # Changing valid BP annotations must not alter session identities or the
    # session-balanced sample.  Only the output labels may change.
    relabeled = subject_info.copy()
    subject_901 = relabeled["ID"].astype(str).str.startswith("9010")
    relabeled.loc[subject_901, "Blood pressure [mmHg]"] = [
        "151/87",
        "151/87",
        "106/65",
        "106/65",
    ]
    subject_902 = relabeled["ID"].astype(str).str.startswith("9020")
    relabeled.loc[subject_902, "Blood pressure [mmHg]"] = ["125/78", "125/78"]
    relabeled.to_csv(tmp_path / "subject-info.csv", index=False)
    relabeled_manifest, _, relabeled_provenance = prepare_butppg(
        tmp_path,
        tmp_path / "prepared-relabeled.csv",
        tmp_path / "qc-relabeled.csv",
        tmp_path / "provenance-relabeled.json",
        maximum_segments_per_subject=2,
        selection_seed=42,
    )
    assert (
        relabeled_provenance["measurement_grouping"]["selected_identity_sha256"]
        == provenance["measurement_grouping"]["selected_identity_sha256"]
    )
    identity_columns = [
        "segment_id",
        "measurement_id",
        "acquisition_session_id",
        "acquisition_session_index",
        "acquisition_session_position",
    ]
    pd.testing.assert_frame_equal(
        manifest[identity_columns].sort_values("segment_id").reset_index(drop=True),
        relabeled_manifest[identity_columns]
        .sort_values("segment_id")
        .reset_index(drop=True),
    )
    assert not manifest[["sbp", "dbp"]].equals(relabeled_manifest[["sbp", "dbp"]])


def test_acquisition_sessions_use_motion_protocol_for_truncated_first_session():
    # A 42-record first acquisition followed by a complete 54-record one is a
    # real release pattern.  The second pressure-motion pair begins at source
    # order 57, which identifies its session start at order 43.
    first_motion = ["0"] * 14 + ["1"] * 2 + ["0"] * 2 + ["2"] * 2
    first_motion += ["0"] * 2 + ["3"] * 2 + ["0"] * 2 + ["4"] * 2
    first_motion += ["0"] * 2 + ["5"] * 2 + ["0"] * 2 + ["6"] * 4
    first_motion += ["0"] * 2 + ["7"] * 2
    full_motion = first_motion + ["0"] * 12
    motion = first_motion + full_motion
    assert len(motion) == 96
    records = pd.DataFrame(
        {
            "ID": [f"990{order:03d}" for order in range(1, 97)],
            "subject_id": ["butppg-990"] * 96,
            "Motion": motion,
        }
    )

    sessions = _derive_acquisition_sessions(records)

    assert sessions.loc[41, "acquisition_session_id"] == "butppg-990-session-01"
    assert sessions.loc[41, "acquisition_session_position"] == 42
    assert sessions.loc[42, "acquisition_session_id"] == "butppg-990-session-02"
    assert sessions.loc[42, "acquisition_session_position"] == 1
    assert sessions.loc[95, "acquisition_session_position"] == 54
    assert sessions["acquisition_session_id"].value_counts().to_dict() == {
        "butppg-990-session-02": 54,
        "butppg-990-session-01": 42,
    }
