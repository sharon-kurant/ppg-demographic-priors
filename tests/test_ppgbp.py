from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppg_bp_incremental.data.ppgbp import prepare_ppgbp


def test_prepare_ppgbp_harmonizes_metadata_and_flags_invalid_signal(tmp_path):
    root = tmp_path / "Data File"
    waveform_root = root / "0_subject"
    waveform_root.mkdir(parents=True)
    metadata = pd.DataFrame(
        {
            "subject_ID": [2, 7],
            "Sex(M/F)": ["Female", "Male"],
            "Age(year)": [45, 60],
            "Height(cm)": [160, 180],
            "Weight(kg)": [64, 81],
            "Systolic Blood Pressure(mmHg)": [120, 145],
            "Diastolic Blood Pressure(mmHg)": [70, 90],
            "Heart Rate(b/m)": [65, 72],
            "BMI(kg/m^2)": [25.0, 25.0],
            "Hypertension": ["Normal", "Stage 1 hypertension"],
            "Diabetes": [np.nan, "Type 2 Diabetes"],
            "cerebral infarction": [np.nan, np.nan],
            "cerebrovascular disease": [np.nan, np.nan],
        }
    )
    workbook = root / "PPG-BP dataset.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        metadata.to_excel(
            writer,
            sheet_name="cardiovascular dataset",
            startrow=1,
            index=False,
        )
    for subject in (2, 7):
        for recording in (1, 2, 3):
            signal = np.arange(1200, dtype=float) + recording
            if subject == 7 and recording == 3:
                signal[:] = 5
            text = "\t".join(str(value) for value in signal) + "\t"
            (waveform_root / f"{subject}_{recording}.txt").write_text(
                text, encoding="utf-8"
            )

    output = tmp_path / "processed.csv"
    qc_path = tmp_path / "qc.csv"
    provenance_path = tmp_path / "provenance.json"
    harmonized, qc, provenance = prepare_ppgbp(
        root, output, qc_path, provenance_path
    )

    assert len(harmonized) == 6
    assert harmonized["subject_id"].nunique() == 2
    assert set(harmonized["sex"]) == {"female", "male"}
    assert harmonized["segment_id"].is_unique
    assert harmonized["bmi_abs_difference"].max() == pytest.approx(0, abs=1e-12)
    assert qc["valid_for_analysis"].sum() == 5
    invalid = qc[~qc["valid_for_analysis"]].iloc[0]
    assert invalid["segment_id"] == "ppgbp-0007-3"
    assert invalid["exclusion_reason"] == "constant_or_invalid_signal"
    assert provenance["subjects"] == 2
    assert provenance["segments"] == 6
    assert output.exists() and qc_path.exists() and provenance_path.exists()
