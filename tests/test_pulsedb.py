from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppg_bp_incremental.data.benchmark import (
    capped_subject_rows,
    load_benchmark_manifest,
    load_record_waveform,
)
from ppg_bp_incremental.data.pulsedb import (
    prepare_pulsedb,
    validate_pulsedb_raw_waveform_contract,
)
from ppg_bp_incremental.training.ridge_benchmark import (
    EVALUATION_SEGMENT_SEED as RIDGE_EVALUATION_SEED,
)


def test_prepare_pulsedb_npz_keeps_source_array_references(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    time = np.arange(1250) / 125
    ppg = np.stack([np.sin(2 * np.pi * time), np.sin(2.4 * np.pi * time)])
    np.savez(
        raw / "subject_1.npz",
        PPG_Record=ppg,
        SegSBP=np.array([120.0, 132.0]),
        SegDBP=np.array([72.0, 79.0]),
        Subject=np.array(["001"]),
        Age=np.array([54.0]),
        Gender=np.array([1]),
        Height=np.array([175.0]),
        Weight=np.array([77.0]),
    )
    output = tmp_path / "vital.csv"
    provenance = tmp_path / "vital.json"
    data, metadata = prepare_pulsedb(raw, "vital", output, provenance)
    loaded = load_benchmark_manifest(output)

    assert len(data) == len(loaded) == 2
    assert loaded["subject_id"].nunique() == 1
    assert set(loaded["sex"]) == {"male"}
    assert set(loaded["waveform_format"]) == {"npz"}
    assert set(loaded["waveform_dataset"]) == {"PPG_Record"}
    np.testing.assert_allclose(load_record_waveform(loaded.iloc[1]), ppg[1])
    assert metadata["segments"] == 2
    assert metadata["subjects"] == 1


def test_prepare_pulsedb_hdf5_reads_dense_matlab_layout(tmp_path):
    import h5py

    raw = tmp_path / "raw_h5"
    raw.mkdir()
    path = raw / "subject_2.mat"
    time = np.arange(1250) / 125
    ppg = np.column_stack([np.sin(2 * np.pi * time), np.cos(2 * np.pi * time)])
    with h5py.File(path, "w") as handle:
        group = handle.create_group("Subj_Wins")
        group.create_dataset("PPG_Record", data=ppg)
        group.create_dataset("SegSBP", data=np.array([118.0, 126.0]))
        group.create_dataset("SegDBP", data=np.array([68.0, 74.0]))
        group.create_dataset("IncludeFlag", data=np.array([1, 0]))
        group.create_dataset("SubjectID", data=np.bytes_("002"))
        group.create_dataset("Age", data=np.array([61.0]))
        group.create_dataset("Gender", data=np.bytes_("F"))

    output = tmp_path / "mimic.csv"
    data, _ = prepare_pulsedb(raw, "mimic", output, tmp_path / "mimic.json")
    loaded = load_benchmark_manifest(output)

    assert len(data) == 2
    assert len(loaded) == 1
    assert data["include_flag"].tolist() == [1, 0]
    assert set(loaded["subject_id"]) == {"pulsedb-mimic-002"}
    assert set(loaded["sex"]) == {"female"}
    assert set(loaded["waveform_axis"]) == {1}
    assert set(loaded["waveform_dataset"]) == {"/Subj_Wins/PPG_Record"}
    np.testing.assert_allclose(load_record_waveform(loaded.iloc[0]), ppg[:, 0])


def test_prepare_pulsedb_hdf5_reads_matlab_struct_reference_layout(tmp_path):
    import h5py

    raw = tmp_path / "raw_refs"
    raw.mkdir()
    path = raw / "m000001.mat"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("Subj_Wins")
        ppg_refs = group.create_dataset("PPG_Record", (2, 1), dtype=h5py.ref_dtype)
        sbp_refs = group.create_dataset("SegSBP", (2, 1), dtype=h5py.ref_dtype)
        dbp_refs = group.create_dataset("SegDBP", (2, 1), dtype=h5py.ref_dtype)
        age_refs = group.create_dataset("Age", (2, 1), dtype=h5py.ref_dtype)
        sex_refs = group.create_dataset("Gender", (2, 1), dtype=h5py.ref_dtype)
        for index in range(2):
            ppg_refs[index, 0] = handle.create_dataset(
                f"ppg_{index}", data=np.linspace(index, index + 1, 1250)
            ).ref
            sbp_refs[index, 0] = handle.create_dataset(
                f"sbp_{index}", data=np.array([120.0 + index])
            ).ref
            dbp_refs[index, 0] = handle.create_dataset(
                f"dbp_{index}", data=np.array([70.0 + index])
            ).ref
            age_refs[index, 0] = handle.create_dataset(
                f"age_{index}", data=np.array([58.0])
            ).ref
            sex_refs[index, 0] = handle.create_dataset(
                f"sex_{index}", data=np.array([ord("M")], dtype=np.uint16)
            ).ref

    output = tmp_path / "references.csv"
    prepare_pulsedb(raw, "mimic", output, tmp_path / "references.json")
    loaded = load_benchmark_manifest(output)

    assert len(loaded) == 2
    assert set(loaded["sex"]) == {"male"}
    np.testing.assert_allclose(
        load_record_waveform(loaded.iloc[1]), np.linspace(1, 2, 1250)
    )


def test_pulsedb_evaluation_subset_is_fixed_across_methods():
    data = pd.DataFrame(
        {
            "subject_id": ["a"] * 30 + ["b"] * 30,
            "segment_id": [f"a-{index}" for index in range(30)]
            + [f"b-{index}" for index in range(30)],
        }
    )

    first = capped_subject_rows(data, 20, RIDGE_EVALUATION_SEED)
    second = capped_subject_rows(data, 20, RIDGE_EVALUATION_SEED)
    assert first["segment_id"].tolist() == second["segment_id"].tolist()
    assert (first.groupby("subject_id").size() == 20).all()


def test_prepare_pulsedb_samples_reproducible_subjects_and_uniform_segments(tmp_path):
    raw = tmp_path / "sample_raw"
    raw.mkdir()
    time = np.arange(1250) / 125
    for subject in range(12):
        ppg = np.stack(
            [np.sin(2 * np.pi * (1 + subject / 100) * time) for _ in range(12)]
        )
        np.savez(
            raw / f"subject_{subject:02d}.npz",
            PPG_Record=ppg,
            SegSBP=110.0 + subject + np.arange(12),
            SegDBP=65.0 + subject / 2 + np.arange(12) / 4,
            Subject=np.array([f"{subject:02d}"]),
            Age=np.array([30.0 + 4 * subject]),
            Gender=np.array([subject % 2]),
        )

    first_selection = tmp_path / "first_selection.csv"
    second_selection = tmp_path / "second_selection.csv"
    waveform_cache = tmp_path / "sampled_waveforms.h5"
    first, first_metadata = prepare_pulsedb(
        raw,
        "vital",
        tmp_path / "first.csv",
        tmp_path / "first.json",
        subject_limit=8,
        segments_per_subject=4,
        selection_seed=42,
        subject_selection_csv=first_selection,
        waveform_cache_hdf5=waveform_cache,
    )
    second, second_metadata = prepare_pulsedb(
        raw,
        "vital",
        tmp_path / "second.csv",
        tmp_path / "second.json",
        subject_limit=8,
        segments_per_subject=4,
        selection_seed=42,
        subject_selection_csv=second_selection,
    )

    assert len(first) == len(second) == 32
    assert first["subject_id"].nunique() == second["subject_id"].nunique() == 8
    first_indices = first[["subject_id", "source_waveform_index"]].rename(
        columns={"source_waveform_index": "waveform_index"}
    )
    assert first_indices.equals(second[["subject_id", "waveform_index"]])
    assert set(first.groupby("subject_id")["source_waveform_index"].apply(tuple)) == {
        (0, 3, 7, 11)
    }
    assert first_metadata["full_source_files"] == 12
    assert first_metadata["eligible_subjects"] == 12
    assert set(first["waveform_path"]) == {str(waveform_cache.resolve())}
    assert first["source_waveform_path"].nunique() == 8
    assert first_metadata["waveform_cache_hdf5"] == str(waveform_cache.resolve())
    assert first_metadata["waveform_cache_sha256"]
    assert set(first["source_waveform_dataset"]) == {"PPG_Record"}
    validate_pulsedb_raw_waveform_contract(first)
    assert load_record_waveform(first.iloc[0]).shape == (1250,)
    assert first_metadata["subject_selection_sha256"] == second_metadata[
        "subject_selection_sha256"
    ]

    reused_selection = tmp_path / "reused_selection.csv"
    reused, _ = prepare_pulsedb(
        raw,
        "vital",
        tmp_path / "reused.csv",
        tmp_path / "reused.json",
        reuse_subject_selection_csv=first_selection,
        subject_selection_csv=reused_selection,
    )
    assert reused[["subject_id", "waveform_index"]].equals(
        second[["subject_id", "waveform_index"]]
    )


def test_pulsedb_rejects_legacy_filtered_waveform_fields(tmp_path):
    raw = tmp_path / "legacy"
    raw.mkdir()
    np.savez(
        raw / "legacy.npz",
        PPG_F=np.ones((2, 1250)),
        SegSBP=np.array([120.0, 121.0]),
        SegDBP=np.array([70.0, 71.0]),
    )

    with pytest.raises(ValueError, match="PPG_Record"):
        prepare_pulsedb(
            raw,
            "vital",
            tmp_path / "legacy.csv",
            tmp_path / "legacy.json",
        )

    manifest = pd.DataFrame(
        {
            "dataset": ["PulseDB-Vital"],
            "source_waveform_dataset": ["/Subj_Wins/PPG_F"],
        }
    )
    with pytest.raises(ValueError, match="reject normalized/filtered"):
        validate_pulsedb_raw_waveform_contract(manifest)
