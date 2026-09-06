from __future__ import annotations

import pandas as pd

from ppg_bp_incremental.data.benchmark_splits import (
    make_benchmark_splits,
    manifest_fingerprint,
    refresh_benchmark_split_fingerprint,
    rows_for_role,
    validate_benchmark_splits,
)


def _common(synthetic_data: pd.DataFrame) -> pd.DataFrame:
    data = synthetic_data.copy()
    data["dataset"] = "Synthetic"
    data["source"] = "fixture"
    data["measurement_id"] = data["segment_id"]
    return data


def test_long_split_manifest_is_subject_disjoint_and_complete(synthetic_data):
    data = _common(synthetic_data)
    manifest = make_benchmark_splits(data, outer_folds=5, seed=99)
    validate_benchmark_splits(data, manifest, outer_folds=5)

    assert len(manifest) == data["subject_id"].nunique() * 5
    for fold in range(5):
        roles = {
            role: set(rows_for_role(data, manifest, fold, role)["subject_id"])
            for role in ("train", "validation", "test")
        }
        assert roles["train"].isdisjoint(roles["validation"])
        assert roles["train"].isdisjoint(roles["test"])
        assert roles["validation"].isdisjoint(roles["test"])
        assert set.union(*roles.values()) == set(data["subject_id"])

    test_counts = manifest[manifest["role"] == "test"].groupby("subject_id").size()
    assert (test_counts == 1).all()


def test_manifest_fingerprint_survives_staging_and_csv_round_trip(
    tmp_path, synthetic_data
):
    data = _common(synthetic_data)
    data["waveform_sha256"] = [f"sha-{index}" for index in range(len(data))]
    staged = data.copy()
    staged["waveform_path"] = "/node-local/cache.h5"
    path = tmp_path / "staged.csv"
    staged.to_csv(path, index=False)
    reloaded = pd.read_csv(path)

    assert manifest_fingerprint(data) == manifest_fingerprint(reloaded)


def test_split_refresh_preserves_roles_and_updates_fingerprint(synthetic_data):
    original = _common(synthetic_data)
    original["waveform_sha256"] = "legacy-filtered-waveform"
    splits = make_benchmark_splits(original, seed=99)
    corrected = original.copy()
    corrected["waveform_sha256"] = "raw-ppg-record-waveform"

    refreshed = refresh_benchmark_split_fingerprint(corrected, splits)

    pd.testing.assert_frame_equal(
        refreshed.drop(columns="dataset_fingerprint"),
        splits.drop(columns="dataset_fingerprint"),
    )
    assert set(refreshed["dataset_fingerprint"]) == {
        manifest_fingerprint(corrected)
    }
