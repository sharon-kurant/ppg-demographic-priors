from __future__ import annotations

import numpy as np
import pytest

from ppg_bp_incremental.data.benchmark_splits import make_benchmark_splits
from ppg_bp_incremental.training.ridge_benchmark import run_ridge_benchmark


def _common(synthetic_data):
    data = synthetic_data.copy()
    data["dataset"] = "Synthetic"
    data["source"] = "fixture"
    data["measurement_id"] = data["segment_id"]
    data["height_cm"] = 170.0
    data["weight_kg"] = 70.0
    return data


def test_ridge_conditions_share_subject_splits_and_emit_raw_bp(synthetic_data):
    data = _common(synthetic_data)
    splits = make_benchmark_splits(data, seed=123)
    index = data[["segment_id"]].copy()
    representation = np.column_stack(
        [
            data["age"].to_numpy(float),
            data["bmi"].to_numpy(float),
            data["sex"].eq("M").to_numpy(float),
        ]
    ).astype(np.float32)
    prediction = run_ridge_benchmark(
        data,
        splits,
        "frozen_demographics",
        "papagei_p",
        index,
        representation,
        targets=("sbp",),
        seed=123,
    )

    assert len(prediction) == data["subject_id"].nunique() * 3
    assert set(prediction["target"]) == {"sbp"}
    assert set(prediction["condition"]) == {"frozen_demographics"}
    assert set(prediction["demographic_fields"]) == {"age/sex/BMI"}
    assert set(prediction["estimator_output_scale"]) == {"raw_mmhg"}
    assert set(prediction["decoder"]) == {"identity"}
    assert set(prediction["ridge_fit_weighting"]) == {
        "equal_participant_then_equal_measurement_then_equal_source_row"
    }
    assert set(prediction["alpha_selection_metric"]) == {
        "pooled_participant_macro_oof_mae"
    }
    np.testing.assert_allclose(
        prediction["estimator_output"], prediction["y_pred_mmhg"]
    )
    assert np.isfinite(prediction["y_pred_mmhg"]).all()
    assert not prediction.duplicated(["segment_id", "target", "model", "condition"]).any()

    without_demographics = run_ridge_benchmark(
        data,
        splits,
        "frozen",
        "papagei_p",
        index,
        representation,
        targets=("sbp",),
        seed=123,
    )
    comparison_keys = ["segment_id", "fold", "target"]
    assert prediction[comparison_keys].equals(without_demographics[comparison_keys])
    diagnostics = prediction.attrs["fit_diagnostics"]
    assert diagnostics
    assert {row["fit_weighting"] for row in diagnostics} == {
        "equal_participant_then_equal_measurement_then_equal_source_row"
    }
    assert {row["alpha_selection_metric"] for row in diagnostics} == {
        "pooled_participant_macro_oof_mae"
    }


def test_ridge_uses_available_demographics_with_train_only_missing_handling(
    synthetic_data,
):
    data = _common(synthetic_data)
    data.loc[data.index[::7], "age"] = np.nan
    data.loc[data.index[::9], "bmi"] = np.nan
    data.loc[data.index[::11], "sex"] = np.nan
    # Height and weight are present for BMI calculation/provenance but are not
    # direct predictors.
    splits = make_benchmark_splits(data, seed=321)
    prediction = run_ridge_benchmark(
        data,
        splits,
        "demographics",
        "demographics",
        targets=("dbp",),
        seed=321,
    )

    assert set(prediction["demographic_fields"]) == {"age/sex/BMI"}
    assert set(prediction["condition"]) == {"demographics"}
    assert prediction["ridge_input_dimension"].min() >= 5
    assert np.isfinite(prediction["y_pred_mmhg"]).all()

    no_bmi = data.copy()
    no_bmi["bmi"] = np.nan
    no_bmi["height_cm"] = np.nan
    no_bmi["weight_kg"] = np.nan
    prediction_no_bmi = run_ridge_benchmark(
        no_bmi,
        make_benchmark_splits(no_bmi, seed=321),
        "demographics",
        "demographics",
        targets=("dbp",),
        seed=321,
    )
    assert set(prediction_no_bmi["demographic_fields"]) == {"age/sex"}


@pytest.mark.parametrize(
    ("condition", "model", "with_representation", "message"),
    [
        ("demographics", "papagei_p", False, "requires model='demographics'"),
        ("demographics", "demographics", True, "must not receive"),
        ("ppg_features", "papagei_p", True, "requires model='handcrafted_ppg'"),
        ("frozen", "handcrafted_ppg", True, "requires a registered foundation model"),
    ],
)
def test_ridge_rejects_mislabeled_condition_model_pairs(
    synthetic_data,
    condition,
    model,
    with_representation,
    message,
):
    data = _common(synthetic_data)
    splits = make_benchmark_splits(data, seed=222)
    index = data[["segment_id"]].copy() if with_representation else None
    representation = (
        np.ones((len(data), 3), dtype=np.float32)
        if with_representation
        else None
    )
    with pytest.raises(ValueError, match=message):
        run_ridge_benchmark(
            data,
            splits,
            condition,
            model,
            index,
            representation,
            targets=("sbp",),
            seed=222,
        )
