from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppg_bp_incremental.evaluation.benchmark import (
    _bootstrap_intervals,
    aggregate_prediction_units,
    heatmap_method_display_label,
    incremental_effects,
    method_display_label,
    participant_balanced_metrics,
    summarize_benchmark,
    validate_complete_benchmark_matrix,
)


def test_demographic_display_shorthand_and_heatmap_row_collapsing():
    assert (
        method_display_label("anyppg", "frozen_demographics", "age/sex/BMI")
        == "Frozen AnyPPG + Demo"
    )
    assert (
        method_display_label("anyppg", "frozen_demographics", "age/sex")
        == "Frozen AnyPPG + Demo†"
    )
    assert (
        method_display_label("demographics", "demographics", "age/sex")
        == "Demographics only†"
    )
    assert (
        heatmap_method_display_label("anyppg", "frozen_demographics")
        == "Frozen AnyPPG + Demo"
    )
    assert "Ridge" not in method_display_label(
        "papagei_p", "frozen", "none"
    )


def _predictions() -> pd.DataFrame:
    rows = []
    for subject, truth in (("s1", 120.0), ("s2", 140.0)):
        for segment in range(3):
            for seed in (20260715,):
                rows.append(
                    {
                        "dataset": "PPG-BP",
                        "source": "PPG-BP",
                        "subject_id": subject,
                        "measurement_id": subject,
                        "segment_id": f"{subject}-{segment}",
                        "fold": 0,
                        "seed": seed,
                        "model": "papagei_p",
                        "condition": "frozen",
                        "target": "sbp",
                        "y_true_mmhg": truth,
                        "y_pred_mmhg": truth + segment,
                        "age": 50.0,
                        "sex": "female",
                        "bmi": 24.0,
                        "height_cm": 165.0,
                        "weight_kg": 65.0,
                        "preprocessing_policy": "official",
                        "padding_policy": "zero",
                        "padding_required": True,
                        "pretraining_overlap": "unknown",
                        "source_fidelity": "released_exact",
                        "contract_version": "source-faithful-v2",
                        "demographic_fields": "none",
                        "estimator_output": truth + segment,
                        "estimator_output_scale": "raw_mmhg",
                        "decoder": "identity",
                        "ridge_input_dimension": 512,
                    }
                )
    return pd.DataFrame(rows)


def test_ppgbp_recordings_are_averaged_before_scoring():
    units = aggregate_prediction_units(_predictions())
    aggregated, metrics = summarize_benchmark(_predictions(), bootstrap_replicates=10)

    assert len(units) == len(aggregated) == 2
    assert set(units["n_segments"]) == {3}
    assert set(units["n_seeds"]) == {1}
    assert metrics.iloc[0]["n_subjects"] == 2
    assert np.isfinite(metrics.iloc[0][["mae", "rmse", "bias", "pearson"]].to_numpy(float)).all()


def test_reflection_padding_is_rejected_from_corrected_artifacts():
    reflected = _predictions()
    reflected["padding_policy"] = "reflect"
    with pytest.raises(ValueError, match="Reflection padding"):
        summarize_benchmark(reflected, bootstrap_replicates=0)


def test_all_available_demographics_are_compared_with_frozen_condition():
    metrics = pd.DataFrame(
        [
            {
                "dataset": "BUT PPG",
                "model": "anyppg",
                "condition": condition,
                "target": "sbp",
                "preprocessing_policy": "official",
                "padding_policy": "zero",
                "mae": mae,
                "rmse": mae + 2,
            }
            for condition, demographic_fields, mae in (
                ("frozen", "none", 12.0),
                ("frozen_demographics", "age/sex/BMI", 11.5),
            )
        ]
    )
    metrics["demographic_fields"] = ["none", "age/sex/BMI"]
    effects = incremental_effects(metrics)
    assert effects.iloc[0]["effect"] == "demographics_added_to_frozen"
    assert effects.iloc[0]["delta_mae"] == -0.5


def test_vectorized_participant_bootstrap_matches_one_explicit_draw():
    data = pd.DataFrame(
        [
            {"subject_id": "s1", "measurement_id": "s1-1", "y_true_mmhg": 110.0, "y_pred_mmhg": 112.0},
            {"subject_id": "s1", "measurement_id": "s1-2", "y_true_mmhg": 120.0, "y_pred_mmhg": 119.0},
            {"subject_id": "s2", "measurement_id": "s2-1", "y_true_mmhg": 130.0, "y_pred_mmhg": 126.0},
            {"subject_id": "s3", "measurement_id": "s3-1", "y_true_mmhg": 140.0, "y_pred_mmhg": 145.0},
            {"subject_id": "s3", "measurement_id": "s3-2", "y_true_mmhg": 150.0, "y_pred_mmhg": 148.0},
            {"subject_id": "s3", "measurement_id": "s3-3", "y_true_mmhg": 160.0, "y_pred_mmhg": 155.0},
        ]
    )
    seed = 91
    subjects = np.asarray(sorted(data["subject_id"].unique()))
    chosen = np.random.default_rng(seed).choice(
        subjects, len(subjects), replace=True
    )
    blocks = []
    for draw, subject in enumerate(chosen):
        block = data[data["subject_id"] == subject].copy()
        block["subject_id"] = f"bootstrap-{draw}"
        blocks.append(block)
    expected = participant_balanced_metrics(pd.concat(blocks, ignore_index=True))
    observed = _bootstrap_intervals(data, replicates=1, confidence=0.95, seed=seed)

    for metric in (
        "mae",
        "rmse",
        "bias",
        "error_std",
        "r2",
        "pearson",
        "calibration_slope",
        "calibration_intercept",
    ):
        assert observed[f"{metric}_ci_low"] == pytest.approx(
            expected[metric], rel=1e-10, abs=1e-10
        )
        assert observed[f"{metric}_ci_high"] == pytest.approx(
            expected[metric], rel=1e-10, abs=1e-10
        )


def test_complete_matrix_enforces_all_methods_and_identical_rows():
    methods = [
        ("demographics", "demographics", "age/sex/BMI"),
        ("handcrafted_ppg", "ppg_features", "none"),
        (
            "handcrafted_ppg",
            "ppg_features_demographics",
            "age/sex/BMI",
        ),
        *[
            (
                model,
                condition,
                "age/sex/BMI" if condition.endswith("demographics") else "none",
            )
            for model in ("papagei_p", "papagei_s", "pulseppg", "anyppg")
            for condition in ("frozen", "frozen_demographics")
        ],
    ]
    rows = []
    for dataset in ("PPG-BP", "PulseDB-Vital", "PulseDB-MIMIC", "BUT PPG"):
        for model, condition, demographic_fields in methods:
            if (
                dataset == "PulseDB-MIMIC"
                and demographic_fields == "age/sex/BMI"
            ):
                demographic_fields = "age/sex"
            for target, truth in (("sbp", 120.0), ("dbp", 75.0)):
                row = _predictions().iloc[0].to_dict()
                row.update(
                    {
                        "dataset": dataset,
                        "source": dataset,
                        "segment_id": f"{dataset}-segment",
                        "model": model,
                        "condition": condition,
                        "target": target,
                        "y_true_mmhg": truth,
                        "y_pred_mmhg": truth + 1,
                        "estimator_output": truth + 1,
                        "demographic_fields": demographic_fields,
                        "padding_policy": (
                            "zero"
                            if dataset == "PPG-BP"
                            and model
                            in {"papagei_p", "papagei_s", "pulseppg", "anyppg"}
                            else "none"
                        ),
                        "padding_required": bool(
                            dataset == "PPG-BP"
                            and model
                            in {"papagei_p", "papagei_s", "pulseppg", "anyppg"}
                        ),
                    }
                )
                rows.append(row)
    complete = pd.DataFrame(rows)
    validate_complete_benchmark_matrix(complete)

    missing = complete[
        ~(
            complete["dataset"].eq("BUT PPG")
            & complete["model"].eq("anyppg")
            & complete["condition"].eq("frozen")
        )
    ]
    with pytest.raises(ValueError, match="active method matrix differs"):
        validate_complete_benchmark_matrix(missing)

    changed = complete.copy()
    changed.loc[
        changed["model"].eq("papagei_p")
        & changed["condition"].eq("frozen"),
        "segment_id",
    ] = "different-segment"
    with pytest.raises(ValueError, match="identical evaluation rows"):
        validate_complete_benchmark_matrix(changed)
