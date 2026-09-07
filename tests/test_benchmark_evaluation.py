from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ppg_bp_incremental.evaluation.benchmark import (
    PAIRED_CONFIGURATIONS,
    _paired_mae_plot,
    _bootstrap_seed_mean_intervals,
    _bootstrap_intervals,
    aggregate_prediction_units,
    heatmap_method_display_label,
    incremental_effects,
    load_prediction_artifacts,
    method_display_label,
    participant_balanced_metrics,
    summarize_benchmark,
    validate_complete_benchmark_matrix,
    validate_prediction_artifact,
)
from ppg_bp_incremental.models.contracts import CONTRACT_VERSION


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
                        "contract_version": CONTRACT_VERSION,
                        "demographic_fields": "none",
                        "estimator_output": truth + segment,
                        "estimator_output_scale": "raw_mmhg",
                        "decoder": "identity",
                        "decoder_mean_mmhg": 0.0,
                        "decoder_standard_deviation_mmhg": 1.0,
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


def test_neural_zscore_output_requires_explicit_affine_decoder():
    neural = _predictions()
    neural["condition"] = "finetuned"
    neural["seed"] = 17
    neural["estimator_output"] = 0.25
    neural["y_pred_mmhg"] = 122.5
    neural["estimator_output_scale"] = "full_development_pool_zscore"
    neural["decoder"] = "affine_full_development_pool_inverse_zscore"
    neural["decoder_mean_mmhg"] = 120.0
    neural["decoder_standard_deviation_mmhg"] = 10.0
    validate_prediction_artifact(neural)

    neural["decoder"] = "identity"
    with pytest.raises(ValueError, match="affine z-score decoding"):
        validate_prediction_artifact(neural)


def test_loader_hydrates_and_verifies_neural_decoder_from_run_metadata(tmp_path):
    neural = _predictions()
    neural["condition"] = "finetuned"
    neural["seed"] = 17
    neural["estimator_output"] = 0.25
    neural["y_pred_mmhg"] = 122.5
    neural["estimator_output_scale"] = "full_development_pool_zscore"
    neural["decoder"] = "affine_full_development_pool_inverse_zscore"
    neural = neural.drop(
        columns=["decoder_mean_mmhg", "decoder_standard_deviation_mmhg"]
    )
    prediction_path = tmp_path / "predictions.csv"
    neural.to_csv(prediction_path, index=False)
    metadata = {
        "dataset": "PPG-BP",
        "model": "papagei_p",
        "condition": "finetuned",
        "target": "sbp",
        "fold": 0,
        "seed": 17,
        "development_target_transform": {
            "mean_mmhg": 120.0,
            "standard_deviation_mmhg": 10.0,
        },
    }
    (tmp_path / "run.json").write_text(json.dumps(metadata), encoding="utf-8")

    hydrated = load_prediction_artifacts([prediction_path])
    validate_prediction_artifact(hydrated)
    assert set(hydrated["decoder_mean_mmhg"]) == {120.0}
    assert set(hydrated["decoder_standard_deviation_mmhg"]) == {10.0}

    hydrated["y_pred_mmhg"] += 1.0
    with pytest.raises(ValueError, match="do not match the declared affine decoder"):
        validate_prediction_artifact(hydrated)


def test_neural_primary_metrics_average_seed_scores_not_predictions():
    frames = []
    for training_seed, offset in ((17, -10.0), (23, 10.0), (42, 0.0)):
        frame = _predictions()
        frame["condition"] = "finetuned"
        frame["seed"] = training_seed
        frame["y_pred_mmhg"] = frame["y_true_mmhg"] + offset
        frame["estimator_output"] = (frame["y_pred_mmhg"] - 120.0) / 10.0
        frame["estimator_output_scale"] = "full_development_pool_zscore"
        frame["decoder"] = "affine_full_development_pool_inverse_zscore"
        frame["decoder_mean_mmhg"] = 120.0
        frame["decoder_standard_deviation_mmhg"] = 10.0
        frames.append(frame)

    units, metrics, seed_units, seed_metrics = summarize_benchmark(
        pd.concat(frames, ignore_index=True),
        bootstrap_replicates=0,
        return_seed_details=True,
    )

    # Mean predictions happen to be perfect, but two of three independently
    # trained models are 10 mmHg wrong. The primary score must not claim the
    # zero-error ensemble result.
    assert np.allclose(units["y_pred_mmhg"], units["y_true_mmhg"])
    assert metrics.iloc[0]["mae"] == pytest.approx(20.0 / 3.0)
    assert metrics.iloc[0]["mae_seed_sd"] == pytest.approx(
        np.std([10.0, 10.0, 0.0], ddof=1)
    )
    assert metrics.iloc[0]["aggregation_mode"] == (
        "mean_of_3_seed_specific_metrics"
    )
    assert metrics.iloc[0]["n_seeds"] == 3
    assert set(seed_metrics["mae"]) == {0.0, 10.0}
    assert set(seed_units["aggregation_mode"]) == {"single_neural_seed"}
    assert set(units["aggregation_mode"]) == {
        "diagnostic_mean_prediction_across_3_training_seeds"
    }


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


def test_finetuned_demographic_effect_is_included():
    metrics = pd.DataFrame(
        [
            {
                "dataset": "BUT PPG",
                "model": "anyppg",
                "condition": condition,
                "target": "dbp",
                "preprocessing_policy": "official",
                "padding_policy": "none",
                "demographic_fields": demographic_fields,
                "mae": mae,
                "rmse": mae + 2,
            }
            for condition, demographic_fields, mae in (
                ("finetuned", "none", 8.0),
                ("finetuned_demographics", "age/sex/BMI", 7.6),
            )
        ]
    )
    effects = incremental_effects(metrics)
    assert len(effects) == 1
    assert effects.iloc[0]["effect"] == "demographics_added_to_finetuned"
    assert effects.iloc[0]["delta_mae"] == pytest.approx(-0.4)


def test_paired_mae_plot_supports_frozen_and_finetuned_groups(tmp_path):
    methods = [("demographics", "demographics")]
    for model, without_demo, with_demo, _ in PAIRED_CONFIGURATIONS:
        methods.extend(((model, without_demo), (model, with_demo)))
    rows = []
    for target_index, target in enumerate(("sbp", "dbp")):
        for method_index, (model, condition) in enumerate(methods):
            mae = 5.0 + target_index + method_index / 10
            rows.append(
                {
                    "dataset": "PulseDB-MIMIC",
                    "model": model,
                    "condition": condition,
                    "target": target,
                    "demographic_fields": (
                        "age/sex" if condition.endswith("demographics") else "none"
                    ),
                    "mae": mae,
                    "mae_seed_sd": (
                        0.2
                        if condition in {"finetuned", "finetuned_demographics"}
                        else 0.0
                    ),
                    "mae_ci_low": mae - 0.2,
                    "mae_ci_high": mae + 0.2,
                }
            )
    path = _paired_mae_plot(
        pd.DataFrame(rows), "PulseDB-MIMIC", tmp_path
    )
    assert path.is_file()
    assert path.stat().st_size > 0


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


def test_seed_mean_bootstrap_resamples_subjects_then_averages_metrics():
    base = pd.DataFrame(
        [
            {"subject_id": "s1", "measurement_id": "s1-1", "y_true_mmhg": 110.0},
            {"subject_id": "s1", "measurement_id": "s1-2", "y_true_mmhg": 120.0},
            {"subject_id": "s2", "measurement_id": "s2-1", "y_true_mmhg": 130.0},
            {"subject_id": "s3", "measurement_id": "s3-1", "y_true_mmhg": 140.0},
        ]
    )
    seeded = []
    for training_seed, errors in (
        (17, [2.0, -1.0, -4.0, 5.0]),
        (23, [-3.0, 2.0, 1.0, -6.0]),
        (42, [1.0, 1.0, -2.0, 2.0]),
    ):
        frame = base.copy()
        frame["seed"] = training_seed
        frame["y_pred_mmhg"] = frame["y_true_mmhg"] + errors
        seeded.append(frame)
    data = pd.concat(seeded, ignore_index=True)
    seed = 137
    subjects = np.asarray(sorted(base["subject_id"].unique()))
    chosen = np.random.default_rng(seed).choice(
        subjects, len(subjects), replace=True
    )
    expected_by_seed = []
    for _, group in data.groupby("seed"):
        blocks = []
        for draw, subject in enumerate(chosen):
            block = group[group["subject_id"] == subject].copy()
            block["subject_id"] = f"bootstrap-{draw}"
            blocks.append(block)
        expected_by_seed.append(
            participant_balanced_metrics(pd.concat(blocks, ignore_index=True))
        )
    observed = _bootstrap_seed_mean_intervals(
        data, replicates=1, confidence=0.95, seed=seed
    )
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
        expected = np.mean([row[metric] for row in expected_by_seed])
        assert observed[f"{metric}_ci_low"] == pytest.approx(
            expected, rel=1e-10, abs=1e-10
        )
        assert observed[f"{metric}_ci_high"] == pytest.approx(
            expected, rel=1e-10, abs=1e-10
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
            for condition in (
                "frozen",
                "frozen_demographics",
                "finetuned",
                "finetuned_demographics",
            )
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
                seeds = (
                    (17, 23, 42)
                    if condition in {"finetuned", "finetuned_demographics"}
                    else (20260715,)
                )
                for seed in seeds:
                    row = _predictions().iloc[0].to_dict()
                    row.update(
                        {
                        "dataset": dataset,
                        "source": dataset,
                        "segment_id": f"{dataset}-segment",
                        "model": model,
                        "condition": condition,
                        "target": target,
                        "seed": seed,
                        "y_true_mmhg": truth,
                        "y_pred_mmhg": truth + 1,
                        "estimator_output": (
                            0.1
                            if condition in {"finetuned", "finetuned_demographics"}
                            else truth + 1
                        ),
                        "estimator_output_scale": (
                            "full_development_pool_zscore"
                            if condition in {"finetuned", "finetuned_demographics"}
                            else "raw_mmhg"
                        ),
                        "decoder": (
                            "affine_full_development_pool_inverse_zscore"
                            if condition in {"finetuned", "finetuned_demographics"}
                            else "identity"
                        ),
                        "decoder_mean_mmhg": (
                            truth
                            if condition in {"finetuned", "finetuned_demographics"}
                            else 0.0
                        ),
                        "decoder_standard_deviation_mmhg": (
                            10.0
                            if condition in {"finetuned", "finetuned_demographics"}
                            else 1.0
                        ),
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

    missing_one_seed_row = complete.drop(
        complete[
            complete["dataset"].eq("PPG-BP")
            & complete["model"].eq("papagei_p")
            & complete["condition"].eq("finetuned")
            & complete["target"].eq("sbp")
            & complete["seed"].eq(42)
        ].index
    )
    with pytest.raises(ValueError, match="expected seed set for every evaluation row"):
        validate_complete_benchmark_matrix(missing_one_seed_row)
