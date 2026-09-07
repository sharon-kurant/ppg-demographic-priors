import numpy as np
import pandas as pd
import pytest

from ppg_bp_incremental.evaluation.demographic_influence import (
    BootstrapSpec,
    between_within_metrics,
    demographic_associations,
    demographic_vs_outer_constant,
    fine_tuning_paired_contrasts,
    paired_subject_mae_contrast,
    seed_aware_paired_subject_mae_contrast,
)
from ppg_bp_incremental.models.contracts import CONTRACT_VERSION


def _method_rows(predictions: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["synthetic"] * 4,
            "subject_id": ["a", "a", "b", "b"],
            "measurement_id": ["a0", "a1", "b0", "b1"],
            "fold": [0, 0, 1, 1],
            "target": ["sbp"] * 4,
            "y_true_mmhg": [100.0, 120.0, 130.0, 150.0],
            "y_pred_mmhg": predictions,
            "contract_version": [CONTRACT_VERSION] * 4,
        }
    )


def _minimal_checked_units() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["synthetic"] * 5,
            "subject_id": [f"s{fold}" for fold in range(5)],
            "measurement_id": [f"m{fold}" for fold in range(5)],
            "fold": list(range(5)),
            "target": ["sbp"] * 5,
            "model": ["demographics"] * 5,
            "condition": ["demographics"] * 5,
            "demographic_fields": ["age/sex/BMI"] * 5,
            "y_true_mmhg": [100.0, 110.0, 120.0, 130.0, 140.0],
            "y_pred_mmhg": [120.0] * 5,
            "contract_version": [CONTRACT_VERSION] * 5,
        }
    )


def test_paired_subject_mae_contrast_uses_equal_subject_weight() -> None:
    candidate = _method_rows([100.0, 120.0, 130.0, 150.0])
    reference = _method_rows([110.0, 110.0, 140.0, 140.0])
    result = paired_subject_mae_contrast(
        candidate,
        reference,
        bootstrap=BootstrapSpec(replicates=200, seed=7),
        seed_parts=("test",),
    )
    assert result["delta_mae_candidate_minus_reference"] == -10.0
    assert result["n_subjects"] == 2
    assert result["n_measurements"] == 4
    assert result["ci_high"] == -10.0
    assert result["bootstrap_method"] == "participant_paired_percentile_fixed_oof"
    assert result["bootstrap_replicates"] == 200


def test_paired_subject_mae_contrast_rejects_mismatched_rows() -> None:
    candidate = _method_rows([100.0, 120.0, 130.0, 150.0])
    reference = _method_rows([110.0, 110.0, 140.0, 140.0]).iloc[:-1]
    try:
        paired_subject_mae_contrast(
            candidate,
            reference,
            bootstrap=BootstrapSpec(replicates=10),
        )
    except ValueError as error:
        assert "identical scored rows" in str(error)
    else:
        raise AssertionError("mismatched rows were accepted")


def _seed_rows(
    *,
    dataset: str = "PPG-BP",
    model: str = "pulseppg",
    target: str = "sbp",
    condition: str,
    seeds: tuple[int, ...],
    absolute_errors: dict[int, float],
) -> pd.DataFrame:
    records = []
    demographic = "demographics" in condition
    for seed in seeds:
        for fold in range(5):
            truth = 120.0 + fold
            records.append(
                {
                    "dataset": dataset,
                    "subject_id": f"{dataset}-s{fold}",
                    "measurement_id": f"{dataset}-m{fold}",
                    "fold": fold,
                    "target": target,
                    "model": model,
                    "condition": condition,
                    "demographic_fields": (
                        "age/sex/BMI" if demographic else "none"
                    ),
                    "seed": seed,
                    "y_true_mmhg": truth,
                    "y_pred_mmhg": truth + absolute_errors[seed],
                    "n_segments": 1,
                    "n_seeds": 1,
                    "aggregation_mode": (
                        "single_neural_seed"
                        if condition.startswith("finetuned")
                        else "single_deterministic_fit"
                    ),
                    "contract_version": CONTRACT_VERSION,
                }
            )
    return pd.DataFrame.from_records(records)


def test_seed_aware_contrast_averages_paired_seed_deltas() -> None:
    candidate = _seed_rows(
        condition="finetuned",
        seeds=(17, 23, 42),
        absolute_errors={17: 2.0, 23: 4.0, 42: 6.0},
    )
    reference = _seed_rows(
        condition="frozen",
        seeds=(20260715,),
        absolute_errors={20260715: 5.0},
    )
    result = seed_aware_paired_subject_mae_contrast(
        candidate,
        reference,
        bootstrap=BootstrapSpec(replicates=50, seed=13),
        seed_parts=("synthetic",),
    )
    assert result["seed_17_delta_mae"] == -3.0
    assert result["seed_23_delta_mae"] == -1.0
    assert result["seed_42_delta_mae"] == 1.0
    assert result["delta_mae_candidate_minus_reference"] == -1.0
    assert result["ci_low"] == -1.0
    assert result["ci_high"] == -1.0
    assert result["n_subjects"] == 5
    assert result["n_measurements_per_seed"] == 5
    assert result["reference_seed_mode"] == (
        "deterministic_reference_reused_across_seeds"
    )


def _complete_seed_specific_matrix() -> pd.DataFrame:
    parts = []
    for dataset in ("PPG-BP", "PulseDB-Vital", "PulseDB-MIMIC", "BUT PPG"):
        demographic_fields = (
            "age/sex" if dataset == "PulseDB-MIMIC" else "age/sex/BMI"
        )
        for model in ("papagei_p", "papagei_s", "pulseppg", "anyppg"):
            for target in ("sbp", "dbp"):
                condition_errors = {
                    "frozen": ((20260715,), {20260715: 4.0}),
                    "frozen_demographics": ((20260715,), {20260715: 3.0}),
                    "finetuned": (
                        (17, 23, 42),
                        {17: 5.0, 23: 4.0, 42: 3.0},
                    ),
                    "finetuned_demographics": (
                        (17, 23, 42),
                        {17: 4.0, 23: 3.0, 42: 2.0},
                    ),
                }
                for condition, (seeds, errors) in condition_errors.items():
                    part = _seed_rows(
                        dataset=dataset,
                        model=model,
                        target=target,
                        condition=condition,
                        seeds=seeds,
                        absolute_errors=errors,
                    )
                    if "demographics" in condition:
                        part["demographic_fields"] = demographic_fields
                    parts.append(part)
    return pd.concat(parts, ignore_index=True)


def test_fine_tuning_contrasts_enforce_three_by_32_complete_matrix() -> None:
    units = _complete_seed_specific_matrix()
    contrasts = fine_tuning_paired_contrasts(
        units,
        bootstrap=BootstrapSpec(replicates=20, seed=17),
    )
    assert len(contrasts) == 96
    assert contrasts.groupby("contrast").size().eq(32).all()
    demographic = contrasts[
        contrasts["contrast"].eq("add_demographics_to_finetuned")
    ]
    assert np.allclose(demographic["delta_mae_candidate_minus_reference"], -1.0)
    assert (demographic["ci_high"] < 0).all()
    assert set(contrasts["training_seeds"]) == {"17/23/42"}

    incomplete = units[
        ~(
            units["dataset"].eq("PPG-BP")
            & units["model"].eq("anyppg")
            & units["target"].eq("sbp")
            & units["condition"].eq("finetuned")
            & units["seed"].eq(42)
        )
    ]
    with pytest.raises(ValueError, match="exactly training seeds"):
        fine_tuning_paired_contrasts(
            incomplete,
            bootstrap=BootstrapSpec(replicates=5, seed=17),
        )


def test_between_within_metrics_separates_subject_level_and_change() -> None:
    parts = []
    for fold, subject in enumerate(("a", "b", "c", "d", "e")):
        center = 110.0 + 10.0 * fold
        parts.append(
            pd.DataFrame(
                {
                    "dataset": ["synthetic", "synthetic"],
                    "subject_id": [subject, subject],
                    "measurement_id": [f"{subject}0", f"{subject}1"],
                    "fold": [fold, fold],
                    "target": ["sbp", "sbp"],
                    "y_true_mmhg": [center - 10.0, center + 10.0],
                    "y_pred_mmhg": [center, center],
                    "contract_version": [CONTRACT_VERSION] * 2,
                }
            )
        )
    rows = pd.concat(parts, ignore_index=True)
    rows["model"] = "demographics"
    rows["condition"] = "demographics"
    rows["demographic_fields"] = "age/sex/BMI"
    result = between_within_metrics(rows).iloc[0]
    assert result["between_subject_mean_mae"] == 0.0
    assert result["within_subject_deviation_mae"] == 10.0
    assert result["mean_predicted_within_subject_rms"] == 0.0
    assert np.isclose(result["mean_true_within_subject_rms"], 10.0)


def test_outer_constant_excludes_held_fold_and_weights_subjects_equally() -> None:
    rows = pd.DataFrame(
        {
            "dataset": ["synthetic"] * 14,
            "subject_id": ["held", "low", "middle"] + ["high"] * 10 + ["center"],
            "measurement_id": ["held-0", "low-0", "middle-0"]
            + [f"high-{index}" for index in range(10)]
            + ["center-0"],
            "fold": [0, 1, 2] + [3] * 10 + [4],
            "target": ["sbp"] * 14,
            "model": ["demographics"] * 14,
            "condition": ["demographics"] * 14,
            "demographic_fields": ["age/sex/BMI"] * 14,
            "y_true_mmhg": [1000.0, 0.0, 10.0] + [20.0] * 10 + [10.0],
            "y_pred_mmhg": [100.0] * 14,
            "contract_version": [CONTRACT_VERSION] * 14,
        }
    )
    _, predictions = demographic_vs_outer_constant(
        rows,
        bootstrap=BootstrapSpec(replicates=20, seed=9),
    )
    held = predictions[
        predictions["subject_id"].eq("held")
        & predictions["constant_method"].eq(
            "outer_fit_participant_weighted_mean"
        )
    ]
    assert len(held) == 1
    # Equal subject weighting gives mean([0, 10, 20, 10]) = 10. Row weighting
    # would over-weight the ten records from the high subject, and leakage
    # would include the held value of 1000.
    assert np.isclose(held.iloc[0]["constant_prediction_mmhg"], 10.0)


def test_checked_units_rejects_wrong_contract() -> None:
    rows = _minimal_checked_units()
    rows["contract_version"] = "superseded"
    with pytest.raises(ValueError, match=f"requires {CONTRACT_VERSION}"):
        between_within_metrics(rows)


def test_checked_units_rejects_subject_in_two_outer_folds() -> None:
    rows = _minimal_checked_units()
    rows.loc[1, "subject_id"] = rows.loc[0, "subject_id"]
    with pytest.raises(ValueError, match="more than one outer fold"):
        between_within_metrics(rows)


def test_demographic_associations_canonicalizes_variable_bmi_by_unique_median() -> None:
    units = _minimal_checked_units()
    raw = pd.DataFrame(
        {
            "dataset": ["synthetic"] * 6,
            "subject_id": ["s0", "s0", "s1", "s2", "s3", "s4"],
            "age": [20.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "sex": ["female", "female", "male", "female", "male", "female"],
            "bmi": [20.0, 30.0, 22.0, 24.0, 26.0, 28.0],
            "model": ["demographics"] * 6,
            "condition": ["demographics"] * 6,
            "contract_version": [CONTRACT_VERSION] * 6,
        }
    )
    _, distributions = demographic_associations(units, raw)
    row = distributions.iloc[0]
    assert row["bmi_subjects_with_multiple_observed_values"] == 1
    assert np.isclose(row["bmi_mean"], 25.0)
