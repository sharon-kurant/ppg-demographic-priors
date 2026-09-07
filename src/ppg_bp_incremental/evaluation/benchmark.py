"""Participant-balanced benchmark metrics, uncertainty, and visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ppg_bp_incremental.models.contracts import CONTRACT_VERSION


PREDICTION_COLUMNS = {
    "dataset", "source", "subject_id", "measurement_id", "segment_id", "fold",
    "seed", "model", "condition", "target", "y_true_mmhg", "y_pred_mmhg",
    "age", "sex", "bmi", "height_cm", "weight_kg", "preprocessing_policy",
    "padding_policy", "padding_required", "pretraining_overlap", "source_fidelity",
    "contract_version", "demographic_fields", "estimator_output",
    "estimator_output_scale", "decoder", "ridge_input_dimension",
}
ACTIVE_CONDITIONS = {
    "demographics",
    "ppg_features",
    "ppg_features_demographics",
    "frozen",
    "frozen_demographics",
    "finetuned",
    "finetuned_demographics",
}
NEURAL_CONDITIONS = {"finetuned", "finetuned_demographics"}
DEMOGRAPHIC_CONDITIONS = {
    "demographics",
    "ppg_features_demographics",
    "frozen_demographics",
    "finetuned_demographics",
}
NEURAL_SEEDS = {17, 23, 42}
EXPECTED_DATASETS = {
    "PPG-BP",
    "PulseDB-Vital",
    "PulseDB-MIMIC",
    "BUT PPG",
}
EXPECTED_DEMOGRAPHIC_FIELDS = {
    "PPG-BP": "age/sex/BMI",
    "PulseDB-Vital": "age/sex/BMI",
    "PulseDB-MIMIC": "age/sex",
    "BUT PPG": "age/sex/BMI",
}
EXPECTED_METHODS = {
    ("demographics", "demographics"),
    ("handcrafted_ppg", "ppg_features"),
    ("handcrafted_ppg", "ppg_features_demographics"),
    *{
        (model, condition)
        for model in ("papagei_p", "papagei_s", "pulseppg", "anyppg")
        for condition in (
            "frozen",
            "frozen_demographics",
            "finetuned",
            "finetuned_demographics",
        )
    },
}
MODEL_DISPLAY = {
    "papagei_p": "PaPaGei-P",
    "papagei_s": "PaPaGei-S",
    "pulseppg": "Pulse-PPG",
    "anyppg": "AnyPPG",
    "handcrafted_ppg": "PPG features",
    "demographics": "Demographics",
}
PAIRED_CONFIGURATIONS = (
    (
        "handcrafted_ppg",
        "ppg_features",
        "ppg_features_demographics",
        "PPG features",
    ),
    ("papagei_p", "frozen", "frozen_demographics", "Frozen PaPaGei-P"),
    ("papagei_s", "frozen", "frozen_demographics", "Frozen PaPaGei-S"),
    ("pulseppg", "frozen", "frozen_demographics", "Frozen Pulse-PPG"),
    ("anyppg", "frozen", "frozen_demographics", "Frozen AnyPPG"),
    ("papagei_p", "finetuned", "finetuned_demographics", "Fine-tuned PaPaGei-P"),
    ("papagei_s", "finetuned", "finetuned_demographics", "Fine-tuned PaPaGei-S"),
    ("pulseppg", "finetuned", "finetuned_demographics", "Fine-tuned Pulse-PPG"),
    ("anyppg", "finetuned", "finetuned_demographics", "Fine-tuned AnyPPG"),
)


def load_prediction_artifacts(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load raw run predictions and attach auditable decoder parameters.

    Fine-tuning jobs keep the development-pool target transform in the sibling
    ``run.json``. This loader copies those parameters into every prediction row
    before validation, allowing the final mmHg value to be checked numerically
    without modifying already-running jobs. Ridge's identity decoder is encoded
    as mean zero and scale one.
    """

    frames: list[pd.DataFrame] = []
    for source in paths:
        path = Path(source)
        frame = pd.read_csv(path)
        neural = frame["condition"].isin(NEURAL_CONDITIONS)
        if neural.any():
            if not neural.all():
                raise ValueError(
                    f"{path} mixes fine-tuned and non-neural prediction rows"
                )
            metadata_path = path.with_name("run.json")
            if not metadata_path.is_file():
                raise ValueError(
                    f"Fine-tuned predictions lack sibling metadata: {metadata_path}"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            transform = metadata.get("development_target_transform", {})
            mean_mmhg = float(transform.get("mean_mmhg", np.nan))
            standard_deviation_mmhg = float(
                transform.get("standard_deviation_mmhg", np.nan)
            )
            if not np.isfinite(mean_mmhg) or not (
                np.isfinite(standard_deviation_mmhg)
                and standard_deviation_mmhg > 0
            ):
                raise ValueError(
                    f"{metadata_path} has an invalid affine BP decoder"
                )
            identity = {
                "dataset": str(frame["dataset"].iloc[0]),
                "model": str(frame["model"].iloc[0]),
                "condition": str(frame["condition"].iloc[0]),
                "target": str(frame["target"].iloc[0]),
                "fold": int(frame["fold"].iloc[0]),
                "seed": int(frame["seed"].iloc[0]),
            }
            for key, observed in identity.items():
                expected = metadata.get(key)
                if str(expected) != str(observed):
                    raise ValueError(
                        f"{metadata_path} identity mismatch for {key}: "
                        f"metadata={expected!r}, predictions={observed!r}"
                    )
            frame["decoder_mean_mmhg"] = mean_mmhg
            frame["decoder_standard_deviation_mmhg"] = (
                standard_deviation_mmhg
            )
        else:
            frame["decoder_mean_mmhg"] = 0.0
            frame["decoder_standard_deviation_mmhg"] = 1.0
        frame["prediction_artifact"] = str(path.resolve())
        frames.append(frame)
    if not frames:
        raise ValueError("At least one prediction artifact is required")
    return pd.concat(frames, ignore_index=True)


def validate_prediction_artifact(
    predictions: pd.DataFrame,
    additional_conditions: set[str] | None = None,
) -> None:
    missing = sorted(PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction artifact is missing columns: {missing}")
    if not set(predictions["target"]).issubset({"sbp", "dbp"}):
        raise ValueError("Prediction targets must be sbp or dbp")
    if not np.isfinite(predictions[["y_true_mmhg", "y_pred_mmhg"]]).all().all():
        raise ValueError("Prediction artifact contains non-finite BP values")
    if set(predictions["contract_version"].astype(str)) != {CONTRACT_VERSION}:
        raise ValueError("Prediction artifacts do not use the active contract version")
    allowed_conditions = ACTIVE_CONDITIONS | (additional_conditions or set())
    if not set(predictions["condition"]).issubset(allowed_conditions):
        raise ValueError("Prediction artifact contains a removed active condition")
    fine_tuned = predictions["condition"].isin(NEURAL_CONDITIONS)
    ridge = predictions.loc[~fine_tuned]
    if not ridge.empty and set(ridge["estimator_output_scale"].astype(str)) != {
        "raw_mmhg"
    }:
        raise ValueError("Active Ridge predictions must declare raw_mmhg estimator output")
    if not ridge.empty and set(ridge["decoder"].astype(str)) != {"identity"}:
        raise ValueError("Active Ridge predictions must declare identity decoding")
    neural = predictions.loc[fine_tuned]
    if not neural.empty:
        if set(neural["estimator_output_scale"].astype(str)) != {
            "full_development_pool_zscore"
        }:
            raise ValueError(
                "Fine-tuned predictions must declare full-development-pool "
                "z-score output"
            )
        if set(neural["decoder"].astype(str)) != {
            "affine_full_development_pool_inverse_zscore"
        }:
            raise ValueError(
                "Fine-tuned predictions must declare explicit full-development-pool "
                "affine z-score decoding"
            )
        if not np.isfinite(
            pd.to_numeric(neural["estimator_output"], errors="coerce")
        ).all():
            raise ValueError("Fine-tuned z-score estimator outputs must be finite")
        decoder_columns = {
            "decoder_mean_mmhg", "decoder_standard_deviation_mmhg"
        }
        observed_decoder_columns = decoder_columns.intersection(neural.columns)
        if observed_decoder_columns and observed_decoder_columns != decoder_columns:
            raise ValueError("Fine-tuned affine decoder metadata are incomplete")
        if observed_decoder_columns:
            decoder_parameters = neural[
                ["decoder_mean_mmhg", "decoder_standard_deviation_mmhg"]
            ].apply(pd.to_numeric, errors="coerce")
            if not np.isfinite(decoder_parameters).all().all() or not (
                decoder_parameters["decoder_standard_deviation_mmhg"] > 0
            ).all():
                raise ValueError("Fine-tuned affine decoder parameters must be finite")
            run_identity = [
                "dataset", "model", "condition", "target", "fold", "seed"
            ]
            if neural.groupby(run_identity, dropna=False)[
                ["decoder_mean_mmhg", "decoder_standard_deviation_mmhg"]
            ].nunique(dropna=False).gt(1).any().any():
                raise ValueError("Fine-tuned decoder parameters changed within one run")
            decoded = (
                pd.to_numeric(
                    neural["estimator_output"], errors="coerce"
                ).to_numpy(float)
                * decoder_parameters[
                    "decoder_standard_deviation_mmhg"
                ].to_numpy(float)
                + decoder_parameters["decoder_mean_mmhg"].to_numpy(float)
            )
            if not np.allclose(
                decoded,
                neural["y_pred_mmhg"].to_numpy(float),
                rtol=1e-6,
                atol=1e-3,
            ):
                raise ValueError(
                    "Fine-tuned mmHg predictions do not match the declared affine decoder"
                )
        if "y_pred_zscore" in neural and not np.allclose(
            pd.to_numeric(neural["y_pred_zscore"], errors="coerce"),
            pd.to_numeric(neural["estimator_output"], errors="coerce"),
            rtol=1e-7,
            atol=1e-7,
        ):
            raise ValueError("Fine-tuned estimator_output differs from y_pred_zscore")
    if not ridge.empty:
        if not np.allclose(
            pd.to_numeric(ridge["estimator_output"], errors="coerce"),
            ridge["y_pred_mmhg"].to_numpy(float),
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError("Ridge identity decoding changed the estimator output")
    if predictions["padding_policy"].astype(str).str.contains("reflect").any():
        raise ValueError("Reflection padding is not part of the corrected benchmark")
    padding = predictions["padding_required"].astype(str).str.lower().eq("true")
    if padding[predictions["dataset"].ne("PPG-BP")].any():
        raise ValueError("Only PPG-BP may require padding")


def validate_complete_benchmark_matrix(predictions: pd.DataFrame) -> None:
    """Require the complete active matrix and identical paired evaluation rows."""

    validate_prediction_artifact(predictions)
    datasets = set(predictions["dataset"].astype(str))
    if datasets != EXPECTED_DATASETS:
        raise ValueError(
            "Corrected aggregation requires exactly "
            f"{sorted(EXPECTED_DATASETS)}; observed {sorted(datasets)}"
        )
    for dataset, cohort in predictions.groupby("dataset", sort=True):
        methods = set(
            cohort[["model", "condition"]].drop_duplicates().itertuples(
                index=False, name=None
            )
        )
        if methods != EXPECTED_METHODS:
            missing = sorted(EXPECTED_METHODS - methods)
            extra = sorted(methods - EXPECTED_METHODS)
            raise ValueError(
                f"{dataset} active method matrix differs: missing={missing}, "
                f"extra={extra}"
            )
        if set(cohort["target"]) != {"sbp", "dbp"}:
            raise ValueError(f"{dataset} must contain separate SBP and DBP tasks")
        uses_demographics = cohort["condition"].isin(DEMOGRAPHIC_CONDITIONS)
        expected_demographics = EXPECTED_DEMOGRAPHIC_FIELDS[str(dataset)]
        if set(cohort.loc[uses_demographics, "demographic_fields"].astype(str)) != {
            expected_demographics
        }:
            raise ValueError(
                f"{dataset} demographic methods must use {expected_demographics}"
            )
        if set(
            cohort.loc[~uses_demographics, "demographic_fields"].astype(str)
        ) != {"none"}:
            raise ValueError(
                f"{dataset} non-demographic methods must declare demographic_fields=none"
            )

        evaluation_keys = [
            "subject_id",
            "measurement_id",
            "segment_id",
            "fold",
            "target",
        ]
        reference = cohort[
            (cohort["model"] == "demographics")
            & (cohort["condition"] == "demographics")
        ]
        if reference.duplicated(evaluation_keys).any():
            raise ValueError(f"{dataset} demographics reference has duplicate rows")
        if reference["seed"].nunique() != 1:
            raise ValueError(
                f"{dataset} deterministic Ridge methods must contain one seed"
            )
        reference = reference.sort_values(evaluation_keys).reset_index(drop=True)
        for (model, condition), method in cohort.groupby(
            ["model", "condition"], sort=True
        ):
            is_neural = condition in NEURAL_CONDITIONS
            observed_seeds = set(method["seed"].astype(int))
            if is_neural and observed_seeds != NEURAL_SEEDS:
                raise ValueError(
                    f"{dataset} {model}/{condition} must contain seeds 17, 23, and 42"
                )
            if not is_neural and len(observed_seeds) != 1:
                raise ValueError(
                    f"{dataset} {model}/{condition} is deterministic and must "
                    "contain one seed"
                )
            per_seed_duplicates = method.duplicated([*evaluation_keys, "seed"])
            if per_seed_duplicates.any():
                raise ValueError(
                    f"{dataset} {model}/{condition} contains duplicate evaluation rows"
                )
            expected_seed_count = len(NEURAL_SEEDS) if is_neural else 1
            seed_counts = method.groupby(
                evaluation_keys, dropna=False
            )["seed"].nunique()
            if not seed_counts.eq(expected_seed_count).all():
                raise ValueError(
                    f"{dataset} {model}/{condition} does not contain the expected "
                    "seed set for every evaluation row"
                )
            if method.groupby(evaluation_keys, dropna=False)[
                "y_true_mmhg"
            ].nunique(dropna=False).gt(1).any():
                raise ValueError(
                    f"{dataset} {model}/{condition} changed target labels across seeds"
                )
            method = (
                method.drop_duplicates(evaluation_keys)
                .sort_values(evaluation_keys)
                .reset_index(drop=True)
            )
            if not method[evaluation_keys].equals(reference[evaluation_keys]):
                raise ValueError(
                    f"{dataset} {model}/{condition} does not use the identical "
                    "evaluation rows"
                )
            if not np.allclose(
                method["y_true_mmhg"].to_numpy(float),
                reference["y_true_mmhg"].to_numpy(float),
                rtol=0,
                atol=1e-10,
            ):
                raise ValueError(
                    f"{dataset} {model}/{condition} changed the target labels"
                )


UNIT_GROUP_COLUMNS = [
    "dataset",
    "source",
    "subject_id",
    "measurement_id",
    "fold",
    "model",
    "condition",
    "target",
    "preprocessing_policy",
    "padding_policy",
    "padding_required",
    "pretraining_overlap",
    "source_fidelity",
    "contract_version",
    "demographic_fields",
]
METRIC_NAMES = (
    "mae",
    "rmse",
    "bias",
    "error_std",
    "r2",
    "pearson",
    "calibration_slope",
    "calibration_intercept",
)


def aggregate_seed_prediction_units(
    predictions: pd.DataFrame,
    additional_conditions: set[str] | None = None,
) -> pd.DataFrame:
    """Create scoring units while keeping every neural training seed separate."""

    validate_prediction_artifact(predictions, additional_conditions)
    data = predictions.copy()
    ppgbp = data["dataset"].eq("PPG-BP")
    data.loc[ppgbp, "measurement_id"] = data.loc[ppgbp, "subject_id"].astype(str)
    aggregated = data.groupby(
        [*UNIT_GROUP_COLUMNS, "seed"], as_index=False, dropna=False
    ).agg(
        y_true_mmhg=("y_true_mmhg", "mean"),
        y_pred_mmhg=("y_pred_mmhg", "mean"),
        n_segments=("segment_id", "nunique"),
    )
    aggregated["n_seeds"] = 1
    aggregated["aggregation_mode"] = np.where(
        aggregated["condition"].isin(NEURAL_CONDITIONS),
        "single_neural_seed",
        "single_deterministic_fit",
    )
    return aggregated


def aggregate_prediction_units(
    predictions: pd.DataFrame,
    additional_conditions: set[str] | None = None,
) -> pd.DataFrame:
    """Create diagnostic units, averaging neural predictions across seeds.

    These units are intentionally used only for prediction-level diagnostic
    figures. Primary neural metrics are the mean of metrics calculated for the
    three independently trained seeds, not metrics of this seed ensemble.
    """

    seed_units = aggregate_seed_prediction_units(predictions, additional_conditions)
    aggregated = seed_units.groupby(
        UNIT_GROUP_COLUMNS, as_index=False, dropna=False
    ).agg(
        y_true_mmhg=("y_true_mmhg", "mean"),
        y_pred_mmhg=("y_pred_mmhg", "mean"),
        n_segments=("n_segments", "max"),
        n_seeds=("seed", "nunique"),
    )
    neural = aggregated["condition"].isin(NEURAL_CONDITIONS)
    if not aggregated.loc[neural, "n_seeds"].eq(len(NEURAL_SEEDS)).all():
        raise ValueError(
            "Diagnostic neural predictions require all three training seeds"
        )
    active_ridge = aggregated["condition"].isin(ACTIVE_CONDITIONS - NEURAL_CONDITIONS)
    if not aggregated.loc[active_ridge, "n_seeds"].eq(1).all():
        raise ValueError("Deterministic Ridge predictions must contain one seed")
    aggregated["aggregation_mode"] = np.select(
        (
            neural,
            aggregated["n_seeds"].eq(1),
        ),
        (
            "diagnostic_mean_prediction_across_3_training_seeds",
            "single_deterministic_fit",
        ),
        default="diagnostic_mean_prediction_across_runs",
    )
    return aggregated


def _weighted_correlation(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    weight = weight / weight.sum()
    x_centered = x - np.sum(weight * x)
    y_centered = y - np.sum(weight * y)
    denominator = np.sqrt(np.sum(weight * x_centered**2) * np.sum(weight * y_centered**2))
    return float(np.sum(weight * x_centered * y_centered) / denominator) if denominator > 0 else np.nan


def participant_balanced_metrics(data: pd.DataFrame) -> dict[str, float]:
    if data.empty:
        raise ValueError("Cannot score an empty prediction group")
    counts = data.groupby("subject_id")["measurement_id"].transform("count").to_numpy(float)
    weights = 1.0 / counts
    weights /= weights.sum()
    y_true = data["y_true_mmhg"].to_numpy(float)
    y_pred = data["y_pred_mmhg"].to_numpy(float)
    error = y_pred - y_true
    mae = float(np.sum(weights * np.abs(error)))
    mse = float(np.sum(weights * error**2))
    true_mean = float(np.sum(weights * y_true))
    total = float(np.sum(weights * (y_true - true_mean) ** 2))
    slope, intercept = (np.nan, np.nan)
    if len(data) >= 2 and np.ptp(y_pred) > 0:
        slope, intercept = np.polyfit(y_pred, y_true, 1, w=np.sqrt(weights))
    return {
        "n_subjects": float(data["subject_id"].nunique()),
        "n_measurements": float(len(data)),
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "bias": float(np.sum(weights * error)),
        "error_std": float(np.sqrt(np.sum(weights * (error - np.sum(weights * error)) ** 2))),
        "r2": float(1 - mse / total) if total > 0 else np.nan,
        "pearson": _weighted_correlation(y_true, y_pred, weights),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
    }


def _bootstrap_intervals(
    data: pd.DataFrame,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    if replicates <= 0:
        return {
            f"{metric}_ci_{side}": np.nan
            for metric in METRIC_NAMES
            for side in ("low", "high")
        }

    # Participant-balanced metrics give every subject equal total weight and
    # every measurement within a subject equal weight. All registered metrics
    # can therefore be reconstructed from per-subject first and second moments.
    # Resampling these compact sufficient statistics is exactly equivalent to
    # rebuilding a DataFrame for every bootstrap draw, but avoids hundreds of
    # thousands of groupby/copy/concat operations.
    subject_moments = []
    for _, subject in data.assign(
        _bootstrap_subject=data["subject_id"].astype(str)
    ).groupby("_bootstrap_subject", sort=True):
        y_true = subject["y_true_mmhg"].to_numpy(float)
        y_pred = subject["y_pred_mmhg"].to_numpy(float)
        error = y_pred - y_true
        subject_moments.append(
            [
                np.mean(np.abs(error)),
                np.mean(error**2),
                np.mean(error),
                np.mean(y_true),
                np.mean(y_true**2),
                np.mean(y_pred),
                np.mean(y_pred**2),
                np.mean(y_true * y_pred),
            ]
        )
    moments = np.asarray(subject_moments, dtype=float)
    n_subjects = len(moments)
    rng = np.random.default_rng(seed)
    sampled_subjects = rng.choice(
        n_subjects,
        size=(replicates, n_subjects),
        replace=True,
    )
    sampled = moments[sampled_subjects].mean(axis=1)
    (
        mae,
        mse,
        bias,
        true_mean,
        true_second_moment,
        predicted_mean,
        predicted_second_moment,
        true_predicted_moment,
    ) = sampled.T
    true_variance = np.maximum(true_second_moment - true_mean**2, 0.0)
    predicted_variance = np.maximum(
        predicted_second_moment - predicted_mean**2, 0.0
    )
    covariance = true_predicted_moment - true_mean * predicted_mean
    correlation_denominator = np.sqrt(true_variance * predicted_variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(predicted_variance > 0, covariance / predicted_variance, np.nan)
        samples = {
            "mae": mae,
            "rmse": np.sqrt(np.maximum(mse, 0.0)),
            "bias": bias,
            "error_std": np.sqrt(np.maximum(mse - bias**2, 0.0)),
            "r2": np.where(true_variance > 0, 1 - mse / true_variance, np.nan),
            "pearson": np.where(
                correlation_denominator > 0,
                covariance / correlation_denominator,
                np.nan,
            ),
            "calibration_slope": slope,
            "calibration_intercept": true_mean - slope * predicted_mean,
        }

    alpha = (1 - confidence) / 2
    result = {}
    for metric, values in samples.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            low, high = np.quantile(finite, [alpha, 1 - alpha])
            result[f"{metric}_ci_low"] = float(low)
            result[f"{metric}_ci_high"] = float(high)
        else:
            result[f"{metric}_ci_low"] = np.nan
            result[f"{metric}_ci_high"] = np.nan
    return result


def _bootstrap_seed_mean_intervals(
    data: pd.DataFrame,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    """Bootstrap participants within each seed, then average seed statistics.

    The same participant resample is used for all seeds because their outer-test
    rows are paired. Predictions are never averaged before calculating a seed's
    statistic, so this cannot accidentally report an ensemble advantage.
    """

    if replicates <= 0:
        return {
            f"{metric}_ci_{side}": np.nan
            for metric in METRIC_NAMES
            for side in ("low", "high")
        }
    seed_groups = [
        (int(training_seed), group.copy())
        for training_seed, group in data.groupby("seed", sort=True)
    ]
    if not seed_groups:
        raise ValueError("Cannot bootstrap an empty seed group")
    subjects = sorted(seed_groups[0][1]["subject_id"].astype(str).unique())
    expected_subjects = set(subjects)
    evaluation_keys = [
        column
        for column in ("subject_id", "measurement_id", "fold")
        if column in data.columns
    ]
    reference_rows = seed_groups[0][1].sort_values(evaluation_keys).reset_index(
        drop=True
    )
    moments_by_seed: list[np.ndarray] = []
    for training_seed, group in seed_groups:
        if set(group["subject_id"].astype(str)) != expected_subjects:
            raise ValueError(
                f"Training seed {training_seed} does not contain identical subjects"
            )
        ordered = group.sort_values(evaluation_keys).reset_index(drop=True)
        if not ordered[evaluation_keys].equals(reference_rows[evaluation_keys]):
            raise ValueError(
                f"Training seed {training_seed} does not contain identical "
                "evaluation units"
            )
        if not np.allclose(
            ordered["y_true_mmhg"].to_numpy(float),
            reference_rows["y_true_mmhg"].to_numpy(float),
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError(
                f"Training seed {training_seed} changed evaluation targets"
            )
        subject_moments: dict[str, list[float]] = {}
        for subject_id, subject in group.assign(
            _bootstrap_subject=group["subject_id"].astype(str)
        ).groupby("_bootstrap_subject", sort=True):
            y_true = subject["y_true_mmhg"].to_numpy(float)
            y_pred = subject["y_pred_mmhg"].to_numpy(float)
            error = y_pred - y_true
            subject_moments[str(subject_id)] = [
                np.mean(np.abs(error)),
                np.mean(error**2),
                np.mean(error),
                np.mean(y_true),
                np.mean(y_true**2),
                np.mean(y_pred),
                np.mean(y_pred**2),
                np.mean(y_true * y_pred),
            ]
        moments_by_seed.append(
            np.asarray([subject_moments[subject] for subject in subjects], dtype=float)
        )

    moments = np.stack(moments_by_seed, axis=0)
    n_subjects = len(subjects)
    sampled_subjects = np.random.default_rng(seed).choice(
        n_subjects,
        size=(replicates, n_subjects),
        replace=True,
    )
    # Shape: training seed x bootstrap replicate x moment.
    sampled = moments[:, sampled_subjects, :].mean(axis=2)
    (
        mae,
        mse,
        bias,
        true_mean,
        true_second_moment,
        predicted_mean,
        predicted_second_moment,
        true_predicted_moment,
    ) = np.moveaxis(sampled, -1, 0)
    true_variance = np.maximum(true_second_moment - true_mean**2, 0.0)
    predicted_variance = np.maximum(
        predicted_second_moment - predicted_mean**2, 0.0
    )
    covariance = true_predicted_moment - true_mean * predicted_mean
    correlation_denominator = np.sqrt(true_variance * predicted_variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(
            predicted_variance > 0, covariance / predicted_variance, np.nan
        )
        per_seed_samples = {
            "mae": mae,
            "rmse": np.sqrt(np.maximum(mse, 0.0)),
            "bias": bias,
            "error_std": np.sqrt(np.maximum(mse - bias**2, 0.0)),
            "r2": np.where(true_variance > 0, 1 - mse / true_variance, np.nan),
            "pearson": np.where(
                correlation_denominator > 0,
                covariance / correlation_denominator,
                np.nan,
            ),
            "calibration_slope": slope,
            "calibration_intercept": true_mean - slope * predicted_mean,
        }

    alpha = (1 - confidence) / 2
    result: dict[str, float] = {}
    for metric, values in per_seed_samples.items():
        # Axis zero is the independently trained seed. This is a mean of
        # seed-specific metrics, not a metric calculated from mean predictions.
        combined = np.mean(values, axis=0)
        finite = combined[np.isfinite(combined)]
        if finite.size:
            low, high = np.quantile(finite, [alpha, 1 - alpha])
            result[f"{metric}_ci_low"] = float(low)
            result[f"{metric}_ci_high"] = float(high)
        else:
            result[f"{metric}_ci_low"] = np.nan
            result[f"{metric}_ci_high"] = np.nan
    return result


def summarize_benchmark(
    predictions: pd.DataFrame,
    bootstrap_replicates: int = 2000,
    bootstrap_confidence: float = 0.95,
    seed: int = 20260715,
    additional_conditions: set[str] | None = None,
    return_seed_details: bool = False,
) -> (
    tuple[pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """Summarize deterministic fits and independent neural training seeds.

    The first returned table contains averaged-seed predictions for diagnostics
    only. Neural point estimates and confidence intervals in ``metrics`` are
    calculated seed by seed and then averaged. With ``return_seed_details``, the
    seed-specific scoring units and metrics are returned as the third and fourth
    values for audit and reporting of training-seed variability.
    """

    seed_units = aggregate_seed_prediction_units(predictions, additional_conditions)
    units = aggregate_prediction_units(predictions, additional_conditions)
    seed_records = []
    grouping = [
        "dataset", "model", "condition", "target", "preprocessing_policy",
        "padding_policy", "padding_required", "pretraining_overlap",
        "source_fidelity", "contract_version", "demographic_fields",
    ]
    for index, (keys, group) in enumerate(
        seed_units.groupby([*grouping, "seed"], dropna=False)
    ):
        condition = str(keys[grouping.index("condition")])
        seed_records.append(
            {
                **dict(zip([*grouping, "seed"], keys)),
                **participant_balanced_metrics(group),
                **_bootstrap_intervals(
                    group, bootstrap_replicates, bootstrap_confidence, seed + index
                ),
                "n_seeds": 1,
                "aggregation_mode": (
                    "single_neural_seed"
                    if condition in NEURAL_CONDITIONS
                    else "single_deterministic_fit"
                ),
                "bootstrap_mode": "participant_resample_within_seed",
            }
        )
    seed_metrics = pd.DataFrame(seed_records).sort_values(
        [*grouping, "seed"]
    ).reset_index(drop=True)

    records = []
    for index, (keys, group) in enumerate(seed_units.groupby(grouping, dropna=False)):
        condition = str(keys[grouping.index("condition")])
        matching = seed_metrics
        for column, value in zip(grouping, keys):
            matching = matching[
                matching[column].eq(value)
                if not pd.isna(value)
                else matching[column].isna()
            ]
        n_seeds = int(matching["seed"].nunique())
        expected_n_seeds = len(NEURAL_SEEDS) if condition in NEURAL_CONDITIONS else 1
        if n_seeds != expected_n_seeds:
            raise ValueError(
                f"{condition} requires {expected_n_seeds} seed-specific metric rows; "
                f"observed {n_seeds}"
            )
        n_subjects = matching["n_subjects"].to_numpy(float)
        n_measurements = matching["n_measurements"].to_numpy(float)
        if not np.allclose(n_subjects, n_subjects[0]) or not np.allclose(
            n_measurements, n_measurements[0]
        ):
            raise ValueError("Training seeds do not use identical evaluation units")
        record: dict[str, object] = {
            **dict(zip(grouping, keys)),
            "n_subjects": float(n_subjects[0]),
            "n_measurements": float(n_measurements[0]),
            "n_seeds": n_seeds,
            "aggregation_mode": (
                "mean_of_3_seed_specific_metrics"
                if condition in NEURAL_CONDITIONS
                else "single_deterministic_fit"
            ),
            "bootstrap_mode": (
                "participant_resample_within_seed_then_mean_seed_metrics"
                if condition in NEURAL_CONDITIONS
                else "participant_resample_within_seed"
            ),
        }
        for metric in METRIC_NAMES:
            values = matching[metric].to_numpy(float)
            record[metric] = float(np.mean(values))
            record[f"{metric}_seed_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        record.update(
            _bootstrap_seed_mean_intervals(
                group,
                bootstrap_replicates,
                bootstrap_confidence,
                seed + 100_000 + index,
            )
        )
        records.append(record)
    metrics = pd.DataFrame(records).sort_values(grouping).reset_index(drop=True)
    if return_seed_details:
        return units, metrics, seed_units, seed_metrics
    return units, metrics


def incremental_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    index_columns = [
        "dataset", "model", "condition", "target", "preprocessing_policy",
        "padding_policy", "demographic_fields",
    ]
    lookup = metrics.set_index(index_columns)
    records = []
    for (
        dataset,
        model,
        condition,
        target,
        preprocessing_policy,
        padding_policy,
        demographic_fields,
    ), row in lookup.iterrows():
        comparisons = []
        if condition == "frozen_demographics":
            comparisons.append(("demographics_added_to_frozen", "frozen"))
        elif condition == "finetuned_demographics":
            comparisons.append(
                ("demographics_added_to_finetuned", "finetuned")
            )
        elif condition == "ppg_features_demographics":
            comparisons.append(("demographics_added_to_ppg_features", "ppg_features"))
        for effect, reference_condition in comparisons:
            reference_demographics = "none" if reference_condition in {
                "frozen", "finetuned", "ppg_features"
            } else demographic_fields
            key = (
                dataset,
                model,
                reference_condition,
                target,
                preprocessing_policy,
                padding_policy,
                reference_demographics,
            )
            if key not in lookup.index:
                continue
            reference = lookup.loc[key]
            records.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "target": target,
                    "preprocessing_policy": preprocessing_policy,
                    "padding_policy": padding_policy,
                    "demographic_fields": demographic_fields,
                    "effect": effect,
                    "candidate_condition": condition,
                    "reference_condition": reference_condition,
                    "delta_mae": row["mae"] - reference["mae"],
                    "delta_rmse": row["rmse"] - reference["rmse"],
                }
            )
    return pd.DataFrame(records)


def _slug(value: object) -> str:
    return str(value).lower().replace(" ", "_").replace("/", "_")


def method_display_label(
    model: str,
    condition: str,
    demographic_fields: str = "",
) -> str:
    """Human-readable label using the benchmark demographic shorthand."""

    display = MODEL_DISPLAY.get(str(model), str(model))
    demographic_marker = "†" if str(demographic_fields).strip() == "age/sex" else ""
    if condition == "demographics":
        return f"Demographics only{demographic_marker}"
    if condition == "ppg_features":
        return "PPG features"
    if condition == "ppg_features_demographics":
        return f"PPG features + Demo{demographic_marker}"
    if condition == "frozen":
        return f"Frozen {display}"
    if condition == "frozen_demographics":
        return f"Frozen {display} + Demo{demographic_marker}"
    if condition == "finetuned":
        return f"Fine-tuned {display}"
    if condition == "finetuned_demographics":
        return f"Fine-tuned {display} + Demo{demographic_marker}"
    raise ValueError(f"Unsupported active condition: {condition}")


def heatmap_method_display_label(model: str, condition: str) -> str:
    """Collapse cohort-specific demographic vectors into one heatmap row."""

    return method_display_label(model, condition, "age/sex/BMI")


def _has_age_sex_only_demographics(data: pd.DataFrame) -> bool:
    return bool(
        (
            data["condition"].isin(DEMOGRAPHIC_CONDITIONS)
            & data["demographic_fields"].astype(str).eq("age/sex")
        ).any()
    )


def _add_demographic_figure_note(figure: object) -> tuple[float, float, float, float]:
    figure.text(
        0.01,
        0.01,
        "† Demo uses age and sex only; unmarked Demo uses age, sex, and BMI.",
        fontsize=9,
        ha="left",
    )
    return (0, 0.05, 1, 0.96)


def _heatmap_method_order() -> list[str]:
    order = [heatmap_method_display_label("demographics", "demographics")]
    for model, without_demo, with_demo, _ in PAIRED_CONFIGURATIONS:
        order.extend(
            (
                heatmap_method_display_label(model, without_demo),
                heatmap_method_display_label(model, with_demo),
            )
        )
    return order


def _method_color(condition: str) -> str:
    if condition == "demographics":
        return "#59A14F"
    if condition.endswith("demographics"):
        return "#F28E2B"
    return "#4E79A7"


def _metric_error(row: pd.Series) -> np.ndarray:
    return np.asarray(
        [
            [max(0.0, float(row["mae"]) - float(row["mae_ci_low"]))],
            [max(0.0, float(row["mae_ci_high"]) - float(row["mae"]))],
        ]
    )


def _metric_annotation(row: pd.Series) -> str:
    text = f"{float(row['mae']):.2f}"
    if str(row["condition"]) in NEURAL_CONDITIONS:
        text += f"\n±{float(row['mae_seed_sd']):.2f}"
    return text


def _add_neural_metric_figure_note(
    figure: object,
    *,
    diagnostic_predictions: bool,
    y: float = 0.035,
) -> None:
    suffix = (
        " Prediction-level neural panels show the mean prediction across seeds "
        "for visualization only."
        if diagnostic_predictions
        else ""
    )
    figure.text(
        0.01,
        y,
        "Fine-tuned MAE is the mean of seed-specific scores for seeds 17, 23, "
        f"and 42; ± denotes training-seed SD.{suffix}",
        fontsize=8.5,
        ha="left",
    )


def _validate_visualization_aggregation(
    units: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    required = {"aggregation_mode", "n_seeds"}
    for label, frame in (("diagnostic predictions", units), ("metrics", metrics)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"{label} lack explicit aggregation metadata: {missing}"
            )
    neural_metrics = metrics[metrics["condition"].isin(NEURAL_CONDITIONS)]
    if not neural_metrics.empty and (
        set(neural_metrics["aggregation_mode"].astype(str))
        != {"mean_of_3_seed_specific_metrics"}
        or not neural_metrics["n_seeds"].eq(3).all()
    ):
        raise ValueError(
            "Fine-tuned visualizations require mean seed-specific metrics from "
            "three training seeds"
        )
    neural_units = units[units["condition"].isin(NEURAL_CONDITIONS)]
    if not neural_units.empty and (
        set(neural_units["aggregation_mode"].astype(str))
        != {"diagnostic_mean_prediction_across_3_training_seeds"}
        or not neural_units["n_seeds"].eq(3).all()
    ):
        raise ValueError(
            "Fine-tuned diagnostic plots require explicitly labeled mean-seed "
            "predictions"
        )


def _paired_mae_plot(
    summary: pd.DataFrame,
    dataset: str,
    output_root: Path,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure, axis = plt.subplots(
        figsize=(max(14, 1.8 * (len(PAIRED_CONFIGURATIONS) + 1)), 7.5)
    )
    centers = np.arange(len(PAIRED_CONFIGURATIONS) + 1, dtype=float)
    width = 0.18
    demo_marker = "†" if _has_age_sex_only_demographics(summary) else ""
    target_hatches = {"sbp": "", "dbp": "///"}

    for target_index, target in enumerate(("sbp", "dbp")):
        demo_only = summary[
            summary["model"].eq("demographics")
            & summary["condition"].eq("demographics")
            & summary["target"].eq(target)
        ].iloc[0]
        position = centers[0] + (-0.5 + target_index) * width
        axis.bar(
            position,
            demo_only["mae"],
            width=width,
            color=_method_color("demographics"),
            hatch=target_hatches[target],
            yerr=_metric_error(demo_only),
            capsize=3,
            edgecolor="#333333",
            linewidth=0.6,
        )
        axis.text(
            position,
            float(demo_only["mae_ci_high"]) + 0.22,
            _metric_annotation(demo_only),
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    for index, (model, without_demo, with_demo, _) in enumerate(
        PAIRED_CONFIGURATIONS, start=1
    ):
        for target_index, target in enumerate(("sbp", "dbp")):
            for demo_index, condition in enumerate((without_demo, with_demo)):
                offset = (-1.5 + target_index * 2 + demo_index) * width
                position = centers[index] + offset
                row = summary[
                    summary["model"].eq(model)
                    & summary["condition"].eq(condition)
                    & summary["target"].eq(target)
                ].iloc[0]
                axis.bar(
                    position,
                    row["mae"],
                    width=width,
                    color=_method_color(condition),
                    hatch=target_hatches[target],
                    yerr=_metric_error(row),
                    capsize=3,
                    edgecolor="#333333",
                    linewidth=0.6,
                )
                axis.text(
                    position,
                    float(row["mae_ci_high"]) + 0.22,
                    _metric_annotation(row),
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )

    axis.set_xticks(
        centers,
        (
            "Demographics\nonly",
            *(label.replace(" ", "\n", 1) for *_, label in PAIRED_CONFIGURATIONS),
        ),
    )
    axis.set_ylabel("MAE (mmHg)")
    axis.set_ylim(
        0,
        float(summary["mae_ci_high"].max())
        + max(1.5, float(summary["mae_ci_high"].max()) * 0.08),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    figure.suptitle(
        f"{dataset} · SBP and DBP MAE with 95% confidence intervals",
        y=0.98,
    )
    figure.legend(
        handles=(
            Patch(color=_method_color("frozen"), label="No Demo"),
            Patch(
                color=_method_color("frozen_demographics"),
                label=f"+ Demo{demo_marker}",
            ),
            Patch(
                color=_method_color("demographics"),
                label=f"Demo only{demo_marker}",
            ),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        frameon=False,
        ncols=3,
    )
    figure.legend(
        handles=(
            Patch(
                facecolor="white",
                edgecolor="#333333",
                label="SBP",
            ),
            Patch(
                facecolor="white",
                edgecolor="#333333",
                hatch=target_hatches["dbp"],
                label="DBP",
            ),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        frameon=False,
        ncols=2,
    )
    bottom = 0.08
    if _has_age_sex_only_demographics(summary):
        _add_demographic_figure_note(figure)
    _add_neural_metric_figure_note(
        figure, diagnostic_predictions=False, y=0.035
    )
    figure.tight_layout(rect=(0, bottom, 1, 0.81))
    path = output_root / f"bars_{_slug(dataset)}.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _ordered_diagnostic_methods(
    group: pd.DataFrame,
) -> list[tuple[tuple[str, str, str], pd.DataFrame] | None]:
    grouped = {
        (str(model), str(condition), str(demographics)): data
        for (model, condition, demographics), data in group.groupby(
            ["model", "condition", "demographic_fields"], dropna=False
        )
    }

    def select(model: str, condition: str):
        matches = [
            (key, data)
            for key, data in grouped.items()
            if key[0] == model and key[1] == condition
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one diagnostic method for {model}/{condition}; "
                f"observed {len(matches)}"
            )
        return matches[0]

    ordered: list[tuple[tuple[str, str, str], pd.DataFrame] | None] = [
        select("demographics", "demographics"),
        None,
    ]
    for model, without_demo, with_demo, _ in PAIRED_CONFIGURATIONS:
        ordered.extend((select(model, without_demo), select(model, with_demo)))
    return ordered


def _demographics_only_spread_plot(
    units: pd.DataFrame,
    output_root: Path,
) -> Path:
    import matplotlib.pyplot as plt

    demographics = units[
        units["model"].eq("demographics")
        & units["condition"].eq("demographics")
    ]
    subject_values = demographics.groupby(
        ["dataset", "target", "subject_id"],
        as_index=False,
    )[["y_true_mmhg", "y_pred_mmhg"]].mean()
    spread = subject_values.groupby(["dataset", "target"]).agg(
        reference_sd=("y_true_mmhg", "std"),
        prediction_sd=("y_pred_mmhg", "std"),
    )
    order = pd.MultiIndex.from_product(
        [
            ("PPG-BP", "PulseDB-Vital", "PulseDB-MIMIC", "BUT PPG"),
            ("sbp", "dbp"),
        ],
        names=("dataset", "target"),
    )
    spread = spread.reindex(order)
    if spread.isna().any().any():
        raise ValueError("Demographics-only spread plot requires all cohorts")

    positions = np.arange(len(spread), dtype=float)
    width = 0.36
    figure, axis = plt.subplots(figsize=(14, 6.5))
    axis.bar(
        positions - width / 2,
        spread["reference_sd"],
        width=width,
        color="#9C9C9C",
        label="Reference BP SD",
    )
    axis.bar(
        positions + width / 2,
        spread["prediction_sd"],
        width=width,
        color=_method_color("demographics"),
        label="Demo-only prediction SD",
    )
    for index, (_, row) in enumerate(spread.iterrows()):
        ratio = float(row["prediction_sd"] / row["reference_sd"])
        axis.text(
            positions[index] + width / 2,
            float(row["prediction_sd"]) + 0.35,
            f"{ratio:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_xticks(
        positions,
        [
            f"{dataset}\n{str(target).upper()}"
            + ("†" if dataset == "PulseDB-MIMIC" else "")
            for dataset, target in spread.index
        ],
    )
    axis.set_ylabel("Subject-level standard deviation (mmHg)")
    axis.set_title(
        "Demographics-only prediction spread versus reference BP spread"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncols=2)
    _add_demographic_figure_note(figure)
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    path = output_root / "demographics_only_prediction_spread.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def generate_visualizations(
    units: pd.DataFrame,
    metrics: pd.DataFrame,
    output_root: str | Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _validate_visualization_aggregation(units, metrics)
    paths: list[Path] = []
    labels = metrics.apply(
        lambda row: method_display_label(
            row["model"], row["condition"], row["demographic_fields"]
        ),
        axis=1,
    )
    metrics = metrics.assign(label=labels)
    for stale in (
        *output_root.glob("bars_*_sbp.png"),
        *output_root.glob("bars_*_dbp.png"),
    ):
        stale.unlink()
    for dataset, summary in metrics.groupby("dataset"):
        paths.append(_paired_mae_plot(summary, dataset, output_root))
    for (dataset, target), summary in metrics.groupby(["dataset", "target"]):
        group = units[(units["dataset"] == dataset) & (units["target"] == target)].copy()
        methods = _ordered_diagnostic_methods(group)
        metric_lookup = summary.set_index(["model", "condition"])["mae"]
        columns = 2
        rows = len(methods) // columns
        all_true = group["y_true_mmhg"].to_numpy(float)
        all_predicted = group["y_pred_mmhg"].to_numpy(float)
        all_errors = all_predicted - all_true
        all_means = (all_predicted + all_true) / 2
        scatter_bounds = (
            min(all_true.min(), all_predicted.min()),
            max(all_true.max(), all_predicted.max()),
        )
        error_bounds = (all_errors.min(), all_errors.max())
        mean_bounds = (all_means.min(), all_means.max())
        error_bins = np.linspace(error_bounds[0], error_bounds[1], 25)
        for kind in ("scatter", "bland_altman", "errors"):
            figure, axes = plt.subplots(
                rows,
                columns,
                figsize=(5 * columns, 3.5 * rows),
                squeeze=False,
            )
            for axis in axes.reshape(-1):
                axis.set_visible(False)
            for axis, method_entry in zip(
                axes.reshape(-1), methods
            ):
                if method_entry is None:
                    continue
                (model, condition, demographic_fields), method = method_entry
                axis.set_visible(True)
                true = method["y_true_mmhg"].to_numpy(float)
                predicted = method["y_pred_mmhg"].to_numpy(float)
                color = _method_color(condition)
                if kind == "scatter":
                    axis.scatter(true, predicted, s=10, alpha=0.45, color=color)
                    axis.plot(
                        scatter_bounds,
                        scatter_bounds,
                        color="black",
                        linestyle="--",
                        linewidth=1,
                        label="identity",
                    )
                    if len(true) >= 2 and np.ptp(true) > 0:
                        calibration_slope, calibration_intercept = np.polyfit(
                            true, predicted, 1
                        )
                        axis.plot(
                            scatter_bounds,
                            calibration_slope * np.asarray(scatter_bounds)
                            + calibration_intercept,
                            color="tab:red",
                            linewidth=1.2,
                            label="unweighted trend",
                        )
                    axis.legend(fontsize=7)
                    axis.set_xlabel("Reference (mmHg)")
                    axis.set_ylabel("Predicted (mmHg)")
                    axis.set_xlim(scatter_bounds)
                    axis.set_ylim(scatter_bounds)
                elif kind == "bland_altman":
                    mean = (true + predicted) / 2
                    difference = predicted - true
                    bias = difference.mean()
                    sd = difference.std(ddof=1) if len(difference) > 1 else 0
                    axis.scatter(mean, difference, s=10, alpha=0.45, color=color)
                    for value, style in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
                        axis.axhline(value, color="black", linestyle=style, linewidth=1)
                    axis.set_xlabel("Mean BP (mmHg)")
                    axis.set_ylabel("Prediction − reference")
                    axis.set_xlim(mean_bounds)
                    axis.set_ylim(error_bounds)
                else:
                    axis.hist(
                        predicted - true,
                        bins=error_bins,
                        alpha=0.8,
                        color=color,
                    )
                    axis.axvline(0, color="black", linestyle="--", linewidth=1)
                    axis.set_xlabel("Error (mmHg)")
                    axis.set_ylabel("Count")
                    axis.set_xlim(error_bounds)
                mae = float(metric_lookup.loc[(model, condition)])
                axis.set_title(
                    f"{method_display_label(model, condition, demographic_fields)}"
                    f"\nMAE {mae:.2f} mmHg",
                    fontsize=9,
                )
                axis.grid(alpha=0.2)
            figure.suptitle(
                f"{dataset} {target.upper()} · paired {kind.replace('_', ' ')}"
            )
            if _has_age_sex_only_demographics(group):
                _add_demographic_figure_note(figure)
            _add_neural_metric_figure_note(
                figure, diagnostic_predictions=True, y=0.035
            )
            rectangle = (0, 0.08, 1, 0.96)
            figure.tight_layout(rect=rectangle)
            path = output_root / f"{kind}_{_slug(dataset)}_{target}.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            paths.append(path)

    paths.append(_demographics_only_spread_plot(units, output_root))

    heatmap_data = metrics.assign(
        label=metrics.apply(
            lambda row: heatmap_method_display_label(
                row["model"], row["condition"]
            ),
            axis=1,
        )
    )
    heatmap = heatmap_data.pivot_table(
        index="label",
        columns=["dataset", "target"],
        values="mae",
    )
    column_order = pd.MultiIndex.from_product(
        [
            ("PPG-BP", "PulseDB-Vital", "PulseDB-MIMIC", "BUT PPG"),
            ("sbp", "dbp"),
        ],
        names=("dataset", "target"),
    )
    heatmap = heatmap.reindex(
        index=_heatmap_method_order(),
        columns=column_order,
    )
    if heatmap.isna().any().any():
        raise ValueError("MAE heatmap must contain every active method and cohort")
    age_sex_only_cells = {
        (row.label, row.dataset, row.target): (
            row.condition
            in DEMOGRAPHIC_CONDITIONS
            and str(row.demographic_fields) == "age/sex"
        )
        for row in heatmap_data.itertuples()
    }
    figure, axis = plt.subplots(figsize=(max(8, heatmap.shape[1] * 1.5), max(5, heatmap.shape[0] * 0.45)))
    image = axis.imshow(heatmap.to_numpy(), aspect="auto", cmap="viridis_r")
    axis.set_xticks(np.arange(heatmap.shape[1]), [" · ".join(map(str, value)) for value in heatmap.columns], rotation=45, ha="right")
    axis.set_yticks(np.arange(heatmap.shape[0]), heatmap.index)
    for row in range(heatmap.shape[0]):
        for column in range(heatmap.shape[1]):
            value = heatmap.iloc[row, column]
            if np.isfinite(value):
                dataset, target = heatmap.columns[column]
                marker = (
                    "†"
                    if age_sex_only_cells.get(
                        (heatmap.index[row], dataset, target), False
                    )
                    else ""
                )
                axis.text(
                    column,
                    row,
                    f"{value:.2f}{marker}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    figure.colorbar(image, ax=axis, label="MAE (mmHg)")
    axis.set_title("MAE comparison across cohorts and targets")
    _add_demographic_figure_note(figure)
    _add_neural_metric_figure_note(
        figure, diagnostic_predictions=False, y=0.035
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))
    path = output_root / "mae_heatmap.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    rank_data = heatmap_data.copy()
    rank_data["rank"] = rank_data.groupby(
        ["dataset", "target"]
    )["mae"].rank(method="min")
    rank_heatmap = rank_data.pivot_table(
        index="label",
        columns=["dataset", "target"],
        values="rank",
    ).reindex(index=_heatmap_method_order(), columns=column_order)
    if rank_heatmap.isna().any().any():
        raise ValueError("Rank heatmap must contain every active method and cohort")
    figure, axis = plt.subplots(
        figsize=(
            max(8, rank_heatmap.shape[1] * 1.5),
            max(5, rank_heatmap.shape[0] * 0.45),
        )
    )
    image = axis.imshow(
        rank_heatmap.to_numpy(),
        aspect="auto",
        cmap="viridis_r",
        vmin=1,
        vmax=len(_heatmap_method_order()),
    )
    axis.set_xticks(
        np.arange(rank_heatmap.shape[1]),
        [" · ".join(map(str, value)) for value in rank_heatmap.columns],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(np.arange(rank_heatmap.shape[0]), rank_heatmap.index)
    for row in range(rank_heatmap.shape[0]):
        for column in range(rank_heatmap.shape[1]):
            dataset, target = rank_heatmap.columns[column]
            marker = (
                "†"
                if age_sex_only_cells.get(
                    (rank_heatmap.index[row], dataset, target), False
                )
                else ""
            )
            axis.text(
                column,
                row,
                f"{int(rank_heatmap.iloc[row, column])}{marker}",
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="MAE rank (1 = best)")
    axis.set_title("Within-cohort MAE rank")
    _add_demographic_figure_note(figure)
    _add_neural_metric_figure_note(
        figure, diagnostic_predictions=False, y=0.035
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))
    path = output_root / "mae_rank_heatmap.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    effects = incremental_effects(metrics)
    effect_labels = {
        (model, with_demo): label
        for model, _, with_demo, label in PAIRED_CONFIGURATIONS
    }
    effects["label"] = [
        effect_labels.get((str(row.model), str(row.candidate_condition)))
        for row in effects.itertuples()
    ]
    if effects["label"].isna().any():
        raise ValueError("Demographic effects contain an unregistered comparison")
    delta_heatmap = effects.pivot_table(
        index="label",
        columns=["dataset", "target"],
        values="delta_mae",
    ).reindex(index=list(effect_labels.values()), columns=column_order)
    if delta_heatmap.isna().any().any():
        raise ValueError(
            "Demographic-effect heatmap must contain every paired comparison"
        )
    max_delta = max(0.25, float(np.abs(delta_heatmap.to_numpy()).max()))
    figure, axis = plt.subplots(
        figsize=(
            max(8, delta_heatmap.shape[1] * 1.5),
            max(4.5, delta_heatmap.shape[0] * 0.65),
        )
    )
    image = axis.imshow(
        delta_heatmap.to_numpy(),
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=-max_delta,
        vmax=max_delta,
    )
    axis.set_xticks(
        np.arange(delta_heatmap.shape[1]),
        [" · ".join(map(str, value)) for value in delta_heatmap.columns],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(np.arange(delta_heatmap.shape[0]), delta_heatmap.index)
    for row in range(delta_heatmap.shape[0]):
        for column in range(delta_heatmap.shape[1]):
            dataset, _ = delta_heatmap.columns[column]
            marker = "†" if dataset == "PulseDB-MIMIC" else ""
            axis.text(
                column,
                row,
                f"{delta_heatmap.iloc[row, column]:+.2f}{marker}",
                ha="center",
                va="center",
                fontsize=9,
            )
    figure.colorbar(
        image,
        ax=axis,
        label="ΔMAE after adding Demo (mmHg)",
    )
    axis.set_title(
        "Effect of demographics on MAE · negative values indicate improvement"
    )
    _add_demographic_figure_note(figure)
    _add_neural_metric_figure_note(
        figure, diagnostic_predictions=False, y=0.035
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))
    path = output_root / "demographic_mae_delta_heatmap.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    return paths
