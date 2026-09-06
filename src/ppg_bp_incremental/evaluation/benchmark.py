"""Participant-balanced benchmark metrics, uncertainty, and visualizations."""

from __future__ import annotations

from pathlib import Path

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
}
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
        for condition in ("frozen", "frozen_demographics")
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
)


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
    if set(predictions["estimator_output_scale"]) != {"raw_mmhg"}:
        raise ValueError("Active Ridge predictions must declare raw_mmhg estimator output")
    if set(predictions["decoder"]) != {"identity"}:
        raise ValueError("Active Ridge predictions must declare identity decoding")
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
    row_keys = ["segment_id", "fold", "seed", "target"]
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
        uses_demographics = cohort["condition"].isin(
            {
                "demographics",
                "ppg_features_demographics",
                "frozen_demographics",
            }
        )
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

        reference = cohort[
            (cohort["model"] == "demographics")
            & (cohort["condition"] == "demographics")
        ].sort_values(row_keys).reset_index(drop=True)
        if reference.duplicated(row_keys).any():
            raise ValueError(f"{dataset} demographics reference has duplicate rows")
        for (model, condition), method in cohort.groupby(
            ["model", "condition"], sort=True
        ):
            method = method.sort_values(row_keys).reset_index(drop=True)
            if not method[row_keys].equals(reference[row_keys]):
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


def aggregate_prediction_units(
    predictions: pd.DataFrame,
    additional_conditions: set[str] | None = None,
) -> pd.DataFrame:
    """Average PPG-BP recordings and neural seeds before primary scoring."""
    validate_prediction_artifact(predictions, additional_conditions)
    data = predictions.copy()
    ppgbp = data["dataset"].eq("PPG-BP")
    data.loc[ppgbp, "measurement_id"] = data.loc[ppgbp, "subject_id"].astype(str)
    unit_columns = [
        "dataset", "source", "subject_id", "measurement_id", "fold", "model",
        "condition", "target", "preprocessing_policy", "padding_policy",
        "padding_required", "pretraining_overlap", "source_fidelity",
        "contract_version", "demographic_fields",
    ]
    aggregated = data.groupby(unit_columns, as_index=False, dropna=False).agg(
        y_true_mmhg=("y_true_mmhg", "mean"),
        y_pred_mmhg=("y_pred_mmhg", "mean"),
        n_segments=("segment_id", "nunique"),
        n_seeds=("seed", "nunique"),
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
    interval_metrics = (
        "mae",
        "rmse",
        "bias",
        "error_std",
        "r2",
        "pearson",
        "calibration_slope",
        "calibration_intercept",
    )
    if replicates <= 0:
        return {
            f"{metric}_ci_{side}": np.nan
            for metric in interval_metrics
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


def summarize_benchmark(
    predictions: pd.DataFrame,
    bootstrap_replicates: int = 2000,
    bootstrap_confidence: float = 0.95,
    seed: int = 20260715,
    additional_conditions: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = aggregate_prediction_units(predictions, additional_conditions)
    records = []
    grouping = [
        "dataset", "model", "condition", "target", "preprocessing_policy",
        "padding_policy", "padding_required", "pretraining_overlap",
        "source_fidelity", "contract_version", "demographic_fields",
    ]
    for index, (keys, group) in enumerate(units.groupby(grouping, dropna=False)):
        records.append(
            {
                **dict(zip(grouping, keys)),
                **participant_balanced_metrics(group),
                **_bootstrap_intervals(
                    group, bootstrap_replicates, bootstrap_confidence, seed + index
                ),
            }
        )
    return units, pd.DataFrame(records).sort_values(grouping).reset_index(drop=True)


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
        elif condition == "ppg_features_demographics":
            comparisons.append(("demographics_added_to_ppg_features", "ppg_features"))
        for effect, reference_condition in comparisons:
            reference_demographics = "none" if reference_condition in {
                "frozen", "ppg_features"
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
    raise ValueError(f"Unsupported active condition: {condition}")


def heatmap_method_display_label(model: str, condition: str) -> str:
    """Collapse cohort-specific demographic vectors into one heatmap row."""

    return method_display_label(model, condition, "age/sex/BMI")


def _has_age_sex_only_demographics(data: pd.DataFrame) -> bool:
    demographic_conditions = {
        "demographics",
        "ppg_features_demographics",
        "frozen_demographics",
    }
    return bool(
        (
            data["condition"].isin(demographic_conditions)
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


def _paired_mae_plot(
    summary: pd.DataFrame,
    dataset: str,
    output_root: Path,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure, axis = plt.subplots(figsize=(14, 7.5))
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
            f"{demo_only['mae']:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
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
                    f"{row['mae']:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

    axis.set_xticks(
        centers,
        (
            "Demographics\nonly",
            "PPG\nfeatures",
            "Frozen\nPaPaGei-P",
            "Frozen\nPaPaGei-S",
            "Frozen\nPulse-PPG",
            "Frozen\nAnyPPG",
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
    bottom = 0.05 if _has_age_sex_only_demographics(summary) else 0
    if bottom:
        _add_demographic_figure_note(figure)
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
            rectangle = (
                _add_demographic_figure_note(figure)
                if _has_age_sex_only_demographics(group)
                else (0, 0, 1, 0.96)
            )
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
            in {
                "demographics",
                "ppg_features_demographics",
                "frozen_demographics",
            }
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
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
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
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    path = output_root / "mae_rank_heatmap.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    effects = incremental_effects(metrics)
    effect_labels = {
        "handcrafted_ppg": "PPG features",
        "papagei_p": "Frozen PaPaGei-P",
        "papagei_s": "Frozen PaPaGei-S",
        "pulseppg": "Frozen Pulse-PPG",
        "anyppg": "Frozen AnyPPG",
    }
    effects["label"] = effects["model"].map(effect_labels)
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
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    path = output_root / "demographic_mae_delta_heatmap.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    return paths
