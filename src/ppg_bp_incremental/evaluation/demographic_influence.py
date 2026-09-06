"""Read-only demographic influence analysis for the reported OOF benchmark.

The functions in this module never refit a foundation encoder or a BP model.
They operate only on the reported ``source-faithful-v2`` out-of-fold
predictions and retain participant-balanced scoring throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ppg_bp_incremental.models.contracts import CONTRACT_VERSION


MODEL_CONDITIONS = {
    "handcrafted_ppg": ("ppg_features", "ppg_features_demographics"),
    "papagei_p": ("frozen", "frozen_demographics"),
    "papagei_s": ("frozen", "frozen_demographics"),
    "pulseppg": ("frozen", "frozen_demographics"),
    "anyppg": ("frozen", "frozen_demographics"),
}

MODEL_LABELS = {
    "handcrafted_ppg": "PPG features",
    "papagei_p": "PaPaGei-P",
    "papagei_s": "PaPaGei-S",
    "pulseppg": "Pulse-PPG",
    "anyppg": "AnyPPG",
    "demographics": "Demographics only",
}

PAIR_KEYS = ["dataset", "subject_id", "measurement_id", "fold", "target"]
EXPECTED_OUTER_FOLDS = frozenset(range(5))


@dataclass(frozen=True)
class BootstrapSpec:
    replicates: int = 10_000
    confidence: float = 0.95
    seed: int = 20260830

    def checked(self) -> "BootstrapSpec":
        if self.replicates < 1:
            raise ValueError("bootstrap replicates must be positive")
        if not 0 < self.confidence < 1:
            raise ValueError("bootstrap confidence must lie between zero and one")
        return self


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    suffix = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
    return int((int(base_seed) + suffix) % (2**32 - 1))


def _checked_units(units: pd.DataFrame) -> pd.DataFrame:
    required = {
        *PAIR_KEYS,
        "model",
        "condition",
        "y_true_mmhg",
        "y_pred_mmhg",
        "demographic_fields",
        "contract_version",
    }
    missing = sorted(required.difference(units.columns))
    if missing:
        raise ValueError(f"aggregated prediction table lacks columns: {missing}")
    checked = units.copy()
    versions = set(checked["contract_version"].astype(str))
    if versions != {CONTRACT_VERSION}:
        raise ValueError(
            f"demographic analysis requires {CONTRACT_VERSION}, observed {versions}"
        )
    if checked.duplicated(PAIR_KEYS + ["model", "condition"]).any():
        raise ValueError("aggregated prediction units contain duplicate method rows")
    numeric = checked[["y_true_mmhg", "y_pred_mmhg"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("aggregated prediction units contain non-finite BP values")

    fold_values = pd.to_numeric(checked["fold"], errors="coerce")
    if fold_values.isna().any() or not np.equal(fold_values, np.floor(fold_values)).all():
        raise ValueError("outer folds must be finite integers")
    checked["fold"] = fold_values.astype(int)
    for dataset, group in checked.groupby("dataset", sort=False):
        observed = frozenset(group["fold"].unique())
        if observed != EXPECTED_OUTER_FOLDS:
            raise ValueError(
                f"{dataset} does not contain the expected outer folds: {observed}"
            )

    subject_folds = checked.groupby(["dataset", "subject_id"])["fold"].nunique()
    if (subject_folds != 1).any():
        raise ValueError("a participant appears in more than one outer fold")

    truth_range = checked.groupby(
        ["dataset", "subject_id", "measurement_id", "target"]
    )["y_true_mmhg"].agg(lambda values: float(values.max() - values.min()))
    if (truth_range > 1e-10).any():
        raise ValueError("reference BP differs across methods for a scoring key")

    field_counts = checked.groupby(
        ["dataset", "model", "condition"]
    )["demographic_fields"].nunique(dropna=False)
    if (field_counts != 1).any():
        raise ValueError("demographic field labels vary within a method")
    has_demographics = checked["condition"].astype(str).str.contains("demographics")
    fields = checked["demographic_fields"].fillna("none").astype(str)
    if fields[has_demographics].eq("none").any():
        raise ValueError("a demographic condition is labeled with no demographics")
    if fields[~has_demographics].ne("none").any():
        raise ValueError("a no-demographic condition carries demographic fields")
    dataset_fields = checked.loc[has_demographics].groupby("dataset")[
        "demographic_fields"
    ].nunique(dropna=False)
    if (dataset_fields != 1).any():
        raise ValueError("demographic field labels disagree within a cohort")
    return checked


def _method_rows(
    units: pd.DataFrame,
    *,
    dataset: str,
    target: str,
    model: str,
    condition: str,
) -> pd.DataFrame:
    selected = units[
        units["dataset"].eq(dataset)
        & units["target"].eq(target)
        & units["model"].eq(model)
        & units["condition"].eq(condition)
    ]
    if selected.empty:
        raise ValueError(f"missing {dataset}/{target}/{model}/{condition} rows")
    return selected


def paired_subject_mae_contrast(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    bootstrap: BootstrapSpec,
    seed_parts: tuple[object, ...] = (),
) -> dict[str, float | int | str]:
    """Candidate-minus-reference MAE with a fixed-OOF percentile bootstrap."""
    bootstrap.checked()
    left = candidate[PAIR_KEYS + ["y_true_mmhg", "y_pred_mmhg"]]
    right = reference[PAIR_KEYS + ["y_true_mmhg", "y_pred_mmhg"]]
    paired = left.merge(
        right,
        on=PAIR_KEYS,
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError("candidate and reference do not have identical scored rows")
    if not np.allclose(
        paired["y_true_mmhg_candidate"],
        paired["y_true_mmhg_reference"],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("candidate and reference truths differ")
    truth = paired["y_true_mmhg_candidate"].to_numpy(float)
    paired["absolute_error_delta"] = np.abs(
        paired["y_pred_mmhg_candidate"].to_numpy(float) - truth
    ) - np.abs(paired["y_pred_mmhg_reference"].to_numpy(float) - truth)
    subject_delta = (
        paired.groupby("subject_id", sort=True)["absolute_error_delta"]
        .mean()
        .to_numpy(float)
    )
    resolved_seed = _stable_seed(bootstrap.seed, *seed_parts)
    rng = np.random.default_rng(resolved_seed)
    sampled = rng.integers(
        0,
        len(subject_delta),
        size=(bootstrap.replicates, len(subject_delta)),
    )
    draws = subject_delta[sampled].mean(axis=1)
    alpha = (1 - bootstrap.confidence) / 2
    low, high = np.quantile(draws, [alpha, 1 - alpha])
    return {
        "n_subjects": int(len(subject_delta)),
        "n_measurements": int(len(paired)),
        "delta_mae_candidate_minus_reference": float(subject_delta.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_probability_candidate_better": float(np.mean(draws < 0)),
        "bootstrap_method": "participant_paired_percentile_fixed_oof",
        "bootstrap_replicates": int(bootstrap.replicates),
        "bootstrap_confidence": float(bootstrap.confidence),
        "bootstrap_seed": int(resolved_seed),
    }


def demographic_paired_contrasts(
    units: pd.DataFrame,
    *,
    bootstrap: BootstrapSpec = BootstrapSpec(),
) -> pd.DataFrame:
    """Quantify demographic lift and PPG lift beyond demographics."""
    data = _checked_units(units)
    records: list[dict[str, object]] = []
    for dataset in sorted(data["dataset"].astype(str).unique()):
        for target in ("sbp", "dbp"):
            demographic_only = _method_rows(
                data,
                dataset=dataset,
                target=target,
                model="demographics",
                condition="demographics",
            )
            for model, (waveform_condition, combined_condition) in MODEL_CONDITIONS.items():
                waveform = _method_rows(
                    data,
                    dataset=dataset,
                    target=target,
                    model=model,
                    condition=waveform_condition,
                )
                combined = _method_rows(
                    data,
                    dataset=dataset,
                    target=target,
                    model=model,
                    condition=combined_condition,
                )
                comparisons = (
                    (
                        "add_demographics_to_waveform",
                        combined,
                        waveform,
                        combined_condition,
                        waveform_condition,
                    ),
                    (
                        "waveform_plus_demographics_vs_demographics_only",
                        combined,
                        demographic_only,
                        combined_condition,
                        "demographics",
                    ),
                    (
                        "waveform_only_vs_demographics_only",
                        waveform,
                        demographic_only,
                        waveform_condition,
                        "demographics",
                    ),
                )
                for contrast, candidate, reference, candidate_condition, reference_condition in comparisons:
                    result = paired_subject_mae_contrast(
                        candidate,
                        reference,
                        bootstrap=bootstrap,
                        seed_parts=(dataset, target, model, contrast),
                    )
                    records.append(
                        {
                            "dataset": dataset,
                            "target": target,
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "contrast": contrast,
                            "candidate_condition": candidate_condition,
                            "reference_condition": reference_condition,
                            "demographic_fields": str(combined["demographic_fields"].iloc[0]),
                            **result,
                        }
                    )
    return pd.DataFrame.from_records(records).sort_values(
        ["contrast", "dataset", "target", "model"]
    ).reset_index(drop=True)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = np.asarray(values, dtype=float)[order]
    ordered_weights = np.asarray(weights, dtype=float)[order]
    threshold = 0.5 * ordered_weights.sum()
    index = int(np.searchsorted(np.cumsum(ordered_weights), threshold, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def demographic_vs_outer_constant(
    units: pd.DataFrame,
    *,
    bootstrap: BootstrapSpec = BootstrapSpec(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare demographics-only OOF predictions with fold-fit constants."""
    data = _checked_units(units)
    demographic = data[
        data["model"].eq("demographics") & data["condition"].eq("demographics")
    ].copy()
    prediction_parts: list[pd.DataFrame] = []
    summary: list[dict[str, object]] = []
    for (dataset, target), group in demographic.groupby(["dataset", "target"], sort=True):
        for fold, held in group.groupby("fold", sort=True):
            fit = group[~group["fold"].eq(fold)]
            counts = fit.groupby("subject_id")["measurement_id"].transform("count").to_numpy(float)
            weights = 1.0 / counts
            values = fit["y_true_mmhg"].to_numpy(float)
            constants = {
                "outer_fit_participant_weighted_median": _weighted_median(values, weights),
                "outer_fit_participant_weighted_mean": float(np.average(values, weights=weights)),
            }
            for method, value in constants.items():
                part = held[PAIR_KEYS + ["y_true_mmhg", "y_pred_mmhg"]].copy()
                part["constant_method"] = method
                part["constant_prediction_mmhg"] = value
                prediction_parts.append(part)
        all_predictions = pd.concat(
            [
                part
                for part in prediction_parts
                if str(part["dataset"].iloc[0]) == str(dataset)
                and str(part["target"].iloc[0]) == str(target)
            ],
            ignore_index=True,
        )
        for method, constant_rows in all_predictions.groupby("constant_method", sort=True):
            candidate = constant_rows[PAIR_KEYS + ["y_true_mmhg", "y_pred_mmhg"]]
            reference = constant_rows[
                PAIR_KEYS + ["y_true_mmhg", "constant_prediction_mmhg"]
            ].rename(columns={"constant_prediction_mmhg": "y_pred_mmhg"})
            result = paired_subject_mae_contrast(
                candidate,
                reference,
                bootstrap=bootstrap,
                seed_parts=(dataset, target, method, "demographics_vs_constant"),
            )
            candidate_subject = constant_rows.assign(
                absolute_error=lambda frame: np.abs(
                    frame["y_pred_mmhg"] - frame["y_true_mmhg"]
                )
            ).groupby("subject_id")["absolute_error"].mean()
            constant_subject = constant_rows.assign(
                absolute_error=lambda frame: np.abs(
                    frame["constant_prediction_mmhg"] - frame["y_true_mmhg"]
                )
            ).groupby("subject_id")["absolute_error"].mean()
            summary.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "constant_method": method,
                    "demographics_only_mae": float(candidate_subject.mean()),
                    "constant_mae": float(constant_subject.mean()),
                    **result,
                }
            )
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["dataset", "target", "constant_method", "fold", "subject_id", "measurement_id"]
    )
    return pd.DataFrame(summary).sort_values(
        ["constant_method", "dataset", "target"]
    ).reset_index(drop=True), predictions.reset_index(drop=True)


def between_within_metrics(units: pd.DataFrame) -> pd.DataFrame:
    """Separate subject-mean BP accuracy from within-subject deviation accuracy."""
    data = _checked_units(units)
    records: list[dict[str, object]] = []
    grouping = ["dataset", "target", "model", "condition", "demographic_fields"]
    for keys, group in data.groupby(grouping, sort=True):
        means = group.groupby("subject_id", sort=True).agg(
            true_subject_mean=("y_true_mmhg", "mean"),
            predicted_subject_mean=("y_pred_mmhg", "mean"),
            n_measurements=("measurement_id", "size"),
        )
        scored = group.join(
            means[["true_subject_mean", "predicted_subject_mean"]], on="subject_id"
        )
        scored["absolute_error"] = np.abs(
            scored["y_pred_mmhg"] - scored["y_true_mmhg"]
        )
        scored["true_deviation"] = (
            scored["y_true_mmhg"] - scored["true_subject_mean"]
        )
        scored["predicted_deviation"] = (
            scored["y_pred_mmhg"] - scored["predicted_subject_mean"]
        )
        scored["within_absolute_error"] = np.abs(
            scored["predicted_deviation"] - scored["true_deviation"]
        )
        subject = scored.groupby("subject_id", sort=True).agg(
            overall_mae=("absolute_error", "mean"),
            within_deviation_mae=("within_absolute_error", "mean"),
            true_within_rms=("true_deviation", lambda values: np.sqrt(np.mean(values**2))),
            predicted_within_rms=(
                "predicted_deviation", lambda values: np.sqrt(np.mean(values**2))
            ),
        )
        between = np.abs(
            means["predicted_subject_mean"] - means["true_subject_mean"]
        )
        true_rms = float(subject["true_within_rms"].mean())
        predicted_rms = float(subject["predicted_within_rms"].mean())
        records.append(
            {
                **dict(zip(grouping, keys)),
                "n_subjects": int(len(subject)),
                "n_measurements": int(len(group)),
                "subjects_with_repeated_measurements": int(
                    (means["n_measurements"] > 1).sum()
                ),
                "overall_participant_macro_mae": float(subject["overall_mae"].mean()),
                "between_subject_mean_mae": float(between.mean()),
                "within_subject_deviation_mae": float(
                    subject["within_deviation_mae"].mean()
                ),
                "mean_true_within_subject_rms": true_rms,
                "mean_predicted_within_subject_rms": predicted_rms,
                "predicted_to_true_within_subject_rms_ratio": (
                    predicted_rms / true_rms if true_rms > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(grouping).reset_index(drop=True)


def demographic_associations(
    units: pd.DataFrame,
    raw_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive subject-level demographic distributions and BP associations."""
    data = _checked_units(units)
    required = {"dataset", "subject_id", "age", "sex", "bmi", "model", "condition"}
    missing = sorted(required.difference(raw_predictions.columns))
    if missing:
        raise ValueError(f"raw prediction table lacks demographic columns: {missing}")
    if "contract_version" not in raw_predictions.columns:
        raise ValueError("raw prediction table lacks contract_version")
    versions = set(raw_predictions["contract_version"].astype(str))
    if versions != {CONTRACT_VERSION}:
        raise ValueError(
            f"raw demographic analysis requires {CONTRACT_VERSION}, observed {versions}"
        )
    raw_demo = raw_predictions[
        raw_predictions["model"].eq("demographics")
        & raw_predictions["condition"].eq("demographics")
    ].copy()
    canonical_records: list[dict[str, object]] = []
    for (dataset, subject_id), group in raw_demo.groupby(
        ["dataset", "subject_id"], sort=True
    ):
        age_values = np.sort(
            pd.to_numeric(group["age"], errors="coerce").dropna().unique()
        )
        if len(age_values) > 1 and np.ptp(age_values) > 1e-10:
            raise ValueError(f"age varies within {dataset}/{subject_id}")
        sex_values = sorted(
            {
                str(value).strip().lower()
                for value in group["sex"].dropna()
                if str(value).strip()
            }
        )
        if len(sex_values) > 1:
            raise ValueError(f"sex varies within {dataset}/{subject_id}")
        bmi_values = np.sort(
            pd.to_numeric(group["bmi"], errors="coerce").dropna().unique()
        )
        canonical_records.append(
            {
                "dataset": dataset,
                "subject_id": subject_id,
                # Age and sex must be participant-consistent. BMI may vary
                # across reported source records, so the descriptive audit
                # declares the median of unique observed values as canonical.
                "age": float(np.median(age_values)) if len(age_values) else np.nan,
                "sex": sex_values[0] if sex_values else "unknown",
                "bmi": float(np.median(bmi_values)) if len(bmi_values) else np.nan,
                "bmi_unique_observed": int(len(bmi_values)),
            }
        )
    demographics = pd.DataFrame.from_records(canonical_records)
    truth = data[
        data["model"].eq("demographics") & data["condition"].eq("demographics")
    ].groupby(["dataset", "subject_id", "target"], as_index=False).agg(
        subject_mean_bp=("y_true_mmhg", "mean")
    )
    subject = truth.merge(
        demographics, on=["dataset", "subject_id"], how="left", validate="many_to_one"
    )
    records: list[dict[str, object]] = []
    for (dataset, target), group in subject.groupby(["dataset", "target"], sort=True):
        for field in ("age", "bmi"):
            observed = group[[field, "subject_mean_bp"]].dropna()
            correlation, p_value = (np.nan, np.nan)
            if len(observed) >= 3 and observed[field].nunique() > 1:
                correlation, p_value = spearmanr(
                    observed[field], observed["subject_mean_bp"]
                )
            records.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "demographic_quantity": field,
                    "n_subjects": int(len(observed)),
                    "effect": float(correlation),
                    "effect_definition": "spearman_rho_with_subject_mean_bp",
                    "unadjusted_p_value": float(p_value),
                }
            )
        normalized_sex = group["sex"].astype(str).str.lower()
        male = group.loc[normalized_sex.eq("male"), "subject_mean_bp"]
        female = group.loc[normalized_sex.eq("female"), "subject_mean_bp"]
        records.append(
            {
                "dataset": dataset,
                "target": target,
                "demographic_quantity": "sex",
                "n_subjects": int(len(male) + len(female)),
                "effect": float(male.mean() - female.mean()),
                "effect_definition": "male_minus_female_mean_subject_bp_mmhg",
                "unadjusted_p_value": np.nan,
            }
        )
    distributions = demographics.groupby("dataset", sort=True).agg(
        n_subjects=("subject_id", "nunique"),
        age_mean=("age", "mean"),
        age_min=("age", "min"),
        age_median=("age", "median"),
        age_max=("age", "max"),
        bmi_observed=("bmi", "count"),
        bmi_subjects_with_multiple_observed_values=(
            "bmi_unique_observed", lambda values: int((values > 1).sum())
        ),
        bmi_mean=("bmi", "mean"),
        bmi_min=("bmi", "min"),
        bmi_median=("bmi", "median"),
        bmi_max=("bmi", "max"),
        female=("sex", lambda values: int(values.astype(str).str.lower().eq("female").sum())),
        male=("sex", lambda values: int(values.astype(str).str.lower().eq("male").sum())),
    ).reset_index()
    return (
        pd.DataFrame.from_records(records).sort_values(
            ["dataset", "target", "demographic_quantity"]
        ).reset_index(drop=True),
        distributions,
    )


def complementarity_table(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Create positive-is-better demographic and waveform contribution axes."""
    key = ["dataset", "target", "model", "model_label", "demographic_fields"]
    values = contrasts.pivot_table(
        index=key,
        columns="contrast",
        values="delta_mae_candidate_minus_reference",
        aggfunc="first",
    ).reset_index()
    values["demographic_lift_mmhg"] = -values["add_demographics_to_waveform"]
    values["waveform_lift_beyond_demographics_mmhg"] = -values[
        "waveform_plus_demographics_vs_demographics_only"
    ]
    values["combined_better_than_both_components"] = (
        values["demographic_lift_mmhg"] > 0
    ) & (values["waveform_lift_beyond_demographics_mmhg"] > 0)
    return values[
        key
        + [
            "demographic_lift_mmhg",
            "waveform_lift_beyond_demographics_mmhg",
            "combined_better_than_both_components",
        ]
    ].sort_values(["dataset", "target", "model"]).reset_index(drop=True)


def plot_complementarity(table: pd.DataFrame, output_path: str | Path) -> Path:
    datasets = ["BUT PPG", "PPG-BP", "PulseDB-Vital", "PulseDB-MIMIC"]
    targets = ["sbp", "dbp"]
    colors = {
        model: plt.get_cmap("tab10")(index)
        for index, model in enumerate(MODEL_CONDITIONS)
    }
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for row, target in enumerate(targets):
        for column, dataset in enumerate(datasets):
            axis = axes[row, column]
            group = table[
                table["dataset"].eq(dataset) & table["target"].eq(target)
            ]
            axis.axhline(0, color="0.65", linewidth=1)
            axis.axvline(0, color="0.65", linewidth=1)
            for point in group.itertuples(index=False):
                axis.scatter(
                    point.demographic_lift_mmhg,
                    point.waveform_lift_beyond_demographics_mmhg,
                    s=55,
                    color=colors[point.model],
                    edgecolor="white",
                    linewidth=0.6,
                    label=point.model_label,
                )
            axis.set_title(f"{dataset} · {target.upper()}")
            if row == 1:
                axis.set_xlabel("Demographic lift (mmHg)")
            if column == 0:
                axis.set_ylabel("Waveform lift beyond demo (mmHg)")
            axis.grid(alpha=0.18)
    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", color=colors[model],
            label=MODEL_LABELS[model]
        )
        for model in MODEL_CONDITIONS
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    figure.suptitle(
        "Beyond demographic priors: complementary MAE contributions\n"
        "Upper-right means the combined model beats both waveform-only and demographics-only",
        fontsize=14,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_demographics_vs_constant(summary: pd.DataFrame, output_path: str | Path) -> Path:
    primary = summary[
        summary["constant_method"].eq("outer_fit_participant_weighted_median")
    ].copy()
    primary["label"] = primary["dataset"] + " · " + primary["target"].str.upper()
    primary = primary.sort_values(
        ["dataset", "target"], ascending=[False, False]
    ).reset_index(drop=True)
    y = np.arange(len(primary))
    values = primary["delta_mae_candidate_minus_reference"].to_numpy(float)
    low = primary["ci_low"].to_numpy(float)
    high = primary["ci_high"].to_numpy(float)
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    axis.axvline(0, color="black", linewidth=1)
    axis.errorbar(
        values,
        y,
        xerr=np.vstack([values - low, high - values]),
        fmt="o",
        color="#2b6cb0",
        ecolor="#7aa6d8",
        capsize=3,
    )
    axis.set_yticks(y, primary["label"])
    axis.set_xlabel("Demographics-only minus outer-fit median MAE (mmHg)")
    axis.set_title("Does the demographic prior beat a true no-waveform constant?\nNegative favors demographics")
    axis.grid(axis="x", alpha=0.2)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def save_demographic_influence_analysis(
    *,
    aggregated_predictions_csv: str | Path,
    raw_predictions_csv: str | Path,
    output_root: str | Path,
    bootstrap: BootstrapSpec = BootstrapSpec(),
) -> dict[str, Path]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(aggregated_predictions_csv)
    raw = pd.read_csv(raw_predictions_csv, low_memory=False)
    contrasts = demographic_paired_contrasts(units, bootstrap=bootstrap)
    constants, constant_predictions = demographic_vs_outer_constant(
        units, bootstrap=bootstrap
    )
    between_within = between_within_metrics(units)
    associations, distributions = demographic_associations(units, raw)
    complementarity = complementarity_table(contrasts)
    tables = {
        "paired_contrasts": contrasts,
        "demographics_vs_outer_constant": constants,
        "outer_constant_predictions": constant_predictions,
        "between_within_metrics": between_within,
        "demographic_associations": associations,
        "demographic_distributions": distributions,
        "complementarity": complementarity,
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    paths["complementarity_figure"] = plot_complementarity(
        complementarity, output / "information_source_complementarity.png"
    )
    paths["constant_figure"] = plot_demographics_vs_constant(
        constants, output / "demographics_vs_outer_constant.png"
    )
    return paths
