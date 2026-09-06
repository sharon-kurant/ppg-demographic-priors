"""Build the paper's focused demographic-effect figure and summary table."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_OUTPUT_VERSION = "paper-demographic-summary-v1"
DATASET_ORDER = ("PPG-BP", "BUT PPG", "PulseDB-Vital", "PulseDB-MIMIC")
TARGET_ORDER = ("sbp", "dbp")
MODEL_ORDER = (
    "handcrafted_ppg",
    "papagei_p",
    "papagei_s",
    "pulseppg",
    "anyppg",
)
MODEL_LABELS = {
    "handcrafted_ppg": "PPG features",
    "papagei_p": "PaPaGei-P",
    "papagei_s": "PaPaGei-S",
    "pulseppg": "Pulse-PPG",
    "anyppg": "AnyPPG",
}


class PaperOutputInputError(ValueError):
    """Raised when an input table cannot reproduce the reported output."""


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise PaperOutputInputError(f"{name} is missing columns: {sorted(missing)}")


def _one(frame: pd.DataFrame, mask: pd.Series, description: str) -> pd.Series:
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise PaperOutputInputError(
            f"Expected exactly one row for {description}; received {len(rows)}"
        )
    return rows.iloc[0]


def prediction_spread_from_aggregated(aggregated: pd.DataFrame) -> pd.DataFrame:
    """Calculate demographics-only participant-mean prediction spread."""
    _require_columns(
        aggregated,
        {
            "dataset",
            "target",
            "model",
            "condition",
            "subject_id",
            "y_true_mmhg",
            "y_pred_mmhg",
        },
        "aggregated predictions",
    )
    demographic = aggregated[
        aggregated["model"].eq("demographics")
        & aggregated["condition"].eq("demographics")
    ].copy()
    subject = demographic.groupby(
        ["dataset", "target", "subject_id"], as_index=False
    )[["y_true_mmhg", "y_pred_mmhg"]].mean()
    records: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        for target in TARGET_ORDER:
            group = subject[
                subject["dataset"].eq(dataset) & subject["target"].eq(target)
            ]
            if group["subject_id"].nunique() < 2:
                raise PaperOutputInputError(
                    f"At least two participants are required for {dataset}/{target}"
                )
            truth_sd = float(group["y_true_mmhg"].std(ddof=1))
            prediction_sd = float(group["y_pred_mmhg"].std(ddof=1))
            if not np.isfinite([truth_sd, prediction_sd]).all() or truth_sd <= 0:
                raise PaperOutputInputError(
                    f"Invalid prediction spread for {dataset}/{target}"
                )
            records.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "n_subjects": int(group["subject_id"].nunique()),
                    "truth_sd_mmhg": truth_sd,
                    "prediction_sd_mmhg": prediction_sd,
                    "prediction_to_truth_sd_ratio": prediction_sd / truth_sd,
                    "demographic_fields": (
                        "age/sex" if dataset == "PulseDB-MIMIC" else "age/sex/BMI"
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def build_information_source_table(
    metrics: pd.DataFrame,
    constants: pd.DataFrame,
    contrasts: pd.DataFrame,
    complementarity: pd.DataFrame,
) -> pd.DataFrame:
    """Build the eight-row information-source table used in the paper."""
    _require_columns(
        metrics,
        {"dataset", "target", "model", "condition", "mae", "demographic_fields"},
        "metrics",
    )
    _require_columns(
        constants,
        {"dataset", "target", "constant_method", "constant_mae"},
        "constant summary",
    )
    _require_columns(
        contrasts,
        {"dataset", "target", "model", "contrast", "delta_mae_candidate_minus_reference"},
        "paired contrasts",
    )
    _require_columns(
        complementarity,
        {"dataset", "target", "model", "waveform_lift_beyond_demographics_mmhg"},
        "complementarity",
    )
    records: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        for target in TARGET_ORDER:
            constant = _one(
                constants,
                constants["dataset"].eq(dataset)
                & constants["target"].eq(target)
                & constants["constant_method"].eq(
                    "outer_fit_participant_weighted_median"
                ),
                f"training-fold median, {dataset}/{target}",
            )
            demo = _one(
                metrics,
                metrics["dataset"].eq(dataset)
                & metrics["target"].eq(target)
                & metrics["model"].eq("demographics")
                & metrics["condition"].eq("demographics"),
                f"demographics only, {dataset}/{target}",
            )
            effect_rows = contrasts[
                contrasts["dataset"].eq(dataset)
                & contrasts["target"].eq(target)
                & contrasts["contrast"].eq("add_demographics_to_waveform")
            ]
            complement_rows = complementarity[
                complementarity["dataset"].eq(dataset)
                & complementarity["target"].eq(target)
            ]
            if set(effect_rows["model"]) != set(MODEL_ORDER) or len(effect_rows) != 5:
                raise PaperOutputInputError(
                    f"Incomplete demographic-effect matrix for {dataset}/{target}"
                )
            if set(complement_rows["model"]) != set(MODEL_ORDER) or len(complement_rows) != 5:
                raise PaperOutputInputError(
                    f"Incomplete complementarity matrix for {dataset}/{target}"
                )
            records.append(
                {
                    "dataset": dataset,
                    "target": target.upper(),
                    "demographic_fields": str(demo["demographic_fields"]),
                    "training_fold_median_mae_mmhg": float(constant["constant_mae"]),
                    "demographics_only_mae_mmhg": float(demo["mae"]),
                    "demographics_improves_waveform_count": int(
                        (effect_rows["delta_mae_candidate_minus_reference"] < 0).sum()
                    ),
                    "waveform_comparisons": 5,
                    "waveform_improves_demographics_count": int(
                        (
                            complement_rows[
                                "waveform_lift_beyond_demographics_mmhg"
                            ]
                            > 0
                        ).sum()
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _overlap_marker(metrics: pd.DataFrame, dataset: str, model: str) -> str:
    if model == "handcrafted_ppg" or not dataset.startswith("PulseDB"):
        return ""
    rows = metrics[
        metrics["dataset"].eq(dataset) & metrics["model"].eq(model)
    ]
    if "pretraining_overlap" not in rows:
        return ""
    values = rows["pretraining_overlap"].dropna().astype(str).str.lower()
    return "‡" if (~values.isin({"none", "not_applicable", ""})).any() else ""


def plot_demographic_effect_and_spread(
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
    spread: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Render the paper's two-panel demographic-effect figure."""
    _require_columns(
        contrasts,
        {
            "dataset",
            "target",
            "model",
            "contrast",
            "delta_mae_candidate_minus_reference",
            "ci_low",
            "ci_high",
        },
        "paired contrasts",
    )
    _require_columns(
        spread,
        {"dataset", "target", "prediction_to_truth_sd_ratio"},
        "prediction spread",
    )
    endpoints = [(dataset, target) for dataset in DATASET_ORDER for target in TARGET_ORDER]
    endpoint_labels = [
        "PPG-BP\nSBP",
        "PPG-BP\nDBP",
        "BUT PPG\nSBP",
        "BUT PPG\nDBP",
        "Vital\nSBP",
        "Vital\nDBP",
        "MIMIC\nSBP†",
        "MIMIC\nDBP†",
    ]
    effects = contrasts[contrasts["contrast"].eq("add_demographics_to_waveform")]
    matrix = np.full((len(MODEL_ORDER), len(endpoints)), np.nan)
    highs = np.full_like(matrix, np.nan)
    for row_index, model in enumerate(MODEL_ORDER):
        for column_index, (dataset, target) in enumerate(endpoints):
            row = _one(
                effects,
                effects["model"].eq(model)
                & effects["dataset"].eq(dataset)
                & effects["target"].eq(target),
                f"demographic effect, {model}/{dataset}/{target}",
            )
            matrix[row_index, column_index] = float(
                row["delta_mae_candidate_minus_reference"]
            )
            highs[row_index, column_index] = float(row["ci_high"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
        }
    )
    figure = plt.figure(figsize=(6.62, 2.55), constrained_layout=False)
    grid = figure.add_gridspec(1, 2, width_ratios=[2.12, 1.0], wspace=0.38)
    axis = figure.add_subplot(grid[0, 0])
    image = axis.imshow(
        matrix,
        cmap=plt.get_cmap("BrBG_r").copy(),
        vmin=-4.0,
        vmax=4.0,
        aspect="auto",
    )
    axis.set_xticks(np.arange(len(endpoint_labels)), endpoint_labels)
    axis.set_yticks(
        np.arange(len(MODEL_ORDER)),
        [MODEL_LABELS[model] for model in MODEL_ORDER],
    )
    axis.tick_params(length=0, pad=2)
    axis.set_title(
        "A  MAE change after adding demographics",
        loc="left",
        fontweight="bold",
        pad=5,
    )
    for row_index, model in enumerate(MODEL_ORDER):
        for column_index, (dataset, _target) in enumerate(endpoints):
            value = matrix[row_index, column_index]
            marker = _overlap_marker(metrics, dataset, model)
            color = "white" if value < -2.5 or value > 3 else "black"
            axis.text(
                column_index,
                row_index,
                f"{value:+.2f}{marker}",
                ha="center",
                va="center",
                fontsize=6.2,
                color=color,
            )
            if highs[row_index, column_index] < 0:
                axis.add_patch(
                    plt.Rectangle(
                        (column_index - 0.46, row_index - 0.43),
                        0.92,
                        0.86,
                        fill=False,
                        edgecolor="#222222",
                        linewidth=0.8,
                    )
                )
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
    colorbar = figure.colorbar(
        image,
        ax=axis,
        orientation="horizontal",
        fraction=0.075,
        pad=0.22,
        aspect=24,
    )
    colorbar.set_label("ΔMAE (mmHg), negative favors demographics", labelpad=1)
    colorbar.ax.tick_params(labelsize=6.2, length=2)

    spread_values: list[float] = []
    for dataset, target in endpoints:
        row = _one(
            spread,
            spread["dataset"].eq(dataset) & spread["target"].eq(target),
            f"prediction spread, {dataset}/{target}",
        )
        spread_values.append(float(row["prediction_to_truth_sd_ratio"]))
    second = figure.add_subplot(grid[0, 1])
    y = np.arange(len(endpoint_labels))
    colors = ["#1f77b4" if target == "sbp" else "#d95f02" for _, target in endpoints]
    second.hlines(y, 0, spread_values, color="#d0d0d0", linewidth=1.1, zorder=1)
    second.scatter(spread_values, y, c=colors, s=20, zorder=2)
    for y_value, value in zip(y, spread_values, strict=True):
        second.text(min(value + 0.035, 0.94), y_value, f"{value:.2f}", va="center", fontsize=6.5)
    second.axvline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    second.set_xlim(0, 1.08)
    second.set_ylim(len(endpoint_labels) - 0.5, -0.5)
    second.set_yticks(y, endpoint_labels)
    second.set_xticks([0, 0.5, 1.0])
    second.set_xlabel("Prediction SD / reference SD")
    second.set_title("B  Demographics-only spread", loc="left", fontweight="bold", pad=5)
    second.grid(axis="x", color="#eeeeee", linewidth=0.6)
    second.spines[["top", "right", "left"]].set_visible(False)
    second.tick_params(axis="y", length=0, pad=2)

    figure.subplots_adjust(left=0.115, right=0.99, top=0.88, bottom=0.24)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    display["Endpoint"] = display["dataset"] + " " + display["target"]
    display["Training-fold median"] = display[
        "training_fold_median_mae_mmhg"
    ].map(lambda value: f"{value:.2f}")
    display["Demo only"] = display["demographics_only_mae_mmhg"].map(
        lambda value: f"{value:.2f}"
    )
    display["Demo improves waveform"] = display.apply(
        lambda row: f"{row.demographics_improves_waveform_count}/{row.waveform_comparisons}",
        axis=1,
    )
    display["Waveform improves demo"] = display.apply(
        lambda row: f"{row.waveform_improves_demographics_count}/{row.waveform_comparisons}",
        axis=1,
    )
    columns = [
        "Endpoint",
        "Training-fold median",
        "Demo only",
        "Demo improves waveform",
        "Waveform improves demo",
    ]
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in display[columns].iterrows()
    ]
    return "\n".join([header, rule, *rows]) + "\n"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_paper_outputs(
    *,
    metrics_csv: str | Path,
    paired_contrasts_csv: str | Path,
    constants_csv: str | Path,
    complementarity_csv: str | Path,
    output_root: str | Path,
    prediction_spread_csv: str | Path | None = None,
    aggregated_predictions_csv: str | Path | None = None,
) -> dict[str, Path]:
    """Validate source tables and write the exact paper figure/table outputs."""
    input_paths = {
        "metrics": Path(metrics_csv),
        "paired_contrasts": Path(paired_contrasts_csv),
        "constants": Path(constants_csv),
        "complementarity": Path(complementarity_csv),
    }
    metrics = pd.read_csv(input_paths["metrics"])
    contrasts = pd.read_csv(input_paths["paired_contrasts"])
    constants = pd.read_csv(input_paths["constants"])
    complementarity = pd.read_csv(input_paths["complementarity"])
    if aggregated_predictions_csv is not None:
        aggregated_path = Path(aggregated_predictions_csv)
        spread = prediction_spread_from_aggregated(pd.read_csv(aggregated_path))
        input_paths["aggregated_predictions"] = aggregated_path
    elif prediction_spread_csv is not None:
        spread_path = Path(prediction_spread_csv)
        spread = pd.read_csv(spread_path)
        input_paths["prediction_spread"] = spread_path
    else:
        raise PaperOutputInputError(
            "Provide prediction_spread_csv or aggregated_predictions_csv"
        )

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    table = build_information_source_table(
        metrics, constants, contrasts, complementarity
    )
    table_csv = output / "table_3_information_sources.csv"
    table_md = output / "table_3_information_sources.md"
    figure = output / "figure_1_demographic_effect_and_spread.png"
    table.to_csv(table_csv, index=False)
    table_md.write_text(_markdown_table(table), encoding="utf-8")
    plot_demographic_effect_and_spread(metrics, contrasts, spread, figure)
    spread_output = output / "demographics_prediction_spread.csv"
    spread.to_csv(spread_output, index=False)
    outputs = {
        "figure": figure,
        "table_csv": table_csv,
        "table_markdown": table_md,
        "prediction_spread": spread_output,
    }
    manifest = {
        "paper_output_version": PAPER_OUTPUT_VERSION,
        "inputs": {name: _file_sha256(path) for name, path in input_paths.items()},
        "outputs": {name: _file_sha256(path) for name, path in outputs.items()},
    }
    manifest_path = output / "paper_outputs.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["manifest"] = manifest_path
    return outputs
