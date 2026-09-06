"""Reproducible cohort-level descriptive analysis for the reported benchmark."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppg_bp_incremental.models.contracts import CONTRACT_VERSION


EDA_VERSION = "data-eda-v2"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _subject_balanced_weights(data: pd.DataFrame) -> np.ndarray:
    counts = data.groupby("subject_id")["subject_id"].transform("size").to_numpy(float)
    return 1.0 / counts


def _released_label_units(data: pd.DataFrame) -> pd.DataFrame:
    """Return one row per released measurement-level BP label.

    This prevents input-window expansion from being mistaken for additional
    independent labels.
    """

    return data.sort_values("segment_id").drop_duplicates(
        ["subject_id", "measurement_id"], keep="first"
    )


def summarize_manifest(data: pd.DataFrame) -> dict:
    required = {
        "dataset", "source", "subject_id", "measurement_id", "segment_id",
        "sample_rate_hz", "n_samples", "sbp", "dbp", "age", "sex", "bmi",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"EDA manifest is missing columns: {sorted(missing)}")
    numeric = data.copy()
    for column in ("sample_rate_hz", "n_samples", "sbp", "dbp", "age", "bmi"):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    duration = numeric["n_samples"] / numeric["sample_rate_hz"]
    label_units = _released_label_units(numeric)
    label_duration = (
        pd.to_numeric(numeric["label_duration_seconds"], errors="coerce")
        if "label_duration_seconds" in numeric
        else duration
    )
    return {
        "dataset": str(numeric["dataset"].iloc[0]),
        "source": str(numeric["source"].iloc[0]),
        "subjects": int(numeric["subject_id"].nunique()),
        "measurements": int(numeric["measurement_id"].nunique()),
        "segments": len(numeric),
        "input_windows": len(numeric),
        "released_label_units": len(label_units),
        "sample_rates_hz": ",".join(
            map(str, sorted(numeric["sample_rate_hz"].dropna().unique().tolist()))
        ),
        "duration_median_seconds": float(duration.median()),
        "duration_min_seconds": float(duration.min()),
        "duration_max_seconds": float(duration.max()),
        "label_duration_median_seconds": float(label_duration.median()),
        "label_scope": (
            ",".join(sorted(numeric["label_scope"].dropna().astype(str).unique()))
            if "label_scope" in numeric
            else "segment_or_measurement_label"
        ),
        "sbp_mean": float(label_units["sbp"].mean()),
        "sbp_sd": float(label_units["sbp"].std(ddof=0)),
        "sbp_min": float(label_units["sbp"].min()),
        "sbp_max": float(label_units["sbp"].max()),
        "dbp_mean": float(label_units["dbp"].mean()),
        "dbp_sd": float(label_units["dbp"].std(ddof=0)),
        "dbp_min": float(label_units["dbp"].min()),
        "dbp_max": float(label_units["dbp"].max()),
        "age_missing_fraction": float(numeric["age"].isna().mean()),
        "sex_missing_or_unknown_fraction": float(
            numeric["sex"].fillna("unknown").astype(str).str.lower().eq("unknown").mean()
        ),
        "bmi_missing_fraction": float(numeric["bmi"].isna().mean()),
        "female_subjects": int(
            numeric.groupby("subject_id")["sex"].first().astype(str).str.lower().eq("female").sum()
        ),
        "male_subjects": int(
            numeric.groupby("subject_id")["sex"].first().astype(str).str.lower().eq("male").sum()
        ),
    }


def _plot_bp_distributions(frames: list[pd.DataFrame], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(frames)))
    for data, color in zip(frames, colors, strict=True):
        label_units = _released_label_units(data)
        weights = _subject_balanced_weights(label_units)
        name = str(data["dataset"].iloc[0])
        axes[0].hist(
            pd.to_numeric(label_units["sbp"], errors="coerce"), bins=25, weights=weights,
            histtype="step", linewidth=2, color=color, label=name, density=True,
        )
        axes[1].hist(
            pd.to_numeric(label_units["dbp"], errors="coerce"), bins=25, weights=weights,
            histtype="step", linewidth=2, color=color, label=name, density=True,
        )
    axes[0].set_title("Subject-balanced SBP distribution")
    axes[1].set_title("Subject-balanced DBP distribution")
    for axis, label in zip(axes, ("SBP (mmHg)", "DBP (mmHg)"), strict=True):
        axis.set_xlabel(label)
        axis.set_ylabel("Density")
        axis.grid(alpha=0.15)
        axis.legend(frameon=False, fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_demographic_distributions(frames: list[pd.DataFrame], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(frames)))
    for data, color in zip(frames, colors, strict=True):
        subject = data.sort_values("segment_id").groupby("subject_id", as_index=False).first()
        name = str(data["dataset"].iloc[0])
        age = pd.to_numeric(subject["age"], errors="coerce").dropna()
        bmi = pd.to_numeric(subject["bmi"], errors="coerce").dropna()
        if len(age):
            axes[0].hist(
                age, bins=20, histtype="step", linewidth=2, color=color,
                label=name, density=True,
            )
        if len(bmi):
            axes[1].hist(
                bmi, bins=20, histtype="step", linewidth=2, color=color,
                label=name, density=True,
            )
    axes[0].set_title("Subject age distribution")
    axes[1].set_title("Subject BMI distribution")
    axes[0].set_xlabel("Age (years)")
    axes[1].set_xlabel("BMI (kg/m²)")
    for axis in axes:
        axis.set_ylabel("Density")
        axis.grid(alpha=0.15)
        axis.legend(frameon=False, fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def generate_data_eda(
    manifest_csvs: Iterable[str | Path],
    output_root: str | Path,
) -> dict:
    paths = [Path(path) for path in manifest_csvs]
    if not paths:
        raise ValueError("At least one EDA manifest is required")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(path) for path in paths]
    summaries = pd.DataFrame([summarize_manifest(frame) for frame in frames])
    summaries.to_csv(output_root / "cohort_summary.csv", index=False)
    _plot_bp_distributions(frames, output_root / "bp_distributions.png")
    _plot_demographic_distributions(frames, output_root / "demographic_distributions.png")
    metadata = {
        "eda_version": EDA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "manifests": [
            {"path": str(path.resolve()), "sha256": _file_sha256(path)} for path in paths
        ],
        "weighting": (
            "one row per released measurement label; equal total plot weight per subject"
        ),
        "outputs": {
            "summary": "cohort_summary.csv",
            "bp_distributions": "bp_distributions.png",
            "demographic_distributions": "demographic_distributions.png",
        },
    }
    with (output_root / "data_eda.run.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "summary": output_root / "cohort_summary.csv",
        "bp_figure": output_root / "bp_distributions.png",
        "demographic_figure": output_root / "demographic_distributions.png",
        "metadata": output_root / "data_eda.run.json",
    }
