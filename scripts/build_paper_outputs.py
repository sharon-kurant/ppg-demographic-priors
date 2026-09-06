#!/usr/bin/env python3
"""Rebuild the reported paper figure and focused information-source table."""

from __future__ import annotations

import argparse
from pathlib import Path

from ppg_bp_incremental.evaluation.paper_outputs import save_paper_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("results/benchmark_metrics.csv"))
    parser.add_argument(
        "--paired-contrasts",
        type=Path,
        default=Path("results/paired_demographic_contrasts.csv"),
    )
    parser.add_argument(
        "--constants",
        type=Path,
        default=Path("results/demographics_vs_constant.csv"),
    )
    parser.add_argument(
        "--complementarity",
        type=Path,
        default=Path("results/information_source_complementarity.csv"),
    )
    parser.add_argument(
        "--prediction-spread",
        type=Path,
        default=Path("results/demographics_prediction_spread.csv"),
    )
    parser.add_argument(
        "--aggregated-predictions",
        type=Path,
        help="recalculate spread from local participant-level outputs",
    )
    parser.add_argument("--output-root", type=Path, default=Path("paper_outputs"))
    args = parser.parse_args()
    outputs = save_paper_outputs(
        metrics_csv=args.metrics,
        paired_contrasts_csv=args.paired_contrasts,
        constants_csv=args.constants,
        complementarity_csv=args.complementarity,
        prediction_spread_csv=(
            None if args.aggregated_predictions else args.prediction_spread
        ),
        aggregated_predictions_csv=args.aggregated_predictions,
        output_root=args.output_root,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
