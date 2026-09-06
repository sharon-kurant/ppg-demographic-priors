#!/usr/bin/env python3
"""Build the reported benchmark's demographic-influence artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from ppg_bp_incremental.evaluation.demographic_influence import (
    BootstrapSpec,
    save_demographic_influence_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregated-predictions",
        type=Path,
        default=Path(
            "artifacts/reproduction/source-faithful-v2/benchmark/"
            "aggregated_predictions.csv"
        ),
    )
    parser.add_argument(
        "--raw-predictions",
        type=Path,
        default=Path(
            "artifacts/reproduction/source-faithful-v2/benchmark/predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/reproduction/source-faithful-v2/demographic_influence"
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    paths = save_demographic_influence_analysis(
        aggregated_predictions_csv=args.aggregated_predictions,
        raw_predictions_csv=args.raw_predictions,
        output_root=args.output_root,
        bootstrap=BootstrapSpec(
            replicates=args.bootstrap_replicates,
            confidence=args.bootstrap_confidence,
            seed=args.seed,
        ),
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
