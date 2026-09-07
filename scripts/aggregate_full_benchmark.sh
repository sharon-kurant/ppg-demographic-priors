#!/usr/bin/env bash

# Aggregate the frozen/Ridge and end-to-end fine-tuning prediction artifacts.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/aggregate_full_benchmark.sh [--artifact-root PATH]

The artifact root must contain `predictions/` from reproduce_benchmark.sh and
`finetuning/` from reproduce_finetuning.sh. The script rebuilds the full
metrics, figures, demographic contrasts, and compact paper outputs.
EOF
}

fail() {
  printf 'aggregate_full_benchmark.sh: %s\n' "$*" >&2
  exit 1
}

CALLER_PWD=$PWD
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
ARTIFACT_ROOT="$REPO_ROOT/artifacts/reproduction/source-faithful-v3"

while (($#)); do
  case "$1" in
    --artifact-root)
      (($# >= 2)) || fail "$1 requires a path"
      ARTIFACT_ROOT=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1 (use --help)"
      ;;
  esac
done

case "$ARTIFACT_ROOT" in
  /*) ;;
  *) ARTIFACT_ROOT="$CALLER_PWD/$ARTIFACT_ROOT" ;;
esac

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "$REPO_ROOT/src/ppg_bp_incremental/release_cli.py" ]]; then
  CLI=("$PYTHON_BIN" -m ppg_bp_incremental.release_cli)
else
  CLI=("$PYTHON_BIN" -m ppg_bp_incremental.cli)
fi

FROZEN_ROOT="$ARTIFACT_ROOT/predictions"
FINETUNING_ROOT="$ARTIFACT_ROOT/finetuning"
BENCHMARK_ROOT="$ARTIFACT_ROOT/benchmark"
FIGURE_ROOT="$ARTIFACT_ROOT/figures"
INFLUENCE_ROOT="$ARTIFACT_ROOT/demographic_influence"
PAPER_OUTPUT_ROOT="$ARTIFACT_ROOT/paper_outputs"
[[ -d "$FROZEN_ROOT" ]] || fail "missing frozen/Ridge predictions: $FROZEN_ROOT"
[[ -d "$FINETUNING_ROOT" ]] || fail "missing fine-tuning predictions: $FINETUNING_ROOT"

mapfile -d '' FROZEN_PREDICTIONS < <(
  find "$FROZEN_ROOT" -type f -name '*.csv' -print0 | sort -z
)
mapfile -d '' NEURAL_PREDICTIONS < <(
  find "$FINETUNING_ROOT" -type f -name 'predictions.csv' -print0 | sort -z
)
if ((${#FROZEN_PREDICTIONS[@]} != 44)); then
  fail "expected 44 frozen/Ridge files; found ${#FROZEN_PREDICTIONS[@]}"
fi
if ((${#NEURAL_PREDICTIONS[@]} != 960)); then
  fail "expected 960 fine-tuning files; found ${#NEURAL_PREDICTIONS[@]}"
fi
PREDICTIONS=("${FROZEN_PREDICTIONS[@]}" "${NEURAL_PREDICTIONS[@]}")

"${CLI[@]}" aggregate \
  --predictions "${PREDICTIONS[@]}" \
  --output-root "$BENCHMARK_ROOT" \
  --bootstrap-replicates 10000 \
  --bootstrap-confidence 0.95 \
  --seed 20260715 \
  --require-complete-matrix

"${CLI[@]}" plot \
  --aggregated-predictions "$BENCHMARK_ROOT/aggregated_predictions.csv" \
  --metrics "$BENCHMARK_ROOT/metrics.csv" \
  --output-root "$FIGURE_ROOT"

"$PYTHON_BIN" "$REPO_ROOT/scripts/analyze_demographic_influence.py" \
  --aggregated-predictions "$BENCHMARK_ROOT/aggregated_predictions.csv" \
  --raw-predictions "$BENCHMARK_ROOT/predictions.csv" \
  --seed-specific-aggregated-predictions \
  "$BENCHMARK_ROOT/seed_specific_aggregated_predictions.csv" \
  --output-root "$INFLUENCE_ROOT" \
  --bootstrap-replicates 10000 \
  --bootstrap-confidence 0.95 \
  --seed 20260830

"$PYTHON_BIN" "$REPO_ROOT/scripts/build_paper_outputs.py" \
  --metrics "$BENCHMARK_ROOT/metrics.csv" \
  --paired-contrasts "$INFLUENCE_ROOT/paired_contrasts.csv" \
  --constants "$INFLUENCE_ROOT/demographics_vs_outer_constant.csv" \
  --complementarity "$INFLUENCE_ROOT/complementarity.csv" \
  --aggregated-predictions "$BENCHMARK_ROOT/aggregated_predictions.csv" \
  --fine-tuning-contrasts "$INFLUENCE_ROOT/fine_tuning_paired_contrasts.csv" \
  --output-root "$PAPER_OUTPUT_ROOT"

printf 'Full benchmark aggregation complete: %s\n' "$BENCHMARK_ROOT/metrics.csv"
