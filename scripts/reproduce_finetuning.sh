#!/usr/bin/env bash

# Reproduce the source-faithful-v3 end-to-end fine-tuning matrix from prepared
# cohort manifests and locked subject-disjoint splits.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reproduce_finetuning.sh [OPTIONS]

The data root must contain:
  processed/{ppgbp,pulsedb_vital,pulsedb_mimic,butppg}.csv
  manifests/{ppgbp,pulsedb_vital,pulsedb_mimic,butppg}_splits.csv

Options:
  --data-root PATH        Prepared-data root
                          (default: data/reproduction/source-faithful-v3).
  --output-root PATH      Fine-tuning artifact root
                          (default: artifacts/reproduction/source-faithful-v3/finetuning).
  --checkpoint-root PATH  Transient resumable-checkpoint root. By default,
                          checkpoints are written below the output root.
  --device DEVICE         auto, cpu, cuda, or cuda:N (default: auto).
  --num-workers N         Data-loader workers (default: 4).
  --resume                Skip runs with predictions and run metadata present.
  -h, --help              Show this message.

The reported configuration uses a 512-to-128-to-1 GELU head, participant-
balanced z-score MSE, batch size 64, gradient clipping at 1.0, patience 3,
and at most 10 epochs. Seed 17 selects the validation epoch; seeds 23 and 42
reuse that selection and refit from the released checkpoint. Pulse-PPG uses
fixed epoch-zero training-pool embedding standardization and zero output-layer
initialization, as implemented by the public CLI defaults.
EOF
}

fail() {
  printf 'reproduce_finetuning.sh: %s\n' "$*" >&2
  exit 1
}

CALLER_PWD=$PWD
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
DATA_ROOT="$REPO_ROOT/data/reproduction/source-faithful-v3"
OUTPUT_ROOT="$REPO_ROOT/artifacts/reproduction/source-faithful-v3/finetuning"
CHECKPOINT_ROOT=
DEVICE=auto
NUM_WORKERS=4
RESUME=0

while (($#)); do
  case "$1" in
    --data-root)
      (($# >= 2)) || fail "$1 requires a path"
      DATA_ROOT=$2
      shift 2
      ;;
    --output-root)
      (($# >= 2)) || fail "$1 requires a path"
      OUTPUT_ROOT=$2
      shift 2
      ;;
    --checkpoint-root)
      (($# >= 2)) || fail "$1 requires a path"
      CHECKPOINT_ROOT=$2
      shift 2
      ;;
    --device)
      (($# >= 2)) || fail "$1 requires a device"
      DEVICE=$2
      shift 2
      ;;
    --num-workers)
      (($# >= 2)) || fail "$1 requires an integer"
      NUM_WORKERS=$2
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
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

make_absolute() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$CALLER_PWD" "$1" ;;
  esac
}

DATA_ROOT=$(make_absolute "$DATA_ROOT")
OUTPUT_ROOT=$(make_absolute "$OUTPUT_ROOT")
if [[ -z "$CHECKPOINT_ROOT" ]]; then
  CHECKPOINT_ROOT="$OUTPUT_ROOT/checkpoints"
else
  CHECKPOINT_ROOT=$(make_absolute "$CHECKPOINT_ROOT")
fi
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || fail "--num-workers must be non-negative"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python not found: $PYTHON_BIN"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
if [[ -f "$REPO_ROOT/src/ppg_bp_incremental/release_cli.py" ]]; then
  CLI=("$PYTHON_BIN" -m ppg_bp_incremental.release_cli)
else
  CLI=("$PYTHON_BIN" -m ppg_bp_incremental.cli)
fi

TOKENS=(ppgbp pulsedb_vital pulsedb_mimic butppg)
COHORT_DIRS=(ppg_bp pulsedb_vital pulsedb_mimic but_ppg)
MODELS=(papagei_p papagei_s pulseppg anyppg)
CONDITIONS=(finetuned finetuned_demographics)
TARGETS=(sbp dbp)
FOLDS=(0 1 2 3 4)
SEEDS=(17 23 42)

for token in "${TOKENS[@]}"; do
  [[ -s "$DATA_ROOT/processed/$token.csv" ]] || \
    fail "missing prepared manifest: $DATA_ROOT/processed/$token.csv"
  [[ -s "$DATA_ROOT/manifests/${token}_splits.csv" ]] || \
    fail "missing locked splits: $DATA_ROOT/manifests/${token}_splits.csv"
done

mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_ROOT"
printf 'Starting fine-tuning matrix at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

completed=0
skipped=0
for dataset_index in "${!TOKENS[@]}"; do
  token=${TOKENS[$dataset_index]}
  cohort_dir=${COHORT_DIRS[$dataset_index]}
  manifest="$DATA_ROOT/processed/$token.csv"
  splits="$DATA_ROOT/manifests/${token}_splits.csv"
  for model in "${MODELS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
      for target in "${TARGETS[@]}"; do
        for fold in "${FOLDS[@]}"; do
          for seed in "${SEEDS[@]}"; do
            run_root="$OUTPUT_ROOT/$cohort_dir/$model/$condition/$target/fold_$fold/seed_$seed"
            if ((RESUME)) && [[ -s "$run_root/predictions.csv" && -s "$run_root/run.json" ]]; then
              ((skipped += 1))
              continue
            fi
            "${CLI[@]}" run-finetune \
              --input-csv "$manifest" \
              --splits "$splits" \
              --model "$model" \
              --condition "$condition" \
              --target "$target" \
              --fold "$fold" \
              --seed "$seed" \
              --batch-size 64 \
              --max-epochs 10 \
              --patience 3 \
              --gradient-clip-norm 1.0 \
              --num-workers "$NUM_WORKERS" \
              --no-mixed-precision \
              --device "$DEVICE" \
              --output-root "$OUTPUT_ROOT" \
              --checkpoint-root "$CHECKPOINT_ROOT"
            ((completed += 1))
          done
        done
      done
    done
  done
done

printf 'Fine-tuning matrix complete at %s: %d run(s) executed, %d reused.\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$completed" "$skipped"
printf 'Artifacts: %s\n' "$OUTPUT_ROOT"
