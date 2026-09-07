#!/usr/bin/env bash

# Reproduce the source-faithful-v3 frozen/Ridge stage from extracted datasets.
# Participant data, foundation-model source trees, and checkpoints remain external
# to the repository; this script records all derived artifacts beneath dedicated
# data and artifact roots.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reproduce_benchmark.sh \
    --ppgbp-root PATH \
    --pulsedb-vital-root PATH \
    --pulsedb-mimic-root PATH \
    --butppg-root PATH [OPTIONS]

Required dataset roots must point to extracted source releases. The script
reproduces all four cohorts, four frozen foundation models, two frozen
conditions, three baselines, and both SBP and DBP targets. Run
`scripts/reproduce_finetuning.sh` afterward for the two neural conditions.

Options:
  --ppgbp-root PATH             Extracted PPG-BP dataset root.
  --pulsedb-vital-root PATH     Extracted PulseDB-Vital v2 root.
  --pulsedb-mimic-root PATH     Extracted PulseDB-MIMIC v2 root.
  --butppg-root PATH            Extracted BUT PPG v2.0.0 root.
  --ppgbp-archive PATH          Optional original PPG-BP archive for provenance.
  --butppg-archive PATH         Optional original BUT PPG archive for provenance.
  --data-root PATH              Derived manifests/caches root
                                (default: data/reproduction/source-faithful-v3).
  --artifact-root PATH          Embeddings/results root
                                (default: artifacts/reproduction/source-faithful-v3).
  --device DEVICE               Encoder device: auto, cpu, cuda, or cuda:N
                                (default: auto).
  --setup-models                Download and verify the pinned public models first.
  --resume                      Reuse validated manifests, splits, and completed
                                prediction files from an interrupted run.
  -h, --help                    Show this message.

Exact reported settings:
  * five subject-disjoint outer folds and three inner subject-disjoint folds;
  * Ridge alpha grid {0.1, 1, 10, 100, 1000}, selected by inner MAE;
  * 300 subjects x 10 segments for each PulseDB cohort (selection seed 42);
  * at most 20 official-quality segments per BUT PPG subject (seed 42);
  * frozen-representation extraction batch size 256;
  * 10,000 participant-level bootstrap replicates.

Examples:
  # First run, including pinned model setup:
  bash scripts/reproduce_benchmark.sh \
    --ppgbp-root /data/PPG-BP \
    --pulsedb-vital-root /data/PulseDB/Vital \
    --pulsedb-mimic-root /data/PulseDB/MIMIC \
    --butppg-root /data/BUT_PPG \
    --setup-models

  # Continue an interrupted run:
  bash scripts/reproduce_benchmark.sh \
    --ppgbp-root /data/PPG-BP \
    --pulsedb-vital-root /data/PulseDB/Vital \
    --pulsedb-mimic-root /data/PulseDB/MIMIC \
    --butppg-root /data/BUT_PPG \
    --resume
EOF
}

fail() {
  printf 'reproduce_benchmark.sh: %s\n' "$*" >&2
  exit 1
}

CALLER_PWD=$PWD
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

PPGBP_ROOT=
PULSEDB_VITAL_ROOT=
PULSEDB_MIMIC_ROOT=
BUTPPG_ROOT=
PPGBP_ARCHIVE=
BUTPPG_ARCHIVE=
DATA_ROOT="$REPO_ROOT/data/reproduction/source-faithful-v3"
ARTIFACT_ROOT="$REPO_ROOT/artifacts/reproduction/source-faithful-v3"
DEVICE=auto
SETUP_MODELS=0
RESUME=0

while (($#)); do
  case "$1" in
    --ppgbp-root)
      (($# >= 2)) || fail "$1 requires a path"
      PPGBP_ROOT=$2
      shift 2
      ;;
    --pulsedb-vital-root)
      (($# >= 2)) || fail "$1 requires a path"
      PULSEDB_VITAL_ROOT=$2
      shift 2
      ;;
    --pulsedb-mimic-root)
      (($# >= 2)) || fail "$1 requires a path"
      PULSEDB_MIMIC_ROOT=$2
      shift 2
      ;;
    --butppg-root)
      (($# >= 2)) || fail "$1 requires a path"
      BUTPPG_ROOT=$2
      shift 2
      ;;
    --ppgbp-archive)
      (($# >= 2)) || fail "$1 requires a path"
      PPGBP_ARCHIVE=$2
      shift 2
      ;;
    --butppg-archive)
      (($# >= 2)) || fail "$1 requires a path"
      BUTPPG_ARCHIVE=$2
      shift 2
      ;;
    --data-root)
      (($# >= 2)) || fail "$1 requires a path"
      DATA_ROOT=$2
      shift 2
      ;;
    --artifact-root)
      (($# >= 2)) || fail "$1 requires a path"
      ARTIFACT_ROOT=$2
      shift 2
      ;;
    --device)
      (($# >= 2)) || fail "$1 requires a device"
      DEVICE=$2
      shift 2
      ;;
    --setup-models)
      SETUP_MODELS=1
      shift
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

[[ -n "$PPGBP_ROOT" ]] || fail "missing required option: --ppgbp-root"
[[ -n "$PULSEDB_VITAL_ROOT" ]] || fail "missing required option: --pulsedb-vital-root"
[[ -n "$PULSEDB_MIMIC_ROOT" ]] || fail "missing required option: --pulsedb-mimic-root"
[[ -n "$BUTPPG_ROOT" ]] || fail "missing required option: --butppg-root"
for variable in PPGBP_ROOT PULSEDB_VITAL_ROOT PULSEDB_MIMIC_ROOT BUTPPG_ROOT; do
  value=$(make_absolute "${!variable}")
  printf -v "$variable" '%s' "$value"
  [[ -d "$value" ]] || fail "dataset root does not exist: $value"
done

DATA_ROOT=$(make_absolute "$DATA_ROOT")
ARTIFACT_ROOT=$(make_absolute "$ARTIFACT_ROOT")
if [[ -n "$PPGBP_ARCHIVE" ]]; then
  PPGBP_ARCHIVE=$(make_absolute "$PPGBP_ARCHIVE")
  [[ -f "$PPGBP_ARCHIVE" ]] || fail "PPG-BP archive does not exist: $PPGBP_ARCHIVE"
fi
if [[ -n "$BUTPPG_ARCHIVE" ]]; then
  BUTPPG_ARCHIVE=$(make_absolute "$BUTPPG_ARCHIVE")
  [[ -f "$BUTPPG_ARCHIVE" ]] || fail "BUT PPG archive does not exist: $BUTPPG_ARCHIVE"
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python executable not found: $PYTHON_BIN"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
if [[ -f "$REPO_ROOT/src/ppg_bp_incremental/release_cli.py" ]]; then
  CLI=("$PYTHON_BIN" -m ppg_bp_incremental.release_cli)
else
  CLI=("$PYTHON_BIN" -m ppg_bp_incremental.cli)
fi

cd "$REPO_ROOT"

# The development tree nests benchmark configs; the source-only release keeps
# the same file at configs/benchmark.yaml. Support both layouts.
PROTOCOL_PATH="$REPO_ROOT/configs/benchmark/matrix.yaml"
if [[ ! -f "$PROTOCOL_PATH" ]]; then
  PROTOCOL_PATH="$REPO_ROOT/configs/benchmark.yaml"
fi
[[ -f "$PROTOCOL_PATH" ]] || fail "machine-readable benchmark protocol is missing"

# Fail if the machine-readable protocol and this executable driver drift apart.
"$PYTHON_BIN" - "$PROTOCOL_PATH" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
expected = {
    "contract_version": "source-faithful-v3",
    "models": ["papagei_p", "papagei_s", "pulseppg", "anyppg"],
    "datasets": ["ppgbp", "pulsedb_vital", "pulsedb_mimic", "butppg"],
    "targets": ["sbp", "dbp"],
    "outer_folds": 5,
    "inner_group_folds": 3,
    "ridge_alphas": [0.1, 1.0, 10.0, 100.0, 1000.0],
    "embedding_inference_batch_size": 256,
    "pulsedb_subjects_per_cohort": 300,
    "maximum_pulsedb_segments_per_subject": 10,
    "pulsedb_selection_seed": 42,
    "maximum_butppg_segments_per_subject": 20,
    "butppg_selection_seed": 42,
    "primary_padding": "zero",
    "padding_scope": "ppgbp_only",
    "final_target_scale": "raw_mmHg",
    "ridge_target_transform": "none",
}
differences = {
    key: {"expected": value, "observed": protocol.get(key)}
    for key, value in expected.items()
    if protocol.get(key) != value
}
if differences:
    raise SystemExit(f"Protocol drift in {path}: {differences}")
print(f"Validated reported protocol: {path}")
PY

if ((SETUP_MODELS)); then
  bash scripts/setup_papagei.sh
  bash scripts/setup_pulseppg.sh
  bash scripts/setup_anyppg.sh
fi

required_model_paths=(
  external/papagei/repository
  external/papagei/weights/papagei_p.pt
  external/papagei/weights/papagei_s.pt
  external/pulseppg/repository
  external/pulseppg/weights/pulseppg/experiments/out/pulseppg/checkpoint_best.pkl
  external/anyppg/repository
  external/anyppg/repository/load_anyppg/anyppg_ckpt.pth
)
missing_model_paths=()
for path in "${required_model_paths[@]}"; do
  [[ -e "$path" ]] || missing_model_paths+=("$path")
done
if ((${#missing_model_paths[@]})); then
  printf 'Missing pinned foundation-model inputs:\n' >&2
  printf '  %s\n' "${missing_model_paths[@]}" >&2
  fail "rerun with --setup-models, or run scripts/setup_{papagei,pulseppg,anyppg}.sh"
fi

PROCESSED_ROOT="$DATA_ROOT/processed"
MANIFEST_ROOT="$DATA_ROOT/manifests"
EMBEDDING_ROOT="$ARTIFACT_ROOT/embeddings"
PREDICTION_ROOT="$ARTIFACT_ROOT/predictions"
BENCHMARK_ROOT="$ARTIFACT_ROOT/benchmark"
FIGURE_ROOT="$ARTIFACT_ROOT/figures"
EDA_ROOT="$ARTIFACT_ROOT/data_eda"
INFLUENCE_ROOT="$ARTIFACT_ROOT/demographic_influence"
PAPER_OUTPUT_ROOT="$ARTIFACT_ROOT/paper_outputs"
mkdir -p \
  "$PROCESSED_ROOT" "$MANIFEST_ROOT" "$EMBEDDING_ROOT" \
  "$PREDICTION_ROOT" "$BENCHMARK_ROOT" "$FIGURE_ROOT" \
  "$EDA_ROOT" "$INFLUENCE_ROOT" "$PAPER_OUTPUT_ROOT" "$ARTIFACT_ROOT/audit"

LOG_PATH="$ARTIFACT_ROOT/reproduce_benchmark.log"
exec > >(tee -a "$LOG_PATH") 2>&1

printf 'Starting frozen/Ridge benchmark reproduction at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'Data root: %s\nArtifact root: %s\nDevice: %s\n' \
  "$DATA_ROOT" "$ARTIFACT_ROOT" "$DEVICE"

prepare_ppgbp() {
  local manifest="$PROCESSED_ROOT/ppgbp.csv"
  if ((RESUME)) && [[ -s "$manifest" ]]; then
    "${CLI[@]}" validate --input-csv "$manifest"
    return
  fi
  local command=(
    prepare --dataset ppgbp --raw-root "$PPGBP_ROOT"
    --output-csv "$manifest"
    --provenance-json "$MANIFEST_ROOT/ppgbp_provenance.json"
    --qc-manifest "$MANIFEST_ROOT/ppgbp_quality_control.csv"
  )
  if [[ -n "$PPGBP_ARCHIVE" ]]; then
    command+=(--source-archive "$PPGBP_ARCHIVE")
  fi
  "${CLI[@]}" "${command[@]}"
}

prepare_pulsedb() {
  local dataset=$1
  local raw_root=$2
  local token=${dataset//-/_}
  local manifest="$PROCESSED_ROOT/$token.csv"
  if ((RESUME)) && [[ -s "$manifest" ]]; then
    "${CLI[@]}" validate --input-csv "$manifest"
    return
  fi
  "${CLI[@]}" prepare \
    --dataset "$dataset" \
    --raw-root "$raw_root" \
    --output-csv "$manifest" \
    --provenance-json "$MANIFEST_ROOT/${token}_provenance.json" \
    --dataset-version 2.0 \
    --sample-rate-hz 125 \
    --subject-limit 300 \
    --segments-per-subject 10 \
    --selection-seed 42 \
    --subject-selection-csv "$MANIFEST_ROOT/${token}_subject_selection.csv" \
    --waveform-cache-hdf5 "$PROCESSED_ROOT/${token}_waveforms.h5"
}

prepare_butppg() {
  local manifest="$PROCESSED_ROOT/butppg.csv"
  if ((RESUME)) && [[ -s "$manifest" ]]; then
    "${CLI[@]}" validate --input-csv "$manifest"
    return
  fi
  local command=(
    prepare --dataset butppg --raw-root "$BUTPPG_ROOT"
    --output-csv "$manifest"
    --provenance-json "$MANIFEST_ROOT/butppg_provenance.json"
    --qc-manifest "$MANIFEST_ROOT/butppg_quality_control.csv"
    --segments-per-subject 20
    --selection-seed 42
  )
  if [[ -n "$BUTPPG_ARCHIVE" ]]; then
    command+=(--source-archive "$BUTPPG_ARCHIVE")
  fi
  "${CLI[@]}" "${command[@]}"
}

prepare_ppgbp
prepare_pulsedb pulsedb-vital "$PULSEDB_VITAL_ROOT"
prepare_pulsedb pulsedb-mimic "$PULSEDB_MIMIC_ROOT"
prepare_butppg

MANIFESTS=(
  "$PROCESSED_ROOT/ppgbp.csv"
  "$PROCESSED_ROOT/pulsedb_vital.csv"
  "$PROCESSED_ROOT/pulsedb_mimic.csv"
  "$PROCESSED_ROOT/butppg.csv"
)
DATASET_TOKENS=(ppgbp pulsedb_vital pulsedb_mimic butppg)

for manifest in "${MANIFESTS[@]}"; do
  "${CLI[@]}" validate --input-csv "$manifest"
done

for index in "${!MANIFESTS[@]}"; do
  manifest=${MANIFESTS[$index]}
  token=${DATASET_TOKENS[$index]}
  split_path="$MANIFEST_ROOT/${token}_splits.csv"
  if ((RESUME)) && [[ -s "$split_path" ]]; then
    printf 'Reusing locked split manifest: %s\n' "$split_path"
  else
    "${CLI[@]}" make-splits \
      --input-csv "$manifest" \
      --output "$split_path" \
      --outer-folds 5 \
      --validation-fraction 0.2 \
      --seed 20260715 \
      --force
  fi
done

# Validate resumed artifacts as strictly as newly generated ones. This prevents
# a differently sized subset or a non-disjoint split from silently entering the
# reported matrix.
"$PYTHON_BIN" - \
  "${MANIFESTS[0]}" "$MANIFEST_ROOT/ppgbp_splits.csv" \
  "${MANIFESTS[1]}" "$MANIFEST_ROOT/pulsedb_vital_splits.csv" \
  "${MANIFESTS[2]}" "$MANIFEST_ROOT/pulsedb_mimic_splits.csv" \
  "${MANIFESTS[3]}" "$MANIFEST_ROOT/butppg_splits.csv" <<'PY'
from pathlib import Path
import sys

import pandas as pd

from ppg_bp_incremental.data.benchmark import load_benchmark_manifest
from ppg_bp_incremental.data.benchmark_splits import validate_benchmark_splits

pairs = list(zip(sys.argv[1::2], sys.argv[2::2], strict=True))
for manifest_path, split_path in pairs:
    data = load_benchmark_manifest(Path(manifest_path))
    splits = pd.read_csv(split_path)
    validate_benchmark_splits(data, splits)
    if set(pd.to_numeric(splits["fold"]).astype(int)) != set(range(5)):
        raise SystemExit(f"Expected five outer folds in {split_path}")
    dataset = str(data["dataset"].iloc[0])
    counts = data.groupby("subject_id", sort=False).size()
    if dataset in {"PulseDB-Vital", "PulseDB-MIMIC"}:
        if data["subject_id"].nunique() != 300 or not counts.eq(10).all():
            raise SystemExit(
                f"Expected 300 subjects x 10 segments in {manifest_path}"
            )
    if dataset == "BUT PPG" and int(counts.max()) > 20:
        raise SystemExit(f"BUT PPG exceeds 20 segments per subject: {manifest_path}")
    print(
        f"Validated {dataset}: {data['subject_id'].nunique()} subjects, "
        f"{len(data)} segments, five subject-disjoint outer folds"
    )
PY

"${CLI[@]}" audit-models \
  --output-json "$ARTIFACT_ROOT/audit/models.json" \
  --device "$DEVICE"

TEMP_FILES=()
cleanup() {
  if ((${#TEMP_FILES[@]})); then
    rm -f "${TEMP_FILES[@]}"
  fi
}
trap cleanup EXIT

LAST_ARTIFACT=
extract_representation() {
  local model=$1
  local manifest=$2
  local output_root=$3
  local transcript
  transcript=$(mktemp "${TMPDIR:-/tmp}/ppg-bp-extract.XXXXXX")
  TEMP_FILES+=("$transcript")
  "${CLI[@]}" extract \
    --model "$model" \
    --input-csv "$manifest" \
    --output-root "$output_root" \
    --device "$DEVICE" \
    --batch-size 256 | tee "$transcript"
  LAST_ARTIFACT=$(sed -n 's/^Validated representation artifact: //p' "$transcript" | tail -n 1)
  [[ -n "$LAST_ARTIFACT" && -d "$LAST_ARTIFACT" ]] || \
    fail "could not resolve the validated $model representation artifact"
}

PREDICTIONS=()
run_ridge() {
  local manifest=$1
  local splits=$2
  local condition=$3
  local model=$4
  local artifact=$5
  local output=$6
  PREDICTIONS+=("$output")
  if ((RESUME)) && [[ -s "$output" && -s "${output%.csv}.run.json" ]]; then
    printf 'Reusing completed prediction file: %s\n' "$output"
    return
  fi
  local command=(
    run-ridge
    --input-csv "$manifest"
    --splits "$splits"
    --condition "$condition"
    --model "$model"
    --targets sbp dbp
    --seed 20260715
    --ridge-alphas 0.1 1 10 100 1000
    --output "$output"
  )
  if [[ -n "$artifact" ]]; then
    command+=(--embedding-artifact "$artifact")
  fi
  "${CLI[@]}" "${command[@]}"
}

MODELS=(papagei_p papagei_s pulseppg anyppg)
for index in "${!MANIFESTS[@]}"; do
  manifest=${MANIFESTS[$index]}
  token=${DATASET_TOKENS[$index]}
  split_path="$MANIFEST_ROOT/${token}_splits.csv"
  dataset_predictions="$PREDICTION_ROOT/$token"
  mkdir -p "$dataset_predictions"

  run_ridge \
    "$manifest" "$split_path" demographics demographics "" \
    "$dataset_predictions/demographics.csv"

  extract_representation handcrafted_ppg "$manifest" "$EMBEDDING_ROOT/$token"
  handcrafted_artifact=$LAST_ARTIFACT
  run_ridge \
    "$manifest" "$split_path" ppg_features handcrafted_ppg "$handcrafted_artifact" \
    "$dataset_predictions/handcrafted_ppg_features.csv"
  run_ridge \
    "$manifest" "$split_path" ppg_features_demographics handcrafted_ppg "$handcrafted_artifact" \
    "$dataset_predictions/handcrafted_ppg_features_demographics.csv"

  for model in "${MODELS[@]}"; do
    extract_representation "$model" "$manifest" "$EMBEDDING_ROOT/$token"
    model_artifact=$LAST_ARTIFACT
    run_ridge \
      "$manifest" "$split_path" frozen "$model" "$model_artifact" \
      "$dataset_predictions/${model}_frozen.csv"
    run_ridge \
      "$manifest" "$split_path" frozen_demographics "$model" "$model_artifact" \
      "$dataset_predictions/${model}_frozen_demographics.csv"
  done
done

if ((${#PREDICTIONS[@]} != 44)); then
  fail "internal matrix error: expected 44 prediction files, found ${#PREDICTIONS[@]}"
fi
for path in "${PREDICTIONS[@]}"; do
  [[ -s "$path" ]] || fail "prediction file is missing or empty: $path"
done

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

"${CLI[@]}" data-eda \
  --input-csv "${MANIFESTS[@]}" \
  --output-root "$EDA_ROOT"

"$PYTHON_BIN" scripts/analyze_demographic_influence.py \
  --aggregated-predictions "$BENCHMARK_ROOT/aggregated_predictions.csv" \
  --raw-predictions "$BENCHMARK_ROOT/predictions.csv" \
  --output-root "$INFLUENCE_ROOT" \
  --bootstrap-replicates 10000 \
  --bootstrap-confidence 0.95 \
  --seed 20260830

"$PYTHON_BIN" scripts/build_paper_outputs.py \
  --metrics "$BENCHMARK_ROOT/metrics.csv" \
  --paired-contrasts "$INFLUENCE_ROOT/paired_contrasts.csv" \
  --constants "$INFLUENCE_ROOT/demographics_vs_outer_constant.csv" \
  --complementarity "$INFLUENCE_ROOT/complementarity.csv" \
  --aggregated-predictions "$BENCHMARK_ROOT/aggregated_predictions.csv" \
  --without-fine-tuning \
  --output-root "$PAPER_OUTPUT_ROOT"

printf 'Frozen/Ridge stage completed at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'Metrics: %s\nFigures: %s\nDemographic analysis: %s\nPaper outputs: %s\nLog: %s\n' \
  "$BENCHMARK_ROOT/metrics.csv" "$FIGURE_ROOT" "$INFLUENCE_ROOT" \
  "$PAPER_OUTPUT_ROOT" "$LOG_PATH"
