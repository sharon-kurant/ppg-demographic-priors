# Demographic Priors in Frozen and Fine-Tuned PPG Foundation Models for Calibration-Free Blood Pressure Estimation

This repository contains the reproducibility code and non-participant aggregate
results for a subject-disjoint comparison of demographic priors and four
photoplethysmography (PPG) foundation models. The study asks how much age, sex,
and body-mass index (BMI) contribute to systolic and diastolic blood-pressure
(SBP and DBP) estimation, both alone and when combined with waveform
representations.

The release is limited to the methods reported in the conference paper. It
does not include participant data, third-party model code or checkpoints,
cached embeddings, fitted estimators, row-level predictions, cluster files, or
discarded exploratory experiments.

## Benchmark

Four foundation models are evaluated separately on four cohorts:

| Foundation model | Parameters | Model input |
|---|---:|---|
| PaPaGei-P | 5.0M | 10 s at 125 Hz, `(B, 1, 1250)` |
| PaPaGei-S | 5.8M | 10 s at 125 Hz, `(B, 1, 1250)` |
| Pulse-PPG | 28.5M | 10 s at 50 Hz, `(B, 1, 500)` |
| AnyPPG | 4.0M | 10 s at 125 Hz, `(B, 1, 1250)` |

The cohorts are PPG-BP, PulseDB-Vital, PulseDB-MIMIC, and BUT PPG. The active
conditions are:

1. demographics only;
2. 38 handcrafted PPG features;
3. handcrafted PPG features plus demographics;
4. frozen foundation-model embedding;
5. frozen embedding plus demographics;
6. end-to-end fine-tuning; and
7. end-to-end fine-tuning plus demographics.

Age, sex, and BMI are used wherever available. PulseDB-MIMIC provides age and
sex but not BMI. The other cohorts use all three fields. BMI is calculated from
height and weight when a supplied BMI is unavailable; height and weight are
not direct predictors.

## Model and BP-output contracts

Each encoder uses its source-audited filtering, normalization, sampling rate,
window length, and native output structure. Native checkpoint outputs remain
representations and are never assigned BP units.

For frozen conditions, the designated 512-dimensional embedding enters a
separate Ridge estimator for SBP or DBP. Ridge is trained directly on raw-mmHg
labels and uses identity decoding.

For fine-tuning, the complete selected encoder path and a `512 → 128 → 1` GELU
head are optimized jointly. The head predicts a BP z-score fitted only within
the applicable training pool. Its recorded affine inverse transform returns
the final prediction to mmHg. This train-only target transform is part of
optimization; it is not subject-specific or post-hoc calibration.

Pulse-PPG fine-tuning uses fixed epoch-zero training-pool embedding
standardization and zero initialization of the scalar output layer. These
stabilization choices were introduced after diagnosing a representation-scale
mismatch and are explicitly recorded in the released configuration and audit
tables.

## Evaluation design

All methods within a cohort use the same records and five subject-disjoint
outer folds.

For Ridge, three subject-disjoint folds inside each outer development pool
select alpha by participant-macro MAE over `{0.1, 1, 10, 100, 1000}`. The
selected pipeline is refitted on the full development pool, then applied once
to untouched outer-test participants.

For fine-tuning, locked validation participants select an epoch for seed 17.
The model is restarted from the released checkpoint and refitted for that many
epochs on the full development pool. Seeds 23 and 42 reuse the same selection
and are independently refitted and scored. Optimization uses participant-
balanced z-score MSE, Adam, batch size 64, gradient clipping at 1.0, patience
three, and a maximum of ten epochs.

Repeated PPG-BP records are averaged within participant before scoring.
PulseDB measurements and BUT PPG sessions use participant-balanced scoring.
Confidence intervals use 10,000 participant-level bootstrap resamples of the
fixed pooled out-of-fold predictions; bootstrapping does not repeat model
selection or training.

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[foundation-models]"
```

The exact neural runtime is recorded in
[`environment.finetuning.yml`](environment.finetuning.yml). The later
aggregation and paper-output environment is recorded in
[`environment.paper.yml`](environment.paper.yml).

## Data and checkpoints

Download all datasets, upstream repositories, and checkpoints from their
original providers. See [`docs/data_acquisition.md`](docs/data_acquisition.md).
Pinned commits and checkpoint SHA-256 values are listed in
[`configs/sources.yaml`](configs/sources.yaml).

PulseDB-Vital and PulseDB-MIMIC are prepared as separate 300-participant
cohorts with ten deterministic segments per participant. PulseDB ingestion
uses raw `PPG_Record`; already processed `PPG_F` is rejected, preventing
double filtering. PPG-BP alone is shorter than the model windows and is
zero-padded after model-specific resampling.

## Reproduction

First prepare the cohorts, extract frozen representations, and run all Ridge
conditions:

```bash
bash scripts/reproduce_benchmark.sh \
  --ppgbp-root /path/to/PPG-BP \
  --pulsedb-vital-root /path/to/PulseDB_Vital \
  --pulsedb-mimic-root /path/to/PulseDB_MIMIC \
  --butppg-root /path/to/BUT-PPG \
  --data-root data/reproduction/source-faithful-v3 \
  --artifact-root artifacts/reproduction/source-faithful-v3 \
  --setup-models
```

Then run the neural matrix. Seed 17 is deliberately evaluated before seeds 23
and 42 because it creates the reusable epoch-selection artifact.

```bash
bash scripts/reproduce_finetuning.sh \
  --data-root data/reproduction/source-faithful-v3 \
  --output-root artifacts/reproduction/source-faithful-v3/finetuning \
  --device cuda \
  --resume
```

Finally combine the two stages and rebuild figures and demographic analyses:

```bash
bash scripts/aggregate_full_benchmark.sh \
  --artifact-root artifacts/reproduction/source-faithful-v3
```

Individual stages are also exposed through `ppg-bp`; run
`ppg-bp <command> --help` for their interfaces.

## Included results

The [`results`](results) directory contains only aggregate, non-identifying
tables. `benchmark_metrics.csv` contains all reported conditions;
`seed_metrics.csv` preserves the three neural seeds; the paired-contrast files
quantify the effect of demographics. Sanitized acceptance receipts establish
matrix completeness and numerical z-score-to-mmHg decoding without publishing
participant rows.

The aggregate evidence records that some source files were subsequently
clarified for provenance labels, target-scope naming, and the PaPaGei dense
output alias. Those edits did not alter the executed numerical path; both the
executed-source hashes and this release's complete tree hash are retained.

## Repository layout

```text
configs/                 versioned benchmark and source identities
docs/                    acquisition, preprocessing, protocol, and model audit
results/                 non-participant aggregate tables and audit receipts
scripts/                 acquisition, reproduction, aggregation, and plotting
src/ppg_bp_incremental/  benchmark implementation and public CLI
```

Citation metadata are in [`CITATION.cff`](CITATION.cff). Datasets, upstream
model code, and checkpoints remain subject to their original providers' terms.
