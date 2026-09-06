# Demographic Priors and Frozen Photoplethysmography Foundation Models for Calibration-Free Blood Pressure Estimation

This repository contains the reproducibility code and non-participant
aggregate results for the study
*Demographic Priors and Frozen Photoplethysmography Foundation Models for
Calibration-Free Blood Pressure Estimation*. The study asks how much frozen
photoplethysmography (PPG)
representations add beyond age, sex, and body-mass index (BMI) when systolic
and diastolic blood pressure (SBP and DBP) are evaluated on held-out people.

The repository is intentionally limited to the analyses reported in the
conference paper. It does not contain participant data, model checkpoints,
cached embeddings, fitted estimators, or row-level predictions. Non-participant
aggregate result tables are included so the paper's focused summary figure and
table can be rebuilt without access to private intermediate artifacts.

## Study scope

The primary benchmark compares four frozen foundation-model encoders:

- PaPaGei-P;
- PaPaGei-S;
- Pulse-PPG; and
- AnyPPG.

The models are evaluated separately on four cohorts:

- PPG-BP;
- PulseDB-Vital;
- PulseDB-MIMIC; and
- BUT PPG.

Every encoder uses its source-audited sampling rate, signal length,
normalization, filtering, and native output contract. Native checkpoint outputs
remain representations; they are never interpreted directly as BP. PaPaGei-P,
PaPaGei-S, Pulse-PPG, and AnyPPG contribute the designated 512-dimensional
embedding to separate raw-mmHg SBP and DBP estimators. PPG-BP alone requires
zero-padding after model-specific resampling. PulseDB is read from the raw
`PPG_Record` field so filtering is applied exactly once.

The active comparisons are:

1. frozen foundation-model embedding;
2. the same embedding plus every demographic field available in the cohort;
3. demographics only;
4. 38 handcrafted PPG features; and
5. the same handcrafted features plus available demographics.

Available demographics are age, sex, and BMI. PulseDB-MIMIC has no BMI and
therefore uses age and sex; the other three cohorts use age, sex, and BMI. When
needed, BMI is calculated from height and weight, while height and weight are
not used as direct predictors.

## Evaluation design

The primary benchmark uses five subject-disjoint outer folds. For each outer
fold, Ridge alpha is selected by mean absolute error (MAE) over
`{0.1, 1, 10, 100, 1000}` using three subject-disjoint inner folds drawn only
from the outer-training participants. The selected pipeline is then refitted on
the full outer-development pool and predicts the untouched outer-test
participants. Encoders, handcrafted baselines, and demographic conditions use
the same locked outer folds and evaluation rows.

Numerical imputation, scaling, demographic encoding, and alpha selection are
fit on training data only. PulseDB measurements and BUT PPG sessions are scored
with participant-balanced aggregation; repeated PPG-BP recordings are averaged
within participant before scoring. Participant bootstrap intervals are computed
from the pooled out-of-fold predictions.

## Installation

Python 3.10 or later is required. Create an isolated environment and install
the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[foundation-models,dev]"
```

The foundation-model extra installs PyTorch. Model extraction may be run on a
CUDA device, while preprocessing, Ridge fitting, aggregation, and plotting can
run on CPU.

Verify the installation with:

```bash
ppg-bp --help
pytest
```

## Data and checkpoints

Data and checkpoints must be downloaded from their original providers. They
must not be committed to this repository. The code validates pinned upstream
model commits and checkpoint hashes before extraction; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the authoritative source
links.

A complete run follows this order:

1. download the upstream datasets and model repositories/checkpoints;
2. prepare and validate a normalized cohort manifest;
3. create the locked subject-disjoint splits;
4. audit the real model checkpoints;
5. extract frozen embeddings or handcrafted features;
6. run the nested Ridge conditions; and
7. aggregate predictions and generate the paper analyses.

After downloading and extracting the resources, pass their roots to the
reproduction driver:

```bash
bash scripts/reproduce_benchmark.sh \
  --ppgbp-root /path/to/PPG-BP \
  --pulsedb-vital-root /path/to/PulseDB_Vital \
  --pulsedb-mimic-root /path/to/PulseDB_MIMIC \
  --butppg-root /path/to/BUT-PPG \
  --setup-models
```

The driver exposes the exact reported settings as command-line calls. The YAML
files are human-readable records of those settings; they are not parsed by the
CLI. Use `ppg-bp <command> --help` for individual stages. All locations are
supplied at run time, and no local machine paths are embedded in the release.

The recorded runtime used for the reported run is summarized in
[`environment.paper.yml`](environment.paper.yml). Exact upstream source and
checkpoint identities are listed in [`configs/sources.yaml`](configs/sources.yaml).

## Repository layout

```text
configs/                 versioned study configurations
docs/                    data, protocol, preprocessing, and model documentation
results/                 non-participant aggregate tables reported in the paper
scripts/                 data acquisition and analysis helpers
src/ppg_bp_incremental/  benchmark implementation and command-line interface
tests/                   unit, contract, and smoke tests
```

## Reproducibility safeguards

- Dataset manifests record source identifiers, waveform hashes, demographic
  availability, and subject roles.
- Model contracts record the upstream commit, checkpoint SHA-256, input
  preprocessing, native tensor structure, and selected representation.
- Every run records its split fingerprint, preprocessing contract, alpha,
  dependency versions, and output scale.
- Pretraining overlap is carried as result metadata. In particular, AnyPPG
  results on both PulseDB cohorts are labeled as exact-dataset-overlap
  evaluations.
- The release excludes all data, weights, caches, row-level predictions,
  generated figures, and machine-specific cluster logs.

## Results and citation

Rebuild the paper's focused summary outputs from the included aggregate values:

```bash
python scripts/build_paper_outputs.py
```

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The original
project code is released under the [MIT License](LICENSE); upstream datasets,
model code, and weights remain governed by their providers' terms.
