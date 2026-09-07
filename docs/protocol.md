# Evaluation protocol

## Locked outer evaluation

The primary estimand is performance on previously unseen participants. Every
method within a cohort uses the same selected records and the same five
subject-disjoint outer folds. Each fold contains outer-training, validation,
and untouched test participants. Training and validation together form the
outer development pool used for final refitting.

All imputation, standardization, target transformations, feature fitting, and
model selection are learned without outer-test participants. SBP and DBP are
modeled separately. Final reported predictions are always in mmHg.

## Frozen representations and Ridge baselines

Three subject-disjoint inner folds are constructed inside each outer
development pool. Numeric imputation, missingness indicators, standardization,
and sex one-hot encoding are fit on each inner-training split. Ridge alpha is
chosen separately for each target by participant-macro MAE over the pooled
inner out-of-fold predictions.

The alpha grid is `{0.1, 1, 10, 100, 1000}`. The selected pipeline is refitted
on the complete outer development pool and applied once to outer-test
participants. Ridge is fitted to raw-mmHg labels and has identity decoding.

## End-to-end fine-tuning

Fine-tuning updates the complete encoder path that produces the designated
512-dimensional embedding. A `512 → 128 → 1` head, with GELU between the two
linear layers, is trained jointly with the encoder. When demographics are
used, their train-fitted vector is concatenated with the embedding before the
head.

The neural head predicts a z-score defined by the applicable training pool.
The transform mean and standard deviation are fitted with the same
participant-balanced weights as the loss. A recorded affine inverse transform
decodes each prediction to raw mmHg. No test label, per-person calibration, or
post-hoc calibration is used.

Seed 17 selects the epoch by validation MAE, with a maximum of ten epochs and
patience three. The model is then restarted from the released checkpoint and
refitted for the selected epoch count on the full development pool. Seeds 23
and 42 reuse the seed-17 selection and perform independent restarts and
refits. The fixed model-level encoder rates are `3e-5` for PaPaGei-P,
PaPaGei-S, and AnyPPG, and `1e-5` for Pulse-PPG. The head rate is ten times the
encoder rate. Adam, batch size 64, participant-balanced MSE, and gradient
clipping at 1.0 are used.

Pulse-PPG additionally uses epoch-zero embedding standardization fitted only
on the relevant training pool and zero initialization of the scalar output
layer. Batch-normalization running statistics remain trainable under the
reported setting.

## Demographics

Every available field from age, sex, and BMI is used. PulseDB-MIMIC uses age
and sex; the other cohorts use age, sex, and BMI. Numeric fields are
median-imputed and standardized using training data only. Sex is one-hot
encoded with an explicit unknown category. Missingness indicators are added
for numeric fields with sporadic missingness. BMI is calculated from height
and weight when needed, but height and weight are not predictors.

## Scoring and uncertainty

Repeated PPG-BP recordings are averaged within participant before scoring.
PulseDB segments are first averaged by measurement and then macro-averaged
across participants. BUT PPG sessions are also participant-balanced. Neural
point estimates are the mean of the three independently scored seeds.

Confidence intervals use 10,000 participant-level bootstrap draws over the
fixed pooled out-of-fold predictions. The bootstrap sits on top of the
cross-validation predictions; it does not repeat splitting, preprocessing,
selection, or training. Neural seed variation is reported separately in
`results/seed_metrics.csv`.
