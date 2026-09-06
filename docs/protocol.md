# Evaluation protocol

The primary estimand is performance on previously unseen participants. All
methods within a cohort use the same selected records and the same five
subject-disjoint outer folds.

For each outer fold, the training and validation roles form an outer
development pool. Three subject-disjoint inner folds are constructed inside
that pool. Numeric imputation, missingness indicators, standardization, and
sex one-hot encoding are fit only on inner-training rows. Ridge alpha is chosen
separately for each target by participant-macro MAE over the pooled inner
out-of-fold predictions. The selected pipeline is refit on the complete outer
development pool and applied once to untouched outer-test participants.

The primary alpha grid is `{0.1, 1, 10, 100, 1000}`. SBP and DBP are fit as
separate raw-mmHg tasks with identity decoding. Native encoder outputs are
representations and never receive BP units.

PulseDB measurements and BUT PPG sessions are averaged before participant
macro-scoring. Repeated PPG-BP recordings are averaged within participant.
Confidence intervals use 10,000 participant-level bootstrap draws over the
fixed pooled out-of-fold predictions; the bootstrap does not repeat splitting,
preprocessing, or alpha selection.
