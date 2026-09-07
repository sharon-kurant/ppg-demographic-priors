# Source-audited foundation-model comparison

This is the literature and executable-code report for contract
`source-faithful-v3`. It is deliberately separate from the empirical benchmark
report. Published numbers below are context; they are not pooled with our new
predictions.

Primary sources:

- [PaPaGei paper](https://arxiv.org/abs/2410.20542) and
  [pinned source](https://github.com/Nokia-Bell-Labs/papagei-foundation-model/commit/0c537dad4d2850e15b724260de820dd68d77f0b0)
- [Pulse-PPG paper](https://arxiv.org/abs/2502.01108) and
  [pinned source](https://github.com/maxxu05/pulseppg/commit/716eaf9cf966e8f76436f2263872ef38b1f90166)
- [AnyPPG paper, arXiv v4](https://arxiv.org/abs/2511.01747) and
  [pinned source](https://github.com/Ngk03/AnyPPG/commit/661b877aa96eac3bba320a462cb2a3bfea991103)

## Executable checkpoint contracts

| Model | Architecture and parameters | Active downstream input | Pooling | Native checkpoint output | Selected BP representation |
|---|---|---|---|---|---|
| PaPaGei-P | 18-block ResNet1D; 4,993,024 | 10 s at 125 Hz; `(B,1,1250)` | backbone pooling plus downstream dense layer | downstream dense `(B,512)`; pooled `(B,512)` | downstream dense embedding |
| PaPaGei-S | 18-block ResNet1D MoE; 5,785,612 | 10 s at 125 Hz; `(B,1,1250)` | backbone pooling plus downstream dense layer | downstream dense `(B,512)`; IPA `(B,1)`; SQI `(B,1)`; pooled `(B,512)` | downstream dense embedding |
| Pulse-PPG | 12 residual blocks/26 convolutions; 28,497,920 | BP path: 10 s at 50 Hz; `(B,1,500)` | released checkpoint uses max | embedding `(B,512)` | returned embedding |
| AnyPPG | ResNet-derived Net1D; 4,044,272 | 10 s at 125 Hz; `(B,1,1250)` | temporal mean | embedding `(B,512)` | returned embedding |

For PaPaGei/Pulse-PPG downstream preprocessing, the pinned pyPPG-equivalent
implementation runs its 50 ms PPG smoothing branch only when the source rate is
at least 75 Hz. Consequently, it is applied to PPG-BP and PulseDB but skipped
for the 30 Hz BUT PPG adaptation.

## Active downstream preprocessing contracts

| Model | Ordered waveform operations before the checkpoint | Fidelity boundary |
|---|---|---|
| PaPaGei-P/S | on PPG-BP only, remove the final stored sample; per-record z-score; 0.5–12 Hz fourth-order Chebyshev-II/20 dB band-pass; 50 ms smoothing when source rate is at least 75 Hz; polyphase resampling to 125 Hz; symmetric zero-padding only for short PPG-BP | released-exact on PPG-BP; source-consistent cross-cohort adaptation elsewhere |
| Pulse-PPG | on PPG-BP only, remove the final stored sample; released PPG-BP z-score/filter/smoothing sequence; polyphase resampling to 50 Hz; symmetric zero-padding only for short PPG-BP | released-exact on PPG-BP; source-consistent cross-cohort adaptation elsewhere |
| AnyPPG | MNE zero-phase 0.5–8 Hz third-order Butterworth filter; polyphase resampling to 125 Hz; one checkpoint-facing time-axis z-score; symmetric zero-padding only for short PPG-BP | source-consistent adaptation because exact downstream BP-array construction is unavailable |

PulseDB inputs are raw `PPG_Record`, not an already normalized or filtered
field. Thus every contract performs its own filtering exactly once. PPG-BP is
the only active cohort requiring duration padding; PulseDB and BUT PPG are
already ten seconds long.

Checkpoint SHA-256 values are, respectively,
`3a6850961af527cbb2e476d3ad0bb374ae86ed86ffdd55b0b0d7e2f49451518e`,
`79d68671e51bdc951fee853e5a8241099e38148ff26ed888b843849332e0b988`,
`485ade5033b3baa9b82e252fc131042dda897d7bdfc0d1a030d9746a1e98857c`,
and
`99b9bb0a3c2b83a1f5d8ca2963fbd25329b6530e8d337de8825722fc6fd5f4fa`.

Native outputs are not BP predictions and have no inferred BP units. In the
active benchmark, only an explicit target-specific estimator emits a BP-task
output: Ridge for frozen probes or the benchmark-defined neural head for
end-to-end fine-tuning.

## Pretraining comparison

| Model | Data and scale | Pretraining input/preprocessing | Objective and loss |
|---|---|---|---|
| PaPaGei-P | VitalDB, MIMIC-III and MESA; 13,517 participants, 20,751,206 segments, 57,641 h | non-overlapping 10 s; 0.5–12 Hz fourth-order Chebyshev filtering; flatline rejection; per-segment z-normalization; 125 Hz | participant identity defines positive pairs; NT-Xent |
| PaPaGei-S | same corpus and input | same base preparation; morphology values sVRI, IPA and SQI are derived before training | sVRI-defined contrastive pairs plus IPA/SQI MoE regression; contrastive loss plus MAE auxiliary losses |
| Pulse-PPG | MOODS field study; 120 participants; 822,247 four-minute 50 Hz windows | global person-specific z-normalization; paper intentionally applies no signal-specific filter | motif reconstruction model uses MSE; the PPG encoder uses relative contrastive (RelCon) learning over within- and between-person candidates |
| AnyPPG | MC-MED, PulseDB, MESA, HSP and CFS; 58,796 participants, 39,566,943 paired segments, 109,909 h | non-overlapping 10 s; PPG 0.5–8 Hz; resample 125 Hz; time-axis z-normalization | frozen ECGFounder guidance and trainable PPG branch; symmetric CLIP-style InfoNCE |

AnyPPG calls these five named datasets; counting PulseDB's MIMIC and VitalDB
sources separately yields the paper abstract's six data sources.

## Demographics: five distinct roles

The word “demographics” is easy to overstate, so the roles are separated here.

| Model | Direct encoder input | Pretraining supervision or sampling metadata | Demographic prediction task | Downstream fusion | Published BP-task input |
|---|---|---|---|---|---|
| PaPaGei-P | none | participant identity, not age/sex values, defines positive pairs; this can indirectly retain subject-specific and demographic information | age regression/classification and sex classification are evaluated from embeddings | paper separately fuses age/sex with PaPaGei-S | standard PaPaGei-P BP result: no; appendix fusion result is variant-ambiguous |
| PaPaGei-S | none | sVRI/IPA/SQI morphology, not demographic values | age and sex prediction from embeddings | age/sex concatenated downstream in the appendix | yes only for the separate `PaPaGei-S + Demo` appendix result |
| Pulse-PPG | none | within/between-person sampling; no age/sex values | none reported | demographics discussed as future context | none |
| AnyPPG | none | no demographic supervision in encoder pretraining | age is one downstream task; age/sex/race are also phenotype baselines | demographic baselines are compared, not fused into published BP probes | none |

Thus PaPaGei did use participant identity in PaPaGei-P and explicitly studied
demographic information, but age and sex were not direct inputs to either
encoder. The PaPaGei appendix table also reports a downstream age/sex fusion
experiment.

The PaPaGei paper is internally inconsistent about its demographic-prediction
values. Its prose says PaPaGei-S obtains age MAE 7.78, age-classification 0.85,
and sex-classification 0.79; the adjacent table gives 8.78, 0.85, and 0.74. Both
are recorded here rather than silently choosing one.

## Downstream BP estimators and outputs

| Model | Published BP estimator/head | Target/output scale and decoder | Dataset/split | Demographics | Reported MAE (SBP/DBP) |
|---|---|---|---|---|---|
| PaPaGei-P | standardized embedding + Ridge | released code fits raw labels; identity mmHg decode | PPG-BP; subject-level 60/20/20 out-of-domain split described for downstream tasks | no | 13.60 / 8.88 |
| PaPaGei-P | standardized embedding + Ridge | paper reports regression on BP labels and MAE in mmHg; the public repository does not provide the Vital Videos target construction or decoder | Vital Videos; subject-level 60/20/20 out-of-domain split | no | 19.11 / 10.87 |
| PaPaGei-S | standardized embedding + Ridge | raw labels; identity decode | PPG-BP; same protocol | no | 14.39 / 8.71 |
| PaPaGei-S | standardized embedding + Ridge | paper reports regression on BP labels and MAE in mmHg; the public repository does not provide the Vital Videos target construction or decoder | Vital Videos; same protocol | no | 14.65 / 8.29 |
| PaPaGei appendix fusion | PaPaGei representation + age/sex in downstream model | appendix reports BP-label regression and MAE in mmHg; no separate decoder is described | PPG-BP; same subject-disjoint out-of-domain protocol | age, sex | 13.20 / 8.61 |
| PaPaGei appendix fusion | PaPaGei-S representation + age/sex in downstream model | appendix reports BP-label regression and MAE in mmHg; no separate decoder is described | Vital Videos; same subject-disjoint out-of-domain protocol | age, sex | 14.27 / 8.26 |
| Pulse-PPG frozen | standardized embedding + L2 linear regression | released probe code consumes raw labels; identity decode | PPG-BP | no | 13.62 / 8.878 |
| Pulse-PPG fine-tuned | paper specifies `512→128→GELU→1`, MSE, Adam | target transform and decoder are not disclosed in released code | PPG-BP | no | 12.33 / 8.695 |
| AnyPPG linear probe | standardized embedding + Ridge; inner five-fold selection | released probe pattern uses supplied labels directly; identity decode | BUT PPG and UCI-BP; paper uses official or 80/20 protocols depending on identifiable subjects | no | BUT 13.31 / 9.62; UCI 15.62 / 7.14 |

## Common benchmark fine-tuning adapter

The active end-to-end experiment deliberately uses one declared BP adapter
across all four checkpoints so that the encoder update itself can be compared:

`selected 512-D representation [ + demographics ] -> Linear(128) -> GELU -> Linear(1)`

The scalar estimates a BP z-score fitted only on the selection-training pool or
the full-development refit pool, as appropriate. The recorded affine inverse
transform returns the output to raw mmHg. This transform is part of supervised
training and is not test-time calibration. SBP and DBP use separate models.

The complete selected encoder path and the BP head are trainable. PaPaGei-S's
IPA and SQI values remain recorded native outputs but their auxiliary heads are
excluded from BP optimization. The loss is participant-balanced z-score MSE.
Training uses Adam, a head learning rate ten times the encoder rate, gradient
clipping at 1.0, batch size 64, validation-MAE epoch selection, and at most ten
epochs. It is a `source_consistent_adaptation`, not a released-exact BP recipe.

This distinction is essential when comparing with published results. The
Pulse-PPG paper specifies the same 512-to-128-to-1 topology, GELU, MSE, and
Adam, but its regression code and BP target transform are unavailable.
PaPaGei and AnyPPG do not release an end-to-end BP fine-tuning recipe for these
cohorts. Their published values therefore remain literature context rather
than validation targets for the common adapter.

The raw Pulse-PPG embedding produced substantially larger gradients under the
common neural head than the other encoders. The audited post-diagnostic
sensitivity therefore adds a fixed epoch-zero standardizer fitted only on the
applicable training pool and zero-initializes the scalar output layer. This
changes optimization, not the waveform input contract or target decoder. Its
results are reported separately from the prespecified unstandardized run.

Vital Videos and PulseDB-Vital are distinct cohorts. The PaPaGei paper describes
Vital Videos as an ongoing study of 231 participants from Europe and
Sub-Saharan Africa. It is not one of our benchmark cohorts and its published
results must not be interpreted as PulseDB-Vital results. The pinned public
PaPaGei repository provides a PPG-BP example notebook, but no corresponding
Vital Videos data path or downstream evaluation notebook. The Vital Videos
values above are consequently paper-reported context, not a released-code
replication.

The PaPaGei demographic appendix header says “PaPaGei-S or -P” for its
non-demographic reference while the fusion column says PaPaGei-S + Demo. For
Vital Videos, the reference values 14.65/8.29 match the main PaPaGei-S results,
and the fusion column reports 14.27/8.26. For PPG-BP, however, the reference
13.60/8.71 pair draws its SBP value from PaPaGei-P and its DBP value from
PaPaGei-S in the main table. The 13.20/8.61 fusion result is therefore retained
with the appendix's variant-label ambiguity. Its prose also claims SBP/DBP
deltas of −0.11/−0.65 in one paragraph, while the displayed PPG-BP table values
imply −0.40/−0.10 relative to 13.60/8.71. A later paragraph states 0.40/0.10
without a sign convention. The table values are reported, and the inconsistent
prose is not used to calculate our benchmark effects.

## Source discrepancies and unavailable details

- PaPaGei-S executable code returns four components. Shorter README examples do
  not fully describe the IPA, SQI, and pooled tensors.
- Pulse-PPG's paper says global average temporal pooling, but the released
  checkpoint configuration instantiates `finalpool="max"`. The active contract
  follows executable code.
- Pulse-PPG describes a regression fine-tuning head and MSE but does not release
  the corresponding regression training path or disclose a BP target transform.
  Its published fine-tuned number remains literature context. Our active
  neural Pulse-PPG result is explicitly labeled a common benchmark
  reconstruction rather than released-exact reproduction.
- AnyPPG releases the encoder and generic linear-probe code, but not the exact
  construction of every downstream `.npz`, including BUT PPG/UCI-BP. Our
  downstream preparation is labeled `source_consistent_adaptation`, not exact
  replication.
- AnyPPG's released PulseDB constructor normalizes stored segments with
  epsilon `1e-8`, and its pretraining loader normalizes them again with epsilon
  `1e-5`. The active contract follows the checkpoint-facing loader convention:
  MNE filtering (including the upstream float32 cast), resampling, then one
  per-record z-normalization with epsilon `1e-5`. Because the exact downstream
  `.npz` path is unavailable, this remains a documented source-consistent
  adaptation outside the released PulseDB preprocessing path.

## Pretraining overlap with the active cohorts

| Model | PPG-BP | PulseDB-Vital | PulseDB-MIMIC | BUT PPG |
|---|---|---|---|---|
| PaPaGei-P/S | none known | source overlap through VitalDB | source overlap through MIMIC-III | none |
| Pulse-PPG | none | none | none | none |
| AnyPPG | none | exact PulseDB dataset overlap | exact PulseDB dataset overlap | none |

AnyPPG's PulseDB results must always be described as pretraining-overlap
evaluation. They are not independent external validation.
