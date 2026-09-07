# Signal preprocessing contracts (source-faithful-v3)

All cohort loaders preserve the original selected waveform. PulseDB uses
`PPG_Record`; already filtered `PPG_F` is rejected. Handcrafted features are
computed from the original unpadded waveform.

## PaPaGei-P and PaPaGei-S

Each record is z-standardized (epsilon `1e-7`), filtered at 0.5–12 Hz with a
fourth-order Chebyshev-II 20-dB band-pass, smoothed with a zero-phase 50-ms
moving average only when the source rate is at least 75 Hz, and polyphase-
resampled to 125 Hz. The final input is `(B, 1, 1250)`.

## Pulse-PPG

The released downstream BP path is used. PPG-BP alone drops its final stored
sample before z-standardization, the 0.5–12 Hz Chebyshev-II filter,
rate-conditional smoothing, and resampling to 50 Hz. The executable checkpoint
uses temporal maximum pooling. The final input is `(B, 1, 500)`. This downstream
path differs from Pulse-PPG's unfiltered, person-normalized MOODS pretraining
pipeline.

## AnyPPG

The source order is a third-order 0.5–8 Hz Butterworth filter with MNE
zero-phase semantics, polyphase resampling to 125 Hz, and time-axis
z-standardization (epsilon `1e-5`). The final input is `(B, 1, 1250)`.

## Length handling

Only the approximately 2.1-second PPG-BP records are shorter than the model
windows. They are symmetrically zero-padded after duration-preserving
resampling. No reflection or repetition policy is part of the released study.
PulseDB and BUT PPG require no padding.

The same preprocessed model tensor is used by the frozen and fine-tuned paths.
Fine-tuning changes encoder weights but does not alter the waveform contract.
