# Pulse-PPG optimization stabilization sensitivity

This is an isolated, post-diagnostic amendment. It does not overwrite the
prespecified primary benchmark. Fixed epoch-zero, training-pool embedding
standardization and zero scalar-output initialization replace only the
fine-tuned Pulse-PPG arms.

- Stabilized MAE is lower than the raw unstandardized run in **16/16** endpoints.
- Stabilized MAE is lower than the matched frozen Pulse-PPG probe in **6/16** endpoints.
- Median maximum pre-clipping gradient norm changes from **552.13** to **15.79**.

| Cohort | Target | Arm | Raw FT MAE | Stabilized FT MAE | Frozen MAE | Stable - frozen |
|---|---|---|---:|---:|---:|---:|
| BUT PPG | DBP | No Demo | 7.229 | 6.606 | 7.228 | -0.622 |
| BUT PPG | DBP | + Demo | 7.617 | 6.604 | 6.633 | -0.029 |
| BUT PPG | SBP | No Demo | 13.836 | 12.220 | 12.811 | -0.591 |
| BUT PPG | SBP | + Demo | 15.890 | 12.264 | 10.250 | +2.014 |
| PPG-BP | DBP | No Demo | 21.499 | 8.846 | 8.511 | +0.335 |
| PPG-BP | DBP | + Demo | 13.252 | 8.861 | 7.774 | +1.087 |
| PPG-BP | SBP | No Demo | 32.258 | 16.219 | 13.586 | +2.633 |
| PPG-BP | SBP | + Demo | 27.269 | 16.214 | 12.964 | +3.251 |
| PulseDB-MIMIC | DBP | No Demo | 11.726 | 10.000 | 9.970 | +0.030 |
| PulseDB-MIMIC | DBP | + Demo | 11.184 | 9.990 | 9.652 | +0.338 |
| PulseDB-MIMIC | SBP | No Demo | 20.538 | 18.398 | 17.643 | +0.755 |
| PulseDB-MIMIC | SBP | + Demo | 20.481 | 18.348 | 17.652 | +0.696 |
| PulseDB-Vital | DBP | No Demo | 10.946 | 9.236 | 9.319 | -0.082 |
| PulseDB-Vital | DBP | + Demo | 10.580 | 9.246 | 9.122 | +0.124 |
| PulseDB-Vital | SBP | No Demo | 16.569 | 14.124 | 14.306 | -0.182 |
| PulseDB-Vital | SBP | + Demo | 16.760 | 14.069 | 14.205 | -0.137 |

Negative values in the last column favor stabilized fine-tuning. All MAEs are
participant-balanced and expressed in mmHg. Neural values are means of the
three independently scored seeds.
