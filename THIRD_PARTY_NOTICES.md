# Third-party sources and notices

This repository contains integration and reproducibility code only. It does
not redistribute third-party datasets, upstream source trees, model
checkpoints, or weights. Users must obtain each resource from its original
provider and comply with the provider's current access and use terms.

## Foundation models

The executable model contracts are pinned to the following upstream commits.
The terms column records the license file observed at that exact source
revision.

| Model | Upstream source | Pinned commit | Observed upstream terms |
|---|---|---|---|
| PaPaGei-P and PaPaGei-S | [Nokia Bell Labs, `papagei-foundation-model`](https://github.com/Nokia-Bell-Labs/papagei-foundation-model) | `0c537dad4d2850e15b724260de820dd68d77f0b0` | [BSD 3-Clause](https://github.com/Nokia-Bell-Labs/papagei-foundation-model/blob/0c537dad4d2850e15b724260de820dd68d77f0b0/LICENSE) |
| Pulse-PPG | [maxxu05, `pulseppg`](https://github.com/maxxu05/pulseppg) | `716eaf9cf966e8f76436f2263872ef38b1f90166` | [MIT](https://github.com/maxxu05/pulseppg/blob/716eaf9cf966e8f76436f2263872ef38b1f90166/LICENSE) |
| AnyPPG | [Ngk03, `AnyPPG`](https://github.com/Ngk03/AnyPPG) | `661b877aa96eac3bba320a462cb2a3bfea991103` | [No `LICENSE` file was observed at the pinned commit](https://github.com/Ngk03/AnyPPG/tree/661b877aa96eac3bba320a462cb2a3bfea991103) |

The release expects the user to clone these repositories at the listed commits
and download checkpoints through the upstream mechanisms. Runtime contracts
verify checkpoint SHA-256 values before use. Upstream code and weights retain
their own copyright and terms; nothing in this repository changes those terms.
No AnyPPG source file, checkpoint, or weight is redistributed here.

Relevant model publications:

- [PaPaGei](https://arxiv.org/abs/2410.20542)
- [Pulse-PPG](https://arxiv.org/abs/2502.01108)
- [AnyPPG](https://arxiv.org/abs/2511.01747)

Pretraining provenance matters when interpreting the benchmark. PaPaGei has
source overlap with VitalDB- and MIMIC-derived cohorts. AnyPPG was pretrained
on PulseDB, so its PulseDB-Vital and PulseDB-MIMIC evaluations are marked as
exact-dataset-overlap evaluations. Pulse-PPG has no identified overlap with the
four benchmark cohorts.

## Datasets

| Cohort used in the study | Original provider | Dataset terms |
|---|---|---|
| PPG-BP | [PPG-BP Database, Figshare, version 5](https://doi.org/10.6084/m9.figshare.5459299.v5) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| PulseDB-Vital | [PulseLabTeam/PulseDB](https://github.com/pulselabteam/PulseDB), VitalDB-source cohort | [CC BY-NC-SA 4.0](https://github.com/pulselabteam/PulseDB/blob/db0824f18d9a462458e46fe94c31283a93a5c0d5/LICENSE_PulseDB_Vital) |
| PulseDB-MIMIC | [PulseLabTeam/PulseDB](https://github.com/pulselabteam/PulseDB), MIMIC-III-source cohort | [ODbL 1.0](https://github.com/pulselabteam/PulseDB/blob/db0824f18d9a462458e46fe94c31283a93a5c0d5/LICENSE_PulseDB_MIMIC) for database rights and the [Database Contents License](https://opendatacommons.org/licenses/dbcl/1-0/) for individual contents |
| BUT PPG | [BUT PPG, PhysioNet, version 2.0.0](https://physionet.org/content/butppg/2.0.0/) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The two PulseDB source cohorts are analyzed separately. Corrected ingestion
reads raw `PPG_Record`; already processed `PPG_F` waveforms are not used for the
source-faithful benchmark.

The repository does not grant access to any dataset. Data citations, licenses,
credentialing requirements, and usage restrictions must be checked at the
provider before download or publication of derived material.

## Python dependencies

Python packages are installed from their normal package indexes and are not
vendored. Their licenses and notices remain available from the respective
projects. The direct dependencies are declared in `pyproject.toml`.
