# Data acquisition

No waveform data are distributed with this repository. Download each dataset
from its original provider and keep it outside Git. The helper scripts require
`curl`, standard checksum tools, and the extraction tools noted below.

## PPG-BP v5

The helper downloads the Figshare archive, verifies its published MD5, and
extracts it. `unzip` is required.

```bash
scripts/download_ppgbp.sh \
  data/raw/ppgbp/PPG-BP_Database_v5.zip \
  data/raw/ppgbp/extracted

ppg-bp prepare \
  --dataset ppgbp \
  --raw-root data/raw/ppgbp/extracted \
  --source-archive data/raw/ppgbp/PPG-BP_Database_v5.zip
```

## PulseDB v2.0

PulseDB publishes MIMIC and Vital as multipart 7-Zip-compatible archives. The
download helper obtains and verifies all 26 official parts. Downloads may also
be distributed across an array by supplying a zero-based part index from 0 to
25. After every part is present, the extraction helper verifies every SHA-1
again and asks 7-Zip to join and extract both archives. Install either `7z`
(commonly `p7zip-full`) or `7zz` first.

```bash
scripts/download_pulsedb_v2.sh data/raw/pulsedb/parts
scripts/extract_pulsedb_v2.sh \
  data/raw/pulsedb/parts \
  data/raw/pulsedb/extracted

ppg-bp prepare \
  --dataset pulsedb-vital \
  --raw-root data/raw/pulsedb/extracted/PulseDB_Vital \
  --subject-limit 300 \
  --segments-per-subject 10 \
  --selection-seed 42

ppg-bp prepare \
  --dataset pulsedb-mimic \
  --raw-root data/raw/pulsedb/extracted/PulseDB_MIMIC \
  --subject-limit 300 \
  --segments-per-subject 10 \
  --selection-seed 42
```

If the archive creates one additional containing directory, point `--raw-root`
at the directory that contains the participant `.mat` files. The loader scans
subdirectories recursively.

## BUT PPG v2.0.0

The PhysioNet helper downloads, validates, and extracts the public archive.

```bash
scripts/download_butppg.sh data/raw/butppg

ppg-bp prepare \
  --dataset butppg \
  --raw-root data/raw/butppg/butppg-2.0.0 \
  --segments-per-subject 20 \
  --selection-seed 42
```

BUT PPG records are retained only when the provider's `Quality = 1` indicator
is present. BMI is calculated from height and weight; height and weight are not
used as direct predictors.

## Study subsets and local outputs

The reported PulseDB subsets contain 300 participants from each source cohort
and ten deterministically selected eligible 10-second segments per participant
(seed 42). Vital and MIMIC are always prepared, split, and reported separately.
PPG-BP uses all eligible participants. BUT PPG retains at most 20 records per
participant.

Local paths belong only in generated manifests. Do not commit raw files,
participant-level manifests, cached waveforms, embeddings, predictions, or
cluster logs.
