#!/bin/bash

set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 PARTS_DIRECTORY [EXTRACT_DIRECTORY]" >&2
  exit 2
fi

PARTS_DIRECTORY=$1
EXTRACT_DIRECTORY=${2:-$PARTS_DIRECTORY/extracted}
SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MANIFEST=${PULSEDB_ARCHIVE_MANIFEST:-$SCRIPT_ROOT/pulsedb_v2_archives.tsv}

if command -v 7z >/dev/null 2>&1; then
  SEVEN_ZIP=7z
elif command -v 7zz >/dev/null 2>&1; then
  SEVEN_ZIP=7zz
else
  echo "7-Zip is required (install p7zip-full or 7zip)" >&2
  exit 1
fi
for executable in sha1sum awk; do
  if ! command -v "$executable" >/dev/null 2>&1; then
    echo "Required executable is unavailable: $executable" >&2
    exit 1
  fi
done
if [[ ! -f "$MANIFEST" ]]; then
  echo "Archive manifest not found: $MANIFEST" >&2
  exit 1
fi

while IFS=$'\t' read -r cohort filename expected_sha1 _url; do
  [[ "$cohort" == "cohort" || -z "$cohort" ]] && continue
  path="$PARTS_DIRECTORY/$filename"
  if [[ ! -f "$path" ]]; then
    echo "Missing PulseDB archive part: $path" >&2
    exit 1
  fi
  actual_sha1=$(sha1sum "$path" | awk '{print $1}')
  if [[ "$actual_sha1" != "$expected_sha1" ]]; then
    echo "Checksum mismatch for $path" >&2
    exit 1
  fi
done < "$MANIFEST"

mkdir -p "$EXTRACT_DIRECTORY"
for cohort in MIMIC Vital; do
  first_part="$PARTS_DIRECTORY/PulseDB_${cohort}.zip.001"
  "$SEVEN_ZIP" t "$first_part" >/dev/null
  "$SEVEN_ZIP" x -y "$first_part" "-o$EXTRACT_DIRECTORY" >/dev/null
  echo "Extracted PulseDB_${cohort} under $EXTRACT_DIRECTORY"
done
