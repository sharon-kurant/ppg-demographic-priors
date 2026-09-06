#!/bin/bash

set -euo pipefail

OUTPUT="${1:-data/raw/ppgbp/PPG-BP_Database_v5.zip}"
EXTRACT_ROOT="${2:-data/raw/ppgbp/extracted}"
EXPECTED_MD5="3b8eae44f45799aeb3c5f55af8589828"
URL="https://ndownloader.figshare.com/files/9441097"

for executable in curl md5sum unzip; do
  if ! command -v "$executable" >/dev/null 2>&1; then
    echo "Required executable is unavailable: $executable" >&2
    exit 1
  fi
done
mkdir -p "$(dirname "$OUTPUT")"
if [[ ! -f "$OUTPUT" ]] || [[ "$(md5sum "$OUTPUT" | awk '{print $1}')" != "$EXPECTED_MD5" ]]; then
  curl --fail --location --retry 5 --continue-at - "$URL" -o "$OUTPUT"
fi

ACTUAL_MD5="$(md5sum "$OUTPUT" | awk '{print $1}')"
if [[ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]]; then
  echo "Checksum mismatch: expected $EXPECTED_MD5, received $ACTUAL_MD5" >&2
  exit 1
fi

mkdir -p "$EXTRACT_ROOT"
unzip -tq "$OUTPUT"
unzip -q -o "$OUTPUT" -d "$EXTRACT_ROOT"

echo "Downloaded and verified $OUTPUT"
echo "Extracted PPG-BP v5 under $EXTRACT_ROOT"
