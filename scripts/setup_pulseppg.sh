#!/bin/bash

set -euo pipefail

ROOT="${1:-external/pulseppg}"
REPOSITORY="$ROOT/repository"
WEIGHTS="$ROOT/weights"
ARCHIVE="$ROOT/pulseppg_model_weights_v4.zip"
COMMIT="716eaf9cf966e8f76436f2263872ef38b1f90166"
REMOTE="https://github.com/maxxu05/pulseppg.git"
URL="https://zenodo.org/api/records/17345536/files/pulseppg_model_weights.zip/content"
EXPECTED_MD5="2d0ebda9afeb9648674464098a698c37"

if [[ ! -d "$REPOSITORY/.git" ]]; then
  mkdir -p "$ROOT"
  git clone --no-checkout "$REMOTE" "$REPOSITORY"
  git -C "$REPOSITORY" checkout --detach "$COMMIT"
else
  CURRENT="$(git -C "$REPOSITORY" rev-parse HEAD)"
  if [[ "$CURRENT" != "$COMMIT" ]]; then
    echo "Existing Pulse-PPG checkout is at $CURRENT, expected $COMMIT" >&2
    exit 1
  fi
fi

if [[ ! -f "$ARCHIVE" ]] || [[ "$(md5sum "$ARCHIVE" | awk '{print $1}')" != "$EXPECTED_MD5" ]]; then
  curl -fL "$URL" -o "$ARCHIVE.tmp"
  ACTUAL_MD5="$(md5sum "$ARCHIVE.tmp" | awk '{print $1}')"
  if [[ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]]; then
    rm -f "$ARCHIVE.tmp"
    echo "Pulse-PPG checksum mismatch: expected $EXPECTED_MD5, received $ACTUAL_MD5" >&2
    exit 1
  fi
  mv "$ARCHIVE.tmp" "$ARCHIVE"
fi

mkdir -p "$WEIGHTS"
unzip -q -o "$ARCHIVE" -d "$WEIGHTS"
echo "Prepared Pulse-PPG repository and weights beneath $ROOT"
