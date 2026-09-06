#!/bin/bash

set -euo pipefail

ROOT="${1:-external/papagei}"
REPOSITORY="$ROOT/repository"
WEIGHTS="$ROOT/weights"
COMMIT="0c537dad4d2850e15b724260de820dd68d77f0b0"
REMOTE="https://github.com/Nokia-Bell-Labs/papagei-foundation-model.git"

if [[ ! -d "$REPOSITORY/.git" ]]; then
  mkdir -p "$ROOT"
  git clone --no-checkout "$REMOTE" "$REPOSITORY"
  git -C "$REPOSITORY" checkout --detach "$COMMIT"
else
  CURRENT="$(git -C "$REPOSITORY" rev-parse HEAD)"
  if [[ "$CURRENT" != "$COMMIT" ]]; then
    echo "Existing PaPaGei checkout is at $CURRENT, expected $COMMIT" >&2
    exit 1
  fi
fi

mkdir -p "$WEIGHTS"

download_weight() {
  local name="$1"
  local expected_md5="$2"
  local output="$WEIGHTS/$name"
  if [[ -f "$output" ]] && [[ "$(md5sum "$output" | awk '{print $1}')" == "$expected_md5" ]]; then
    echo "Already verified $output"
    return
  fi
  curl -fL "https://zenodo.org/api/records/13983110/files/$name/content" -o "$output.tmp"
  local actual_md5
  actual_md5="$(md5sum "$output.tmp" | awk '{print $1}')"
  if [[ "$actual_md5" != "$expected_md5" ]]; then
    rm -f "$output.tmp"
    echo "Checksum mismatch for $name: expected $expected_md5, received $actual_md5" >&2
    exit 1
  fi
  mv "$output.tmp" "$output"
  echo "Downloaded and verified $output"
}

download_weight "papagei_p.pt" "052b50807465fae61e08e2b7acbb5c53"
download_weight "papagei_s.pt" "a4cdb32392e2a7b25999128af92813b5"
