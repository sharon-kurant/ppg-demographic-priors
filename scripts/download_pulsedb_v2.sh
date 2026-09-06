#!/bin/bash

set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 DESTINATION [ZERO_BASED_PART_INDEX]" >&2
  exit 2
fi

DESTINATION=$1
PART_INDEX=${2:-${SLURM_ARRAY_TASK_ID:-}}
SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MANIFEST=${PULSEDB_ARCHIVE_MANIFEST:-$SCRIPT_ROOT/pulsedb_v2_archives.tsv}

for executable in curl sha1sum flock; do
  if ! command -v "$executable" >/dev/null 2>&1; then
    echo "Required executable is unavailable: $executable" >&2
    exit 1
  fi
done
if [[ ! -f "$MANIFEST" ]]; then
  echo "Archive manifest not found: $MANIFEST" >&2
  exit 1
fi

mapfile -t ENTRIES < <(tail -n +2 "$MANIFEST" | sed '/^[[:space:]]*$/d')
if [[ "${#ENTRIES[@]}" -ne 26 ]]; then
  echo "Expected 26 official PulseDB archive parts, found ${#ENTRIES[@]}" >&2
  exit 1
fi
mkdir -p "$DESTINATION"

download_one() {
  local index=$1
  if (( index < 0 || index >= ${#ENTRIES[@]} )); then
    echo "Part index $index is outside 0-$(( ${#ENTRIES[@]} - 1 ))" >&2
    return 2
  fi

  local cohort filename expected_sha1 url
  IFS=$'\t' read -r cohort filename expected_sha1 url <<<"${ENTRIES[$index]}"
  local final_path="$DESTINATION/$filename"
  local partial_path="$final_path.partial"
  local lock_path="$final_path.lock"

  exec 9>"$lock_path"
  if ! flock -n 9; then
    echo "Another process is already downloading $filename" >&2
    return 1
  fi

  if [[ -f "$final_path" ]]; then
    if [[ "$(sha1sum "$final_path" | awk '{print $1}')" == "$expected_sha1" ]]; then
      echo "$filename already exists and passed SHA-1 verification"
      return 0
    fi
    mv "$final_path" "$final_path.checksum-failed.$(date -u +%Y%m%dT%H%M%SZ)"
  fi

  echo "Downloading PulseDB v2.0 $cohort part $filename"
  curl \
    --fail \
    --location \
    --continue-at - \
    --retry 20 \
    --retry-all-errors \
    --retry-delay 15 \
    --output "$partial_path" \
    "$url"

  local actual_sha1
  actual_sha1=$(sha1sum "$partial_path" | awk '{print $1}')
  if [[ "$actual_sha1" != "$expected_sha1" ]]; then
    mv "$partial_path" "$partial_path.checksum-failed.$(date -u +%Y%m%dT%H%M%SZ)"
    echo "$filename failed SHA-1 verification: expected $expected_sha1, observed $actual_sha1" >&2
    return 1
  fi
  mv "$partial_path" "$final_path"
  echo "$filename download and SHA-1 verification complete"
}

if [[ -n "$PART_INDEX" ]]; then
  download_one "$PART_INDEX"
else
  for index in "${!ENTRIES[@]}"; do
    download_one "$index"
  done
fi
