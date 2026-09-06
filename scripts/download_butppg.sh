#!/bin/bash

set -euo pipefail

if [[ "$#" -gt 1 ]]; then
  echo "Usage: $0 [DESTINATION_DIRECTORY]" >&2
  exit 2
fi

DESTINATION=${1:-data/raw/butppg}
VERSION=2.0.0
URL="https://physionet.org/content/butppg/get-zip/${VERSION}/"
ARCHIVE="$DESTINATION/butppg-${VERSION}.zip"

for executable in curl unzip; do
  if ! command -v "$executable" >/dev/null 2>&1; then
    echo "Required executable is unavailable: $executable" >&2
    exit 1
  fi
done

mkdir -p "$DESTINATION"
curl --fail --location --retry 5 --continue-at - --output "$ARCHIVE" "$URL"
unzip -tq "$ARCHIVE"
unzip -q -o "$ARCHIVE" -d "$DESTINATION"

echo "Downloaded BUT PPG ${VERSION} to $DESTINATION"
echo "Source: https://physionet.org/content/butppg/${VERSION}/"
