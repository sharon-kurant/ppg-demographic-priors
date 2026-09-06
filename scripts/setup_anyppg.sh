#!/bin/bash
set -euo pipefail

REPOSITORY="external/anyppg/repository"
REPOSITORY_URL="https://github.com/Ngk03/AnyPPG.git"
COMMIT="661b877aa96eac3bba320a462cb2a3bfea991103"
CHECKPOINT="$REPOSITORY/load_anyppg/anyppg_ckpt.pth"
CHECKPOINT_SHA256="99b9bb0a3c2b83a1f5d8ca2963fbd25329b6530e8d337de8825722fc6fd5f4fa"

mkdir -p "$(dirname "$REPOSITORY")"
if [[ ! -d "$REPOSITORY/.git" ]]; then
  git clone "$REPOSITORY_URL" "$REPOSITORY"
fi

git -C "$REPOSITORY" fetch origin "$COMMIT"
git -C "$REPOSITORY" checkout --detach "$COMMIT"
test "$(git -C "$REPOSITORY" rev-parse HEAD)" = "$COMMIT"
test -f "$CHECKPOINT"
echo "$CHECKPOINT_SHA256  $CHECKPOINT" | sha256sum --check --status

echo "AnyPPG official repository and checkpoint verified"
