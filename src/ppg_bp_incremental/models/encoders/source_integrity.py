"""Strict authentication helpers for executable upstream model sources.

Foundation-model wrappers execute Python source from pinned upstream Git
checkouts.  A matching ``HEAD`` alone is insufficient because tracked files
can be edited without changing the commit.  These helpers therefore require
both the declared commit and a clean tracked working tree, and authenticate
checkpoint bytes against a declared SHA-256 before deserialization.

Untracked files are deliberately ignored: official checkpoints are sometimes
downloaded into the checkout and need not be committed.  They are
independently authenticated by :func:`verify_checkpoint_sha256`.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess


def file_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one regular file."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"File not found: {resolved}")
    digest = sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_sha256(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    model_name: str,
) -> str:
    """Authenticate a checkpoint before any framework deserializes it."""

    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"{model_name} contract has an invalid checkpoint SHA-256")
    observed = file_sha256(checkpoint_path)
    if observed != expected:
        raise ValueError(
            f"{model_name} checkpoint SHA-256 {observed} does not match pinned "
            f"SHA-256 {expected}"
        )
    return observed


def _git_output(repository_path: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository_path), *arguments],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "output", "")
        suffix = f": {str(detail).strip()}" if str(detail).strip() else ""
        raise ValueError(
            f"Upstream repository must be a readable Git checkout{suffix}"
        ) from error


def verify_git_checkout(
    repository_path: str | Path,
    *,
    expected_commit: str,
    model_name: str,
) -> str:
    """Require the pinned commit and no tracked working-tree modifications.

    ``git status --untracked-files=no`` catches staged changes, unstaged
    changes, tracked deletions, and modified tracked submodules.  Untracked
    artifacts are outside this source check because checkpoint files are
    verified separately by content hash.
    """

    repository = Path(repository_path)
    if not repository.is_dir():
        raise FileNotFoundError(f"{model_name} repository not found: {repository}")
    commit = _git_output(repository, "rev-parse", "--verify", "HEAD^{commit}")
    expected = str(expected_commit).strip().lower()
    if commit.lower() != expected:
        raise ValueError(
            f"{model_name} repository commit {commit} does not match pinned "
            f"commit {expected}"
        )
    tracked_status = _git_output(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--ignore-submodules=none",
    )
    if tracked_status:
        changed = ", ".join(
            line.strip() for line in tracked_status.splitlines()[:5]
        )
        if len(tracked_status.splitlines()) > 5:
            changed += ", ..."
        raise ValueError(
            f"{model_name} repository has tracked modifications at pinned commit "
            f"{expected}: {changed}"
        )
    return commit.lower()
