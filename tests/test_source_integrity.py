from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import subprocess

import pytest

from ppg_bp_incremental.evaluation.model_audit import (
    validate_fingerprint_against_contract,
)
from ppg_bp_incremental.models.contracts import CONTRACT_VERSION, MODEL_CONTRACTS
from ppg_bp_incremental.models.encoders import papagei, pulseppg
from ppg_bp_incremental.models.encoders.base import EncoderFingerprint
from ppg_bp_incremental.models.encoders.source_integrity import (
    verify_checkpoint_sha256,
    verify_git_checkout,
)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments],
        text=True,
    ).strip()


def _clean_git_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "upstream"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    _git(repository, "config", "user.name", "Source Integrity Test")
    _git(repository, "config", "user.email", "source-integrity@example.invalid")
    tracked = repository / "model.py"
    tracked.write_text("MODEL_VERSION = 1\n", encoding="utf-8")
    _git(repository, "add", "model.py")
    _git(repository, "commit", "-q", "-m", "fixture")
    return repository, _git(repository, "rev-parse", "HEAD")


def _contract_fingerprint(model_key: str) -> EncoderFingerprint:
    contract = MODEL_CONTRACTS[model_key]
    return EncoderFingerprint(
        contract_version=CONTRACT_VERSION,
        model_name=model_key,
        model_version="unit-test",
        checkpoint_sha256=contract.checkpoint_sha256,
        repository_commit=contract.source_commit,
        preprocessing={},
        input_sampling_rate_hz=contract.input_sampling_rate_hz,
        input_samples=int(
            contract.input_sampling_rate_hz * contract.input_duration_seconds
        ),
        embedding_dimension=contract.selected_dimension,
    )


def test_git_authentication_rejects_tracked_modifications_but_not_untracked_files(
    tmp_path,
):
    repository, commit = _clean_git_repository(tmp_path)
    (repository / "downloaded_checkpoint.pt").write_bytes(b"untracked checkpoint")

    assert verify_git_checkout(
        repository,
        expected_commit=commit,
        model_name="Fixture",
    ) == commit

    (repository / "model.py").write_text("MODEL_VERSION = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked modifications"):
        verify_git_checkout(
            repository,
            expected_commit=commit,
            model_name="Fixture",
        )


def test_git_authentication_rejects_wrong_commit_without_model_dependencies(tmp_path):
    repository, _ = _clean_git_repository(tmp_path)
    with pytest.raises(ValueError, match="does not match pinned commit"):
        verify_git_checkout(
            repository,
            expected_commit="0" * 40,
            model_name="Fixture",
        )


def test_checkpoint_authentication_rejects_wrong_bytes(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not the declared checkpoint")
    observed = sha256(checkpoint.read_bytes()).hexdigest()

    assert verify_checkpoint_sha256(
        checkpoint,
        expected_sha256=observed,
        model_name="Fixture",
    ) == observed
    with pytest.raises(ValueError, match="does not match pinned SHA-256"):
        verify_checkpoint_sha256(
            checkpoint,
            expected_sha256="f" * 64,
            model_name="Fixture",
        )


@pytest.mark.parametrize("variant", ["p", "s"])
def test_papagei_constructor_rejects_non_pinned_checkpoint_before_loading_model(
    tmp_path,
    monkeypatch,
    variant,
):
    checkpoint = tmp_path / f"papagei_{variant}.pt"
    checkpoint.write_bytes(b"fake PaPaGei checkpoint")
    monkeypatch.setattr(
        papagei,
        "verify_git_checkout",
        lambda *args, **kwargs: papagei.PAPAGEI_REPOSITORY_COMMIT,
    )

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        papagei.PapageiEncoder(variant, tmp_path, checkpoint, device="cpu")


def test_pulseppg_constructor_rejects_non_pinned_checkpoint_before_loading_model(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "pulseppg.pkl"
    checkpoint.write_bytes(b"fake Pulse-PPG checkpoint")
    monkeypatch.setattr(
        pulseppg,
        "verify_git_checkout",
        lambda *args, **kwargs: pulseppg.PULSEPPG_REPOSITORY_COMMIT,
    )

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        pulseppg.PulsePPGEncoder(tmp_path, checkpoint, device="cpu")


@pytest.mark.parametrize("model_key", sorted(MODEL_CONTRACTS))
def test_model_audit_accepts_only_contract_source_identity(model_key):
    fingerprint = _contract_fingerprint(model_key)
    validate_fingerprint_against_contract(model_key, fingerprint)

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        validate_fingerprint_against_contract(
            model_key,
            replace(fingerprint, checkpoint_sha256="0" * 64),
        )
    with pytest.raises(ValueError, match="repository_commit"):
        validate_fingerprint_against_contract(
            model_key,
            replace(fingerprint, repository_commit="0" * 40),
        )
