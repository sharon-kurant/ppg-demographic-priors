"""Executable source/checkpoint audit for the four locked encoders."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import numpy as np

from ppg_bp_incremental.models.registry import MODEL_SPECS, create_encoder
from ppg_bp_incremental.models.contracts import (
    CONTRACT_VERSION,
    MODEL_CONTRACTS,
    contract_registry_payload,
)
from ppg_bp_incremental.models.encoders.base import EncoderFingerprint


EXPECTED_NATIVE_SHAPES = {
    "papagei_p": {"projected_embedding": (2, 512), "pooled_embedding": (2, 512)},
    "papagei_s": {
        "projected_embedding": (2, 512),
        "ipa": (2, 1),
        "sqi": (2, 1),
        "pooled_embedding": (2, 512),
    },
    "pulseppg": {"embedding": (2, 512)},
    "anyppg": {"embedding": (2, 512)},
}


def validate_fingerprint_against_contract(
    model_key: str,
    fingerprint: EncoderFingerprint,
) -> None:
    """Reject an executed encoder whose identity differs from its registry.

    The wrapper authenticates source and checkpoint bytes at construction.
    This independent audit-layer comparison prevents a wrapper or a saved
    fingerprint from silently drifting away from ``MODEL_CONTRACTS``.
    """

    if model_key not in MODEL_CONTRACTS:
        raise ValueError(f"No source contract is registered for {model_key!r}")
    contract = MODEL_CONTRACTS[model_key]
    mismatches: list[str] = []
    if fingerprint.contract_version != CONTRACT_VERSION:
        mismatches.append(
            "contract_version "
            f"{fingerprint.contract_version!r} != {CONTRACT_VERSION!r}"
        )
    if fingerprint.model_name != model_key:
        mismatches.append(
            f"model_name {fingerprint.model_name!r} != {model_key!r}"
        )
    if fingerprint.repository_commit.lower() != contract.source_commit.lower():
        mismatches.append(
            "repository_commit "
            f"{fingerprint.repository_commit!r} != {contract.source_commit!r}"
        )
    if fingerprint.checkpoint_sha256.lower() != contract.checkpoint_sha256.lower():
        mismatches.append(
            "checkpoint_sha256 "
            f"{fingerprint.checkpoint_sha256!r} != {contract.checkpoint_sha256!r}"
        )
    if mismatches:
        raise ValueError(
            f"{model_key} encoder fingerprint violates MODEL_CONTRACTS: "
            + "; ".join(mismatches)
        )


def audit_models(output_json: str | Path, device: str = "cpu") -> dict:
    records = []
    for model_key, spec in MODEL_SPECS.items():
        contract = MODEL_CONTRACTS[model_key]
        if spec.repository_commit.lower() != contract.source_commit.lower():
            raise ValueError(
                f"{model_key} construction registry commit differs from MODEL_CONTRACTS"
            )
        if spec.parameters != contract.parameter_count:
            raise ValueError(
                f"{model_key} construction registry parameter count differs from "
                "MODEL_CONTRACTS"
            )
        encoder = create_encoder(model_key, device=device)
        fingerprint = encoder.fingerprint()
        validate_fingerprint_against_contract(model_key, fingerprint)
        parameter_count = sum(parameter.numel() for parameter in encoder.model.parameters())
        if parameter_count != contract.parameter_count:
            raise ValueError(
                f"{model_key} has {parameter_count:,} parameters; expected "
                f"{contract.parameter_count:,}"
            )
        samples = fingerprint.input_samples
        batch = np.stack(
            [np.sin(np.linspace(0, 20, samples)), np.cos(np.linspace(0, 20, samples))]
        ).astype(np.float32)
        result = encoder.encode_checked(batch)
        shapes = result.component_shapes()
        if shapes != EXPECTED_NATIVE_SHAPES[model_key]:
            raise ValueError(f"{model_key} native shapes changed: {shapes}")
        records.append(
            {
                "model": model_key,
                "display_name": spec.display_name,
                "parameters": parameter_count,
                "fingerprint": asdict(fingerprint),
                "source_authentication": {
                    "checkpoint_sha256_matches_contract": True,
                    "repository_commit_matches_contract": True,
                    "tracked_repository_state": "clean_at_encoder_construction",
                },
                "native_output_shapes": {name: list(shape) for name, shape in shapes.items()},
                "native_output_semantics": dict(result.component_semantics),
                "selected_embedding_shape": list(result.embedding.shape),
                "source_contract": MODEL_CONTRACTS[model_key].as_dict(),
            }
        )
    packages = {}
    for package in ("numpy", "scipy", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    payload = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_version": CONTRACT_VERSION,
        "contract_registry": contract_registry_payload(),
        "software_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        },
        "target_contract": {
            "targets": ["sbp", "dbp"],
            "active_estimator": "separate standardized-feature Ridge per target",
            "estimator_output_scale": "raw_mmhg",
            "decoder": {"type": "identity"},
            "note": (
                "This BP adapter contract applies only after the selected embedding; "
                "it does not assign BP units to native encoder outputs."
            ),
        },
        "models": records,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
