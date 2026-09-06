from __future__ import annotations

from pathlib import Path

import pytest

from ppg_bp_incremental.evaluation.model_audit import (
    EXPECTED_NATIVE_SHAPES,
    audit_models,
)
from ppg_bp_incremental.models.registry import MODEL_SPECS


@pytest.mark.integration
def test_real_checkpoints_match_locked_native_contracts(tmp_path):
    missing = [str(spec.checkpoint) for spec in MODEL_SPECS.values() if not spec.checkpoint.exists()]
    if missing:
        pytest.skip(f"Real checkpoint files are not installed: {missing}")
    payload = audit_models(tmp_path / "models.json", device="cpu")
    observed = {
        record["model"]: {
            name: tuple(shape) for name, shape in record["native_output_shapes"].items()
        }
        for record in payload["models"]
    }

    assert observed == EXPECTED_NATIVE_SHAPES
    assert all(record["fingerprint"]["checkpoint_sha256"] for record in payload["models"])
