from __future__ import annotations

import numpy as np
import pytest

from ppg_bp_incremental.models.contracts import (
    CONTRACT_VERSION,
    MODEL_CONTRACTS,
    contract_registry_payload,
    decode_bp_output,
    source_fidelity,
)


def test_versioned_registry_locks_every_native_component_and_bp_adapter():
    assert CONTRACT_VERSION == "source-faithful-v2"
    assert set(MODEL_CONTRACTS) == {
        "papagei_p",
        "papagei_s",
        "pulseppg",
        "anyppg",
    }
    expected_components = {
        "papagei_p": ["projected_embedding", "pooled_embedding"],
        "papagei_s": [
            "projected_embedding",
            "ipa",
            "sqi",
            "pooled_embedding",
        ],
        "pulseppg": ["embedding"],
        "anyppg": ["embedding"],
    }
    for model, contract in MODEL_CONTRACTS.items():
        assert [component.name for component in contract.native_outputs] == (
            expected_components[model]
        )
        assert contract.selected_dimension == 512
        assert contract.bp_target_scale == "raw_mmhg"
        assert contract.bp_decoder == "identity"
        assert contract.direct_encoder_demographics == ()
        assert len(contract.source_commit) == 40
        assert len(contract.checkpoint_sha256) == 64

    payload = contract_registry_payload()
    assert payload["contract_version"] == CONTRACT_VERSION
    assert set(payload["models"]) == set(MODEL_CONTRACTS)
    assert MODEL_CONTRACTS["pulseppg"].preprocessing_operations[0] == (
        "for PPG-BP only, discard the final raw sample as in released PPGBP.py"
    )


def test_bp_task_decoder_never_infers_units_from_shape():
    raw = decode_bp_output(
        np.array([[120.0], [135.0]]),
        target="sbp",
        output_scale="raw_mmhg",
        decoder_metadata={"type": "identity"},
    )
    np.testing.assert_array_equal(raw.estimator_output, [120.0, 135.0])
    np.testing.assert_array_equal(raw.mmhg_prediction, [120.0, 135.0])

    standardized = decode_bp_output(
        np.array([-1.0, 0.5]),
        target="dbp",
        output_scale="z_score",
        decoder_metadata={
            "type": "affine_z_score",
            "mean_mmhg": 80.0,
            "std_mmhg": 10.0,
        },
    )
    np.testing.assert_array_equal(standardized.mmhg_prediction, [70.0, 85.0])

    with pytest.raises(ValueError, match="identity decoder"):
        decode_bp_output(
            np.array([1.0]),
            target="sbp",
            output_scale="raw_mmhg",
            decoder_metadata={"type": "affine_z_score"},
        )
    with pytest.raises((KeyError, ValueError)):
        decode_bp_output(
            np.array([1.0]),
            target="sbp",
            output_scale="z_score",
            decoder_metadata={"type": "affine_z_score"},
        )


def test_source_fidelity_marks_exact_adapted_and_pretraining_overlap_paths():
    assert source_fidelity("papagei_p", "PPG-BP") == "released_exact"
    assert source_fidelity("pulseppg", "PPG-BP") == "released_exact"
    assert source_fidelity("anyppg", "PulseDB-Vital") == "paper_specified"
    assert (
        source_fidelity("anyppg", "BUT PPG")
        == "source_consistent_adaptation"
    )
