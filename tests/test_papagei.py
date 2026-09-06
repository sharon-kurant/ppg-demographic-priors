from __future__ import annotations

import numpy as np
import pytest

from ppg_bp_incremental.models.encoders.papagei import (
    preprocess_papagei,
    preprocess_ppgbp_official,
)
from ppg_bp_incremental.models.encoders.pulseppg import (
    PULSEPPG_INPUT_SAMPLES,
    PULSEPPG_TARGET_RATE_HZ,
    PulsePPGEncoder,
    preprocess_pulseppg,
    pulseppg_preprocessing_shape_trace,
)


def test_papagei_preprocessing_matches_official_length_and_padding():
    time = np.arange(2100) / 1000
    signal = np.sin(2 * np.pi * 1.2 * time) + 0.2 * np.sin(2 * np.pi * 4 * time)
    processed = preprocess_papagei(signal, 1000)

    assert processed.shape == (1250,)
    assert processed.dtype == np.float32
    assert np.isfinite(processed).all()
    assert np.all(processed[:493] == 0)
    assert np.all(processed[-494:] == 0)
    assert np.any(processed[493:-494] != 0)


def test_papagei_preprocessing_rejects_implicit_truncation():
    signal = np.sin(np.arange(10001) / 20)
    with pytest.raises(ValueError, match="segment the signal explicitly"):
        preprocess_papagei(signal, 1000)


def test_pulseppg_preprocessing_uses_official_50_hz_length():
    time = np.arange(2100) / 1000
    signal = np.sin(2 * np.pi * 1.2 * time)
    processed = preprocess_pulseppg(signal, 1000, dataset="PPG-BP")

    assert processed.shape == (500,)
    assert processed.dtype == np.float32
    assert np.all(processed[:197] == 0)
    assert np.all(processed[-198:] == 0)
    assert np.any(processed[197:-198] != 0)


def test_pulseppg_ppgbp_drops_final_raw_sample_before_all_other_operations():
    time = np.arange(2100) / 1000
    signal = (
        np.sin(2 * np.pi * 1.2 * time)
        + 0.1 * np.cos(2 * np.pi * 3.7 * time)
    ).astype(np.float32)
    signal[-1] = 25.0

    released = preprocess_pulseppg(signal, 1000, dataset="PPG-BP")
    expected = preprocess_ppgbp_official(
        signal[:-1],
        1000,
        PULSEPPG_TARGET_RATE_HZ,
        PULSEPPG_INPUT_SAMPLES,
    )
    untrimmed_adaptation = preprocess_pulseppg(
        signal,
        1000,
        dataset="PulseDB-Vital",
    )

    np.testing.assert_array_equal(released, expected)
    assert not np.allclose(released, untrimmed_adaptation)


def test_pulseppg_final_sample_trim_is_ppgbp_only_and_visible_in_trace():
    signal = np.linspace(-1.0, 1.0, 2100, dtype=np.float32)
    ppgbp_trace = pulseppg_preprocessing_shape_trace(
        signal,
        1000,
        dataset="PPG-BP",
    )
    pulsedb_trace = pulseppg_preprocessing_shape_trace(
        signal,
        1000,
        dataset="PulseDB-Vital",
    )

    assert [step["name"] for step in ppgbp_trace[:3]] == [
        "raw_waveform",
        "released_ppgbp_final_sample_trim",
        "z_normalized",
    ]
    assert ppgbp_trace[0]["shape"] == [2100]
    assert ppgbp_trace[1] == {
        "name": "released_ppgbp_final_sample_trim",
        "shape": [2099],
        "sampling_rate_hz": 1000,
        "samples_removed_from_end": 1,
    }
    assert "released_ppgbp_final_sample_trim" not in {
        step["name"] for step in pulsedb_trace
    }


def test_pulseppg_ppgbp_trim_has_a_dataset_specific_cache_fingerprint():
    encoder = PulsePPGEncoder.__new__(PulsePPGEncoder)
    encoder.repository_commit = "716eaf9cf966e8f76436f2263872ef38b1f90166"
    encoder.checkpoint_sha256 = "4" * 64

    generic = encoder.fingerprint()
    ppgbp = encoder.fingerprint_for_dataset("PPG-BP")
    pulsedb = encoder.fingerprint_for_dataset("PulseDB-Vital")

    assert pulsedb == generic
    assert ppgbp.digest() != generic.digest()
    assert ppgbp.model_version.endswith("-ppgbp-trim-v1")
    assert ppgbp.preprocessing["ppgbp_final_raw_sample_trim"] == {
        "applied": True,
        "samples_removed_from_end": 1,
        "operation_order": "before_zscore_filter_resample_and_padding",
        "source": "pulseppg/data/process/PPGBP.py:90",
    }


def test_removed_reflection_padding_argument_is_rejected():
    time = np.arange(2100) / 1000
    signal = np.sin(2 * np.pi * 1.2 * time)
    with pytest.raises(TypeError, match="padding_policy"):
        preprocess_papagei(signal, 1000, padding_policy="reflect")
