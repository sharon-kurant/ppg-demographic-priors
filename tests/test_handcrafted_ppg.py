from __future__ import annotations

import numpy as np

from ppg_bp_incremental.models.handcrafted_ppg import (
    HANDCRAFTED_FEATURE_NAMES,
    HandcraftedFeatureEncoder,
    extract_handcrafted_features,
)


def _synthetic_pulse(rate_hz: float = 1.2, sampling_rate_hz: int = 125) -> np.ndarray:
    time = np.arange(10 * sampling_rate_hz) / sampling_rate_hz
    fundamental = np.sin(2 * np.pi * rate_hz * time)
    harmonic = 0.3 * np.sin(4 * np.pi * rate_hz * time - 0.4)
    return 2.5 + 1.7 * (fundamental + harmonic)


def test_handcrafted_features_recover_waveform_heart_rate():
    features = extract_handcrafted_features(_synthetic_pulse(), 125)
    values = np.asarray([features[name] for name in HANDCRAFTED_FEATURE_NAMES])

    assert values.shape == (len(HANDCRAFTED_FEATURE_NAMES),)
    assert np.isfinite(values).all()
    assert 68 <= features["waveform_heart_rate_bpm"] <= 76
    assert 68 <= features["spectral_peak_bpm"] <= 76


def test_handcrafted_encoder_follows_shared_contract():
    signal = _synthetic_pulse()
    encoder = HandcraftedFeatureEncoder()

    features = encoder.preprocess_checked(signal, 125)
    output = encoder.encode_checked(np.stack([features, features]))

    assert output.embedding.shape == (2, len(HANDCRAFTED_FEATURE_NAMES))
    assert output.component_shapes() == {
        "handcrafted_features": (2, len(HANDCRAFTED_FEATURE_NAMES))
    }
