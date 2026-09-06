from __future__ import annotations

import numpy as np
import pytest

from ppg_bp_incremental.models.encoders.anyppg import preprocess_anyppg


def test_anyppg_preprocessing_resamples_normalizes_and_pads_short_signal():
    time = np.arange(2100) / 1000
    signal = 4.0 + 2.5 * np.sin(2 * np.pi * 1.2 * time)
    processed = preprocess_anyppg(signal, 1000)

    assert processed.shape == (1250,)
    assert processed.dtype == np.float32
    assert np.isfinite(processed).all()
    assert np.all(processed[:493] == 0)
    assert np.all(processed[-494:] == 0)
    active = processed[493:-494]
    assert abs(float(active.mean())) < 1e-5
    assert float(active.std()) == pytest.approx(1.0, abs=1e-4)


def test_anyppg_preprocessing_keeps_ten_second_duration():
    time = np.arange(1000) / 100
    processed = preprocess_anyppg(np.sin(2 * np.pi * time), 100)

    assert processed.shape == (1250,)
    assert not np.all(processed[:10] == 0)
    assert abs(float(processed.mean())) < 1e-5
    assert float(processed.std()) == pytest.approx(1.0, abs=1e-4)


def test_anyppg_preprocessing_rejects_implicit_truncation():
    signal = np.sin(np.arange(1100) / 20)
    with pytest.raises(ValueError, match="segment the signal explicitly"):
        preprocess_anyppg(signal, 100)
