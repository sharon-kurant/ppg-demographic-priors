from __future__ import annotations

from math import gcd

import mne
import numpy as np
from scipy.signal import cheby2, filtfilt, resample_poly

from ppg_bp_incremental.models.encoders.anyppg import (
    ANYPPG_ZSCORE_EPSILON,
    preprocess_anyppg,
)
from ppg_bp_incremental.models.encoders.papagei import preprocess_papagei
from ppg_bp_incremental.models.encoders.pulseppg import preprocess_pulseppg


def _released_ppgbp_reference(
    signal: np.ndarray,
    source_rate: int,
    target_rate: int,
    target_samples: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Independent transcription of the pinned PaPaGei/Pulse-PPG notebooks."""

    source = np.asarray(signal, dtype=np.float32)
    normalized = (source - source.mean(dtype=np.float32)) / (
        source.std(dtype=np.float32) + 1e-7
    )
    b, a = cheby2(4, 20, [0.5, 12], btype="bandpass", fs=source_rate)
    filtered = filtfilt(b, a, normalized)
    if source_rate >= 75:
        window = round(source_rate * 50 / 1000)
        smoothed = filtfilt(np.ones(window) / window, 1, filtered)
    else:
        smoothed = filtered
    divisor = gcd(source_rate, target_rate)
    resampled = resample_poly(
        smoothed,
        target_rate // divisor,
        source_rate // divisor,
    )
    missing = target_samples - len(resampled)
    padded = np.pad(resampled, (missing // 2, missing - missing // 2))
    return [source, normalized, filtered, smoothed, resampled], padded.astype(
        np.float32
    )


def test_papagei_matches_released_preprocessing_numerically_step_for_step():
    rate = 1000
    time = np.arange(2100) / rate
    signal = (
        0.4
        + 0.8 * np.sin(2 * np.pi * 1.1 * time)
        + 0.15 * np.sin(2 * np.pi * 5.3 * time)
    )
    # Pinned example_papagei.ipynb applies ``values.squeeze()[:-1]`` before
    # torch_ecg z-normalization, filtering, resampling, and padding.
    steps, expected = _released_ppgbp_reference(signal[:-1], rate, 125, 1250)
    observed = preprocess_papagei(signal, rate, dataset="PPG-BP")

    assert [len(step) for step in steps] == [2099, 2099, 2099, 2099, 263]
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)
    assert np.count_nonzero(observed[:493]) == 0
    assert np.count_nonzero(observed[-494:]) == 0


def test_pulseppg_matches_released_preprocessing_numerically_step_for_step():
    rate = 1000
    time = np.arange(2100) / rate
    signal = (
        0.7
        + np.sin(2 * np.pi * 1.3 * time)
        + 0.1 * np.sin(2 * np.pi * 7.0 * time)
    )
    # Pinned PPGBP.py line 90 applies ``[:-1]`` before normalization.
    steps, expected = _released_ppgbp_reference(signal[:-1], rate, 50, 500)
    observed = preprocess_pulseppg(signal, rate, dataset="PPG-BP")

    assert [len(step) for step in steps] == [2099, 2099, 2099, 2099, 105]
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)
    assert np.count_nonzero(observed[:197]) == 0
    assert np.count_nonzero(observed[-198:]) == 0


def test_released_pyppg_rule_skips_smoothing_for_30_hz_but_ppg():
    rate = 30
    time = np.arange(300) / rate
    signal = np.sin(2 * np.pi * 1.2 * time) + 0.15 * np.sin(
        2 * np.pi * 8.0 * time
    )
    steps, papagei_expected = _released_ppgbp_reference(
        signal, rate, 125, 1250
    )
    _, pulse_expected = _released_ppgbp_reference(signal, rate, 50, 500)

    # The "smoothed" reference step is exactly the filter output below 75 Hz,
    # matching the conditional in the pinned upstream Preprocess.get_signals.
    np.testing.assert_array_equal(steps[2], steps[3])
    np.testing.assert_allclose(
        preprocess_papagei(signal, rate),
        papagei_expected,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        preprocess_pulseppg(signal, rate),
        pulse_expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_released_pyppg_smoothing_threshold_is_inclusive_at_75_hz():
    rate = 75
    time = np.arange(750) / rate
    signal = np.sin(2 * np.pi * 1.2 * time) + 0.2 * np.sin(
        2 * np.pi * 10.0 * time
    )
    steps, expected = _released_ppgbp_reference(signal, rate, 50, 500)

    # Pinned pyPPG uses ``if s.fs >= 75``.  At exactly 75 Hz the 50 ms window
    # rounds to four samples, and therefore changes the band-pass result.
    assert not np.array_equal(steps[2], steps[3])
    np.testing.assert_allclose(
        preprocess_pulseppg(signal, rate, dataset="PulseDB-Vital"),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_anyppg_matches_pinned_mne_filter_resample_normalize_order():
    rate = 1000
    time = np.arange(2100) / rate
    signal = (
        2.0
        + np.sin(2 * np.pi * 1.2 * time)
        + 0.2 * np.sin(2 * np.pi * 10.0 * time)
    )
    filtered = mne.filter.filter_data(
        signal[np.newaxis, :].astype(float),
        sfreq=rate,
        l_freq=0.5,
        h_freq=8.0,
        method="iir",
        iir_params={"order": 3, "ftype": "butter", "output": "ba"},
        verbose=False,
    )[0].astype(np.float32)
    resampled = resample_poly(filtered, 1, 8)
    normalized = (resampled - resampled.mean()) / (
        resampled.std() + ANYPPG_ZSCORE_EPSILON
    )
    expected = np.pad(normalized, (493, 494)).astype(np.float32)

    observed = preprocess_anyppg(signal, rate)

    assert [len(signal), len(filtered), len(resampled), len(normalized)] == [
        2100,
        2100,
        263,
        263,
    ]
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)
