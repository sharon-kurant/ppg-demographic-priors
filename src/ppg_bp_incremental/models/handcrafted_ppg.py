"""Deterministic statistical and pulse-morphology PPG features."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.signal import cheby2, find_peaks, peak_widths, sosfiltfilt, welch
from scipy.stats import kurtosis, skew

from ppg_bp_incremental.models.encoders.base import (
    EncoderFingerprint,
    NativeEncoderOutput,
    PPGEncoder,
)
from ppg_bp_incremental.models.contracts import CONTRACT_VERSION


HANDCRAFTED_FEATURE_NAMES = (
    "waveform_heart_rate_bpm",
    "spectral_peak_bpm",
    "raw_mean",
    "raw_std",
    "raw_range",
    "raw_iqr",
    "raw_skew",
    "raw_kurtosis",
    "filtered_std",
    "filtered_range",
    "filtered_iqr",
    "filtered_skew",
    "filtered_kurtosis",
    "derivative_std",
    "derivative_abs_mean",
    "second_derivative_std",
    "spectral_entropy",
    "bandpower_0p5_2_hz",
    "bandpower_2_4_hz",
    "bandpower_4_8_hz",
    "bandpower_low_high_ratio",
    "peak_count_per_second",
    "interval_mean_s",
    "interval_std_s",
    "interval_rmssd_s",
    "interval_cv",
    "peak_amplitude_mean",
    "peak_amplitude_std",
    "peak_prominence_mean",
    "peak_prominence_std",
    "peak_width_mean_s",
    "peak_width_std_s",
    "rise_time_mean_s",
    "rise_time_std_s",
    "decay_time_mean_s",
    "decay_time_std_s",
    "pulse_amplitude_mean",
    "pulse_amplitude_std",
)

def _mean(values: np.ndarray | list[float]) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)) if values.size else 0.0


def _sample_std(values: np.ndarray | list[float]) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if values.size >= 2 else 0.0


def _bandpower(frequencies: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    selected = (frequencies >= low) & (frequencies < high)
    if np.count_nonzero(selected) < 2:
        return 0.0
    # NumPy 1.26 calls the same trapezoidal integration routine ``trapz``;
    # NumPy 2.x exposes the clearer ``trapezoid`` name.
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0
        integrate = np.trapz
    return float(integrate(power[selected], frequencies[selected]))


def extract_handcrafted_features(
    signal: np.ndarray,
    sampling_rate_hz: int,
) -> dict[str, float]:
    """Extract fixed, label-independent features from one PPG recording."""
    signal = np.asarray(signal, dtype=np.float64).squeeze()
    if signal.ndim != 1 or len(signal) < max(32, sampling_rate_hz):
        raise ValueError("Handcrafted features require at least one second of PPG")
    if sampling_rate_hz <= 24:
        raise ValueError("Sampling rate must exceed twice the 12 Hz cutoff")
    if not np.isfinite(signal).all():
        raise ValueError("Handcrafted features received non-finite samples")
    raw_std = float(np.std(signal))
    if raw_std <= 0:
        raise ValueError("Handcrafted features require a non-constant signal")

    normalized = (signal - np.mean(signal)) / (raw_std + 1e-7)
    filter_sos = cheby2(
        4,
        20,
        [0.5, 12],
        btype="bandpass",
        fs=sampling_rate_hz,
        output="sos",
    )
    filtered = sosfiltfilt(filter_sos, normalized)
    derivative = np.diff(filtered) * sampling_rate_hz
    second_derivative = np.diff(filtered, n=2) * sampling_rate_hz**2

    # Use the complete short recording for the finest attainable pulse-rate
    # resolution. PPG-BP recordings are only about two seconds long.
    nperseg = len(filtered)
    frequencies, power = welch(
        filtered,
        fs=sampling_rate_hz,
        nperseg=nperseg,
        detrend="constant",
    )
    pulse_band = (frequencies >= 0.5) & (frequencies <= 4.0)
    if np.any(pulse_band):
        pulse_frequencies = frequencies[pulse_band]
        pulse_power = power[pulse_band]
        spectral_peak_hz = float(pulse_frequencies[np.argmax(pulse_power)])
    else:
        spectral_peak_hz = 0.0

    entropy_band = (frequencies >= 0.5) & (frequencies <= 12.0)
    entropy_power = power[entropy_band]
    entropy_total = float(np.sum(entropy_power))
    if entropy_total > 0 and len(entropy_power) > 1:
        probabilities = entropy_power / entropy_total
        spectral_entropy = float(
            -np.sum(probabilities * np.log(probabilities + 1e-12))
            / np.log(len(probabilities))
        )
    else:
        spectral_entropy = 0.0

    low_power = _bandpower(frequencies, power, 0.5, 2.0)
    mid_power = _bandpower(frequencies, power, 2.0, 4.0)
    high_power = _bandpower(frequencies, power, 4.0, 8.0)

    peaks, properties = find_peaks(
        filtered,
        distance=max(1, round(0.3 * sampling_rate_hz)),
        prominence=max(0.1, 0.2 * float(np.std(filtered))),
    )
    intervals = np.diff(peaks) / sampling_rate_hz
    interval_bpm = 60.0 / float(np.median(intervals)) if intervals.size else 0.0
    if not 30 <= interval_bpm <= 200:
        interval_bpm = spectral_peak_hz * 60.0
    interval_rmssd = (
        float(np.sqrt(np.mean(np.diff(intervals) ** 2)))
        if intervals.size >= 2
        else 0.0
    )
    interval_mean = _mean(intervals)
    interval_std = _sample_std(intervals)
    interval_cv = interval_std / interval_mean if interval_mean > 0 else 0.0

    prominences = np.asarray(properties.get("prominences", []), dtype=float)
    if peaks.size:
        widths = peak_widths(filtered, peaks, rel_height=0.5)[0] / sampling_rate_hz
        peak_amplitudes = filtered[peaks]
    else:
        widths = np.array([], dtype=float)
        peak_amplitudes = np.array([], dtype=float)

    troughs, _ = find_peaks(
        -filtered,
        distance=max(1, round(0.2 * sampling_rate_hz)),
    )
    rise_times: list[float] = []
    decay_times: list[float] = []
    pulse_amplitudes: list[float] = []
    for peak in peaks:
        previous = troughs[troughs < peak]
        following = troughs[troughs > peak]
        if not previous.size or not following.size:
            continue
        left = int(previous[-1])
        right = int(following[0])
        rise_times.append((peak - left) / sampling_rate_hz)
        decay_times.append((right - peak) / sampling_rate_hz)
        pulse_amplitudes.append(
            float(filtered[peak] - 0.5 * (filtered[left] + filtered[right]))
        )

    raw_q25, raw_q75 = np.quantile(signal, [0.25, 0.75])
    filtered_q25, filtered_q75 = np.quantile(filtered, [0.25, 0.75])
    features = {
        "waveform_heart_rate_bpm": interval_bpm,
        "spectral_peak_bpm": spectral_peak_hz * 60.0,
        "raw_mean": float(np.mean(signal)),
        "raw_std": raw_std,
        "raw_range": float(np.ptp(signal)),
        "raw_iqr": float(raw_q75 - raw_q25),
        "raw_skew": float(skew(signal, bias=False)),
        "raw_kurtosis": float(kurtosis(signal, fisher=True, bias=False)),
        "filtered_std": float(np.std(filtered)),
        "filtered_range": float(np.ptp(filtered)),
        "filtered_iqr": float(filtered_q75 - filtered_q25),
        "filtered_skew": float(skew(filtered, bias=False)),
        "filtered_kurtosis": float(kurtosis(filtered, fisher=True, bias=False)),
        "derivative_std": float(np.std(derivative)),
        "derivative_abs_mean": float(np.mean(np.abs(derivative))),
        "second_derivative_std": float(np.std(second_derivative)),
        "spectral_entropy": spectral_entropy,
        "bandpower_0p5_2_hz": low_power,
        "bandpower_2_4_hz": mid_power,
        "bandpower_4_8_hz": high_power,
        "bandpower_low_high_ratio": low_power / (mid_power + high_power + 1e-12),
        "peak_count_per_second": len(peaks) / (len(signal) / sampling_rate_hz),
        "interval_mean_s": interval_mean,
        "interval_std_s": interval_std,
        "interval_rmssd_s": interval_rmssd,
        "interval_cv": interval_cv,
        "peak_amplitude_mean": _mean(peak_amplitudes),
        "peak_amplitude_std": _sample_std(peak_amplitudes),
        "peak_prominence_mean": _mean(prominences),
        "peak_prominence_std": _sample_std(prominences),
        "peak_width_mean_s": _mean(widths),
        "peak_width_std_s": _sample_std(widths),
        "rise_time_mean_s": _mean(rise_times),
        "rise_time_std_s": _sample_std(rise_times),
        "decay_time_mean_s": _mean(decay_times),
        "decay_time_std_s": _sample_std(decay_times),
        "pulse_amplitude_mean": _mean(pulse_amplitudes),
        "pulse_amplitude_std": _sample_std(pulse_amplitudes),
    }
    values = np.asarray([features[name] for name in HANDCRAFTED_FEATURE_NAMES])
    if not np.isfinite(values).all():
        invalid = [
            name
            for name in HANDCRAFTED_FEATURE_NAMES
            if not np.isfinite(features[name])
        ]
        raise ValueError(f"Handcrafted feature extraction produced invalid {invalid}")
    return features


class HandcraftedFeatureEncoder(PPGEncoder):
    """Expose deterministic features through the shared representation cache."""

    def __init__(self) -> None:
        self.feature_names = HANDCRAFTED_FEATURE_NAMES

    def preprocess(self, signal: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
        features = extract_handcrafted_features(signal, sampling_rate_hz)
        return np.asarray([features[name] for name in self.feature_names], dtype=np.float32)

    def preprocessing_shape_trace(
        self, signal: np.ndarray, sampling_rate_hz: int
    ) -> list[dict[str, object]]:
        samples = int(np.asarray(signal).size)
        return [
            {"name": "raw_unpadded_waveform", "shape": [samples], "sampling_rate_hz": sampling_rate_hz},
            {"name": "bandpass_filtered", "shape": [samples], "sampling_rate_hz": sampling_rate_hz},
            {"name": "first_derivative", "shape": [samples - 1], "sampling_rate_hz": sampling_rate_hz},
            {"name": "second_derivative", "shape": [samples - 2], "sampling_rate_hz": sampling_rate_hz},
            {"name": "handcrafted_feature_vector", "shape": [len(self.feature_names)], "sampling_rate_hz": 0},
        ]

    def encode(self, batch: np.ndarray) -> NativeEncoderOutput:
        values = np.asarray(batch, dtype=np.float32)
        return NativeEncoderOutput(
            embedding=values,
            components={"handcrafted_features": values},
            component_semantics={
                "handcrafted_features": "fixed label-independent PPG feature vector"
            },
        )

    def fingerprint(self) -> EncoderFingerprint:
        implementation = Path(__file__).read_bytes()
        return EncoderFingerprint(
            contract_version=CONTRACT_VERSION,
            model_name="handcrafted_ppg",
            model_version="deterministic-v1",
            checkpoint_sha256=sha256(b"no-checkpoint").hexdigest(),
            repository_commit="not-applicable",
            preprocessing={
                "name": "handcrafted_morphology_and_statistics_v1",
                "implementation_sha256": sha256(implementation).hexdigest(),
                "feature_names": list(self.feature_names),
                "filter": "cheby2_bandpass_0.5_12_hz",
                "normalization": "per_recording_zscore_for_morphology",
                "raw_amplitude_features": True,
                "missing_morphology_value": 0.0,
            },
            input_sampling_rate_hz=0,
            input_samples=len(self.feature_names),
            embedding_dimension=len(self.feature_names),
        )
