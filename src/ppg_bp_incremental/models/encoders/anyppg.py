"""Frozen wrapper for the official AnyPPG checkpoint."""

from __future__ import annotations

from math import ceil, gcd
from pathlib import Path
import importlib.util

import mne
import numpy as np
from scipy.signal import resample_poly

from ppg_bp_incremental.models.encoders.base import (
    EncoderFingerprint,
    NativeEncoderOutput,
    PPGEncoder,
)
from ppg_bp_incremental.models.contracts import CONTRACT_VERSION, MODEL_CONTRACTS
from ppg_bp_incremental.models.encoders.source_integrity import (
    load_torch_checkpoint,
    verify_checkpoint_sha256,
    verify_git_checkout,
)


ANYPPG_REPOSITORY_COMMIT = MODEL_CONTRACTS["anyppg"].source_commit
ANYPPG_CHECKPOINT_SHA256 = MODEL_CONTRACTS["anyppg"].checkpoint_sha256
ANYPPG_TARGET_RATE_HZ = 125
ANYPPG_INPUT_SAMPLES = 1250
ANYPPG_EMBEDDING_DIMENSION = 512
ANYPPG_ZSCORE_EPSILON = 1e-5
ANYPPG_PREPROCESSING_POLICY = (
    "source_consistent_filter_resample_zscore_adaptation"
)


def preprocess_anyppg(
    signal: np.ndarray,
    sampling_rate_hz: int,
) -> np.ndarray:
    """Apply the source-consistent AnyPPG adaptation to one 10-second input.

    The third-order Butterworth coefficients and zero-phase filtering match the
    upstream ``mne.filter.filter_data(..., method="iir",
    iir_params={"order": 3, "ftype": "butter", "output": "ba"})`` contract.
    The exact public construction of AnyPPG's downstream BP arrays is not
    released, so this operation sequence is not labeled an exact inference
    pipeline.
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal.size < 2 or not np.isfinite(signal).all():
        raise ValueError("AnyPPG input must contain at least two finite samples")
    if sampling_rate_hz < 1:
        raise ValueError("AnyPPG sampling rate must be positive")
    if sampling_rate_hz <= 16:
        raise ValueError("AnyPPG sampling rate must exceed twice the 8 Hz cutoff")
    signal = mne.filter.filter_data(
        signal[np.newaxis, :].astype(float),
        sfreq=float(sampling_rate_hz),
        l_freq=0.5,
        h_freq=8.0,
        method="iir",
        iir_params={"order": 3, "ftype": "butter", "output": "ba"},
        verbose=False,
    )[0].astype(np.float32)
    divisor = gcd(int(sampling_rate_hz), ANYPPG_TARGET_RATE_HZ)
    resampled = resample_poly(
        signal,
        ANYPPG_TARGET_RATE_HZ // divisor,
        int(sampling_rate_hz) // divisor,
    )
    if len(resampled) > ANYPPG_INPUT_SAMPLES:
        raise ValueError(
            "AnyPPG input exceeds 10 seconds after resampling; segment the signal "
            "explicitly rather than silently truncating it"
        )
    mean = float(np.mean(resampled))
    std = float(np.std(resampled))
    normalized = (resampled - mean) / (std + ANYPPG_ZSCORE_EPSILON)
    missing = ANYPPG_INPUT_SAMPLES - len(normalized)
    left = missing // 2
    right = missing - left
    padded = np.pad(normalized, (left, right), mode="constant")
    return np.asarray(padded, dtype=np.float32)


def _load_network_class(repository_path: Path):
    source = repository_path / "load_anyppg" / "resnet1d.py"
    if not source.exists():
        raise FileNotFoundError(
            f"Official AnyPPG network source not found: {source}. "
            "Run scripts/setup_anyppg.sh first."
        )
    spec = importlib.util.spec_from_file_location("anyppg_official_resnet1d", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official AnyPPG network source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Net1D


class AnyPPGEncoder(PPGEncoder):
    def __init__(
        self,
        repository_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "auto",
    ) -> None:
        self.repository_path = Path(repository_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.repository_commit = verify_git_checkout(
            self.repository_path,
            expected_commit=ANYPPG_REPOSITORY_COMMIT,
            model_name="AnyPPG",
        )
        self.checkpoint_sha256 = verify_checkpoint_sha256(
            self.checkpoint_path,
            expected_sha256=ANYPPG_CHECKPOINT_SHA256,
            model_name="AnyPPG",
        )

        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        network_class = _load_network_class(self.repository_path)
        self.model = network_class(
            in_channels=1,
            base_filters=64,
            ratio=1.0,
            filter_list=[64, 160, 160, 400, 400, 512],
            m_blocks_list=[2, 2, 2, 3, 3, 1],
            kernel_size=3,
            stride=2,
            groups_width=16,
            use_bn=True,
            use_do=True,
            verbose=False,
        )
        state_dict = load_torch_checkpoint(self.checkpoint_path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def preprocess(self, signal: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
        return preprocess_anyppg(signal, sampling_rate_hz)

    def preprocessing_shape_trace(
        self, signal: np.ndarray, sampling_rate_hz: int
    ) -> list[dict[str, object]]:
        original_samples = int(np.asarray(signal).size)
        resampled_samples = min(
            ANYPPG_INPUT_SAMPLES,
            int(ceil(original_samples * ANYPPG_TARGET_RATE_HZ / sampling_rate_hz)),
        )
        return [
            {"name": "raw_waveform", "shape": [original_samples], "sampling_rate_hz": sampling_rate_hz},
            {"name": "bandpass_filtered", "shape": [original_samples], "sampling_rate_hz": sampling_rate_hz},
            {"name": "resampled", "shape": [resampled_samples], "sampling_rate_hz": ANYPPG_TARGET_RATE_HZ},
            {"name": "time_axis_z_normalized", "shape": [resampled_samples], "sampling_rate_hz": ANYPPG_TARGET_RATE_HZ},
            {
                "name": "zero_padded_model_ready" if resampled_samples < ANYPPG_INPUT_SAMPLES else "model_ready_no_padding",
                "shape": [ANYPPG_INPUT_SAMPLES],
                "sampling_rate_hz": ANYPPG_TARGET_RATE_HZ,
                "padding_required": bool(resampled_samples < ANYPPG_INPUT_SAMPLES),
            },
        ]

    def encode(self, batch: np.ndarray) -> NativeEncoderOutput:
        import torch

        tensor = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
        tensor = tensor.unsqueeze(1)
        self.model.eval()
        with torch.inference_mode():
            embeddings = self.model(tensor)
        array = embeddings.detach().cpu().numpy()
        return NativeEncoderOutput(
            embedding=array,
            components={"embedding": array},
            component_semantics={
                "embedding": "512-dimensional temporally pooled encoder representation"
            },
        )

    def fingerprint(self) -> EncoderFingerprint:
        return EncoderFingerprint(
            contract_version=CONTRACT_VERSION,
            model_name="anyppg",
            model_version="official-github-2026-06-06",
            checkpoint_sha256=self.checkpoint_sha256,
            repository_commit=self.repository_commit,
            preprocessing={
                "name": ANYPPG_PREPROCESSING_POLICY,
                "resampling": "scipy.signal.resample_poly",
                "long_signal_handling": "forbidden",
                "normalization": "per_signal_zscore",
                "filter": "mne.filter.filter_data_iir_ba_butterworth_bandpass_0.5_8_hz_order_3",
                "mne_version": mne.__version__,
                "zscore_epsilon": ANYPPG_ZSCORE_EPSILON,
                "padding": "symmetric_zero_only_for_short_ppgbp_adaptation_to_1250",
                "truncation": "forbidden",
                "pooling": "official_temporal_mean",
            },
            input_sampling_rate_hz=ANYPPG_TARGET_RATE_HZ,
            input_samples=ANYPPG_INPUT_SAMPLES,
            embedding_dimension=ANYPPG_EMBEDDING_DIMENSION,
        )
