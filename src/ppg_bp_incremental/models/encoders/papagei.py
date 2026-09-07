"""Frozen wrappers for the official PaPaGei-P and PaPaGei-S checkpoints."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from math import ceil, gcd
from pathlib import Path
from types import ModuleType
from typing import Literal
import numpy as np
from scipy.signal import cheby2, filtfilt, resample_poly

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


PAPAGEI_REPOSITORY_COMMIT = MODEL_CONTRACTS["papagei_p"].source_commit
PAPAGEI_CHECKPOINT_SHA256 = {
    "p": MODEL_CONTRACTS["papagei_p"].checkpoint_sha256,
    "s": MODEL_CONTRACTS["papagei_s"].checkpoint_sha256,
}
PAPAGEI_TARGET_RATE_HZ = 125
PAPAGEI_INPUT_SAMPLES = 1250
PAPAGEI_EMBEDDING_DIMENSION = 512
PPGBP_DATASET_NAME = "PPG-BP"


def trim_ppgbp_final_raw_sample(signal: np.ndarray) -> np.ndarray:
    """Apply the released PPG-BP ``[:-1]`` operation before normalization."""

    squeezed = np.asarray(signal).squeeze()
    if squeezed.ndim != 1 or len(squeezed) < 3:
        raise ValueError(
            "Released PPG-BP final-sample trimming requires at least three raw samples"
        )
    return squeezed[:-1]


def ppgbp_final_sample_trim_metadata(*, source: str) -> dict[str, object]:
    """Return one canonical provenance record for the released PPG-BP trim."""

    return {
        "applied": True,
        "samples_removed_from_end": 1,
        "operation_order": "before_zscore_filter_resample_and_padding",
        "source": source,
    }


def _resample(signal: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    original = Fraction(original_rate).limit_denominator()
    target = Fraction(target_rate).limit_denominator()
    common_denominator = np.lcm(original.denominator, target.denominator)
    original_scaled = original * common_denominator
    target_scaled = target * common_denominator
    divisor = gcd(original_scaled.numerator, target_scaled.numerator)
    up = target_scaled.numerator // divisor
    down = original_scaled.numerator // divisor
    return resample_poly(signal, up, down, axis=0)


def preprocess_ppgbp_official(
    signal: np.ndarray,
    sampling_rate_hz: int,
    target_rate_hz: int,
    input_samples: int,
) -> np.ndarray:
    """Reproduce shared operations after dataset-specific raw preparation.

    The sequence is global z-score normalization, pyPPG-equivalent Chebyshev-II
    band-pass, conditional 50 ms smoothing at source rates of at least 75 Hz,
    polyphase resampling to the requested model rate, then symmetric zero
    padding to the requested model input length.  PaPaGei and Pulse-PPG wrappers
    apply the released PPG-BP final-sample trim before entering this function.
    """
    signal = np.asarray(signal, dtype=np.float32).squeeze()
    if signal.ndim != 1 or len(signal) < 2:
        raise ValueError("PPG-BP preprocessing requires a one-dimensional signal")
    if not np.isfinite(signal).all():
        raise ValueError("PPG-BP preprocessing received non-finite samples")
    if sampling_rate_hz <= 24:
        raise ValueError("Sampling rate must exceed twice the 12 Hz cutoff")

    normalized = (
        signal - np.mean(signal, dtype=np.float32)
    ) / (np.std(signal, dtype=np.float32) + 1e-7)
    b, a = cheby2(
        4,
        20,
        [0.5, 12],
        btype="bandpass",
        fs=sampling_rate_hz,
    )
    filtered = filtfilt(b, a, normalized)
    # The pinned pyPPG implementation applies its PPG smoothing branch only at
    # source rates of at least 75 Hz. This matters for 30 Hz BUT PPG; inventing
    # a two-sample smoother there would not follow the executable source.
    if sampling_rate_hz >= 75:
        smoothing_samples = round(sampling_rate_hz * 50 / 1000)
        smoothing = np.ones(smoothing_samples) / smoothing_samples
        filtered = filtfilt(smoothing, 1, filtered)
    resampled = _resample(filtered, sampling_rate_hz, target_rate_hz)
    padding = input_samples - len(resampled)
    if padding < 0:
        raise ValueError(
            "Requested PPG-BP model input cannot represent signals longer than "
            f"{input_samples / target_rate_hz:g} seconds; "
            "segment the signal explicitly before encoding"
        )
    left = padding // 2
    right = padding - left
    padded = np.pad(resampled, (left, right), mode="constant")
    return np.asarray(padded, dtype=np.float32)


def preprocess_papagei(
    signal: np.ndarray,
    sampling_rate_hz: int,
    *,
    dataset: str | None = None,
) -> np.ndarray:
    if dataset == PPGBP_DATASET_NAME:
        signal = trim_ppgbp_final_raw_sample(signal)
    return preprocess_ppgbp_official(
        signal,
        sampling_rate_hz,
        PAPAGEI_TARGET_RATE_HZ,
        PAPAGEI_INPUT_SAMPLES,
    )


def papagei_preprocessing_shape_trace(
    signal: np.ndarray,
    sampling_rate_hz: int,
    *,
    dataset: str | None = None,
) -> list[dict[str, object]]:
    """Trace PaPaGei preprocessing, including the released PPG-BP trim."""

    signal = np.asarray(signal)
    if dataset != PPGBP_DATASET_NAME:
        return ppgbp_preprocessing_shape_trace(
            signal,
            sampling_rate_hz,
            PAPAGEI_TARGET_RATE_HZ,
            PAPAGEI_INPUT_SAMPLES,
        )

    squeezed = signal.squeeze()
    trimmed = trim_ppgbp_final_raw_sample(squeezed)
    downstream = ppgbp_preprocessing_shape_trace(
        trimmed,
        sampling_rate_hz,
        PAPAGEI_TARGET_RATE_HZ,
        PAPAGEI_INPUT_SAMPLES,
    )
    return [
        {
            "name": "raw_waveform",
            "shape": [int(squeezed.size)],
            "sampling_rate_hz": sampling_rate_hz,
        },
        {
            "name": "released_ppgbp_final_sample_trim",
            "shape": [int(trimmed.size)],
            "sampling_rate_hz": sampling_rate_hz,
            "samples_removed_from_end": 1,
        },
        *downstream[1:],
    ]


def ppgbp_preprocessing_shape_trace(
    signal: np.ndarray,
    sampling_rate_hz: int,
    target_rate_hz: int,
    input_samples: int,
) -> list[dict[str, object]]:
    original_samples = int(np.asarray(signal).size)
    resampled_samples = min(
        input_samples,
        int(ceil(original_samples * target_rate_hz / sampling_rate_hz)),
    )
    return [
        {"name": "raw_waveform", "shape": [original_samples], "sampling_rate_hz": sampling_rate_hz},
        {"name": "z_normalized", "shape": [original_samples], "sampling_rate_hz": sampling_rate_hz},
        {"name": "bandpass_filtered", "shape": [original_samples], "sampling_rate_hz": sampling_rate_hz},
        {
            "name": (
                "smoothed_50_ms"
                if sampling_rate_hz >= 75
                else "smoothing_skipped_below_75_hz_upstream_rule"
            ),
            "shape": [original_samples],
            "sampling_rate_hz": sampling_rate_hz,
        },
        {"name": "resampled", "shape": [resampled_samples], "sampling_rate_hz": target_rate_hz},
        {
            "name": "zero_padded_model_ready" if resampled_samples < input_samples else "model_ready_no_padding",
            "shape": [input_samples],
            "sampling_rate_hz": target_rate_hz,
            "padding_required": bool(resampled_samples < input_samples),
        },
    ]


def _load_resnet_module(repository_path: Path) -> ModuleType:
    source = repository_path / "models" / "resnet.py"
    if not source.exists():
        raise FileNotFoundError(
            f"Official PaPaGei model source not found: {source}. "
            "Run scripts/setup_papagei.sh first."
        )
    # The pinned upstream file imports plotting/progress/reporting packages that
    # are unused by the architecture classes. Execute the same source after
    # removing only those exact imports, keeping inference lightweight.
    unused_imports = {
        "from tqdm import tqdm",
        "from matplotlib import pyplot as plt",
        "from sklearn.metrics import classification_report",
    }
    source_text = source.read_text(encoding="utf-8")
    filtered_source = "\n".join(
        line for line in source_text.splitlines() if line.strip() not in unused_imports
    )
    module = ModuleType("papagei_official_resnet")
    module.__file__ = str(source)
    exec(compile(filtered_source, str(source), "exec"), module.__dict__)
    return module


class PapageiEncoder(PPGEncoder):
    def __init__(
        self,
        variant: Literal["p", "s"],
        repository_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "auto",
    ) -> None:
        if variant not in {"p", "s"}:
            raise ValueError("PaPaGei variant must be 'p' or 's'")
        self.variant = variant
        self.repository_path = Path(repository_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.repository_commit = verify_git_checkout(
            self.repository_path,
            expected_commit=PAPAGEI_REPOSITORY_COMMIT,
            model_name="PaPaGei",
        )
        self.checkpoint_sha256 = verify_checkpoint_sha256(
            self.checkpoint_path,
            expected_sha256=PAPAGEI_CHECKPOINT_SHA256[variant],
            model_name=f"PaPaGei-{variant.upper()}",
        )

        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        module = _load_resnet_module(self.repository_path)
        common = dict(
            in_channels=1,
            base_filters=32,
            kernel_size=3,
            stride=2,
            groups=1,
            n_block=18,
            n_classes=PAPAGEI_EMBEDDING_DIMENSION,
        )
        if variant == "s":
            self.model = module.ResNet1DMoE(**common, n_experts=3)
        else:
            self.model = module.ResNet1D(**common)
        checkpoint = load_torch_checkpoint(self.checkpoint_path)
        state_dict = {
            key.removeprefix("module."): value for key, value in checkpoint.items()
        }
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def preprocess(self, signal: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
        return preprocess_papagei(signal, sampling_rate_hz)

    def preprocess_for_dataset(
        self,
        signal: np.ndarray,
        sampling_rate_hz: int,
        *,
        dataset: str,
    ) -> np.ndarray:
        return preprocess_papagei(
            signal,
            sampling_rate_hz,
            dataset=dataset,
        )

    def preprocessing_shape_trace(
        self, signal: np.ndarray, sampling_rate_hz: int
    ) -> list[dict[str, object]]:
        return papagei_preprocessing_shape_trace(
            signal,
            sampling_rate_hz,
        )

    def preprocessing_shape_trace_for_dataset(
        self,
        signal: np.ndarray,
        sampling_rate_hz: int,
        *,
        dataset: str,
    ) -> list[dict[str, object]]:
        return papagei_preprocessing_shape_trace(
            signal,
            sampling_rate_hz,
            dataset=dataset,
        )

    def encode(self, batch: np.ndarray) -> NativeEncoderOutput:
        import torch

        tensor = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
        tensor = tensor.unsqueeze(1)
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(tensor)
        if self.variant == "s":
            names = (
                "downstream_dense_embedding",
                "ipa",
                "sqi",
                "pooled_embedding",
            )
            semantics = {
                "downstream_dense_embedding": (
                    "512-dimensional dense-transformed downstream representation"
                ),
                "ipa": "raw inflection-point-area morphology prediction",
                "sqi": "raw signal-quality-index morphology prediction",
                "pooled_embedding": "512-dimensional pooled backbone representation",
            }
        else:
            names = ("downstream_dense_embedding", "pooled_embedding")
            semantics = {
                "downstream_dense_embedding": (
                    "512-dimensional dense-transformed downstream representation"
                ),
                "pooled_embedding": "512-dimensional pooled backbone representation",
            }
        if not isinstance(outputs, tuple) or len(outputs) != len(names):
            raise ValueError(
                f"Unexpected PaPaGei-{self.variant.upper()} native output structure"
            )
        components = {
            name: value.detach().cpu().numpy() for name, value in zip(names, outputs)
        }
        return NativeEncoderOutput(
            embedding=components["downstream_dense_embedding"],
            components=components,
            component_semantics=semantics,
        )

    def fingerprint(self) -> EncoderFingerprint:
        return EncoderFingerprint(
            contract_version=CONTRACT_VERSION,
            model_name=f"papagei_{self.variant}",
            model_version="zenodo-13983110",
            checkpoint_sha256=self.checkpoint_sha256,
            repository_commit=self.repository_commit,
            preprocessing={
                "name": "official_ppgbp_notebook_equivalent",
                "zscore_epsilon": 1e-7,
                "filter": "cheby2_bandpass",
                "filter_order": 4,
                "filter_stopband_attenuation_db": 20,
                "filter_low_hz": 0.5,
                "filter_high_hz": 12,
                "smoothing_window_ms": 50,
                "smoothing_applied_when_source_rate_hz_gte": 75,
                "resampling": "scipy.signal.resample_poly",
                "padding": "symmetric_zero_only_if_short_to_1250",
                "dataset_specific_operations": {
                    PPGBP_DATASET_NAME: {
                        "final_raw_sample_trim": {
                            "samples_removed_from_end": 1,
                            "operation_order": (
                                "before_zscore_filter_resample_and_padding"
                            ),
                            "source": "example_papagei.ipynb (PPG-BP preparation cell)",
                        }
                    }
                },
                "output_index": 0,
            },
            input_sampling_rate_hz=PAPAGEI_TARGET_RATE_HZ,
            input_samples=PAPAGEI_INPUT_SAMPLES,
            embedding_dimension=PAPAGEI_EMBEDDING_DIMENSION,
        )

    def fingerprint_for_dataset(self, dataset: str) -> EncoderFingerprint:
        fingerprint = self.fingerprint()
        if dataset != PPGBP_DATASET_NAME:
            return fingerprint
        preprocessing = dict(fingerprint.preprocessing)
        preprocessing["ppgbp_final_raw_sample_trim"] = (
            ppgbp_final_sample_trim_metadata(
                source="example_papagei.ipynb (PPG-BP preparation cell)"
            )
        )
        return replace(
            fingerprint,
            model_version=f"{fingerprint.model_version}-ppgbp-trim-v1",
            preprocessing=preprocessing,
        )
