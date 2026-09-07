"""Frozen wrapper for the official field-trained Pulse-PPG checkpoint."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import ModuleType
import warnings

import numpy as np

from ppg_bp_incremental.models.encoders.base import (
    EncoderFingerprint,
    NativeEncoderOutput,
    PPGEncoder,
)
from ppg_bp_incremental.models.encoders.papagei import (
    PPGBP_DATASET_NAME,
    ppgbp_final_sample_trim_metadata,
    ppgbp_preprocessing_shape_trace,
    preprocess_ppgbp_official,
    trim_ppgbp_final_raw_sample,
)
from ppg_bp_incremental.models.contracts import CONTRACT_VERSION, MODEL_CONTRACTS
from ppg_bp_incremental.models.encoders.source_integrity import (
    load_torch_checkpoint,
    verify_checkpoint_sha256,
    verify_git_checkout,
)


PULSEPPG_REPOSITORY_COMMIT = MODEL_CONTRACTS["pulseppg"].source_commit
PULSEPPG_CHECKPOINT_SHA256 = MODEL_CONTRACTS["pulseppg"].checkpoint_sha256
PULSEPPG_TARGET_RATE_HZ = 50
PULSEPPG_INPUT_SAMPLES = 500
PULSEPPG_EMBEDDING_DIMENSION = 512


def preprocess_pulseppg(
    signal: np.ndarray,
    sampling_rate_hz: int,
    *,
    dataset: str | None = None,
) -> np.ndarray:
    """Apply the effective Pulse-PPG waveform contract.

    The released ``pulseppg/data/process/PPGBP.py`` script removes the final
    raw PPG-BP sample with ``signal.values.squeeze()[:-1]`` *before*
    normalization.  This is a dataset-specific downstream operation, not a
    generic encoder requirement, so other cohorts retain every raw sample.
    """

    signal = np.asarray(signal)
    if dataset == PPGBP_DATASET_NAME:
        signal = trim_ppgbp_final_raw_sample(signal)
    return preprocess_ppgbp_official(
        signal,
        sampling_rate_hz,
        PULSEPPG_TARGET_RATE_HZ,
        PULSEPPG_INPUT_SAMPLES,
    )


def pulseppg_preprocessing_shape_trace(
    signal: np.ndarray,
    sampling_rate_hz: int,
    *,
    dataset: str | None = None,
) -> list[dict[str, object]]:
    """Trace Pulse-PPG preprocessing, including the released PPG-BP trim."""

    signal = np.asarray(signal)
    if dataset != PPGBP_DATASET_NAME:
        return ppgbp_preprocessing_shape_trace(
            signal,
            sampling_rate_hz,
            PULSEPPG_TARGET_RATE_HZ,
            PULSEPPG_INPUT_SAMPLES,
        )

    squeezed = signal.squeeze()
    trimmed = trim_ppgbp_final_raw_sample(squeezed)
    downstream = ppgbp_preprocessing_shape_trace(
        trimmed,
        sampling_rate_hz,
        PULSEPPG_TARGET_RATE_HZ,
        PULSEPPG_INPUT_SAMPLES,
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


def _load_network_module(repository_path: Path) -> ModuleType:
    source = repository_path / "pulseppg" / "nets" / "ResNet1D" / "ResNet1D_Net.py"
    if not source.exists():
        raise FileNotFoundError(
            f"Official Pulse-PPG network source not found: {source}. "
            "Run scripts/setup_pulseppg.sh first."
        )
    unused_imports = {
        "from tqdm import tqdm",
        "from matplotlib import pyplot as plt",
        "from sklearn.metrics import classification_report",
    }
    source_text = source.read_text(encoding="utf-8")
    filtered_source = "\n".join(
        line for line in source_text.splitlines() if line.strip() not in unused_imports
    )
    module = ModuleType("pulseppg_official_resnet")
    module.__file__ = str(source)
    exec(compile(filtered_source, str(source), "exec"), module.__dict__)
    return module


class PulsePPGEncoder(PPGEncoder):
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
            expected_commit=PULSEPPG_REPOSITORY_COMMIT,
            model_name="Pulse-PPG",
        )
        self.checkpoint_sha256 = verify_checkpoint_sha256(
            self.checkpoint_path,
            expected_sha256=PULSEPPG_CHECKPOINT_SHA256,
            model_name="Pulse-PPG",
        )

        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        module = _load_network_module(self.repository_path)
        self.model = module.Net(
            in_channels=1,
            base_filters=128,
            kernel_size=11,
            stride=2,
            groups=1,
            n_block=12,
            finalpool="max",
        )
        checkpoint = load_torch_checkpoint(self.checkpoint_path)
        if "net" not in checkpoint:
            raise ValueError("Pulse-PPG checkpoint does not contain a 'net' state dict")
        self.model.load_state_dict(checkpoint["net"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def preprocess(self, signal: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
        return preprocess_pulseppg(signal, sampling_rate_hz)

    def preprocess_for_dataset(
        self,
        signal: np.ndarray,
        sampling_rate_hz: int,
        *,
        dataset: str,
    ) -> np.ndarray:
        return preprocess_pulseppg(
            signal,
            sampling_rate_hz,
            dataset=dataset,
        )

    def preprocessing_shape_trace(
        self, signal: np.ndarray, sampling_rate_hz: int
    ) -> list[dict[str, object]]:
        return pulseppg_preprocessing_shape_trace(
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
        return pulseppg_preprocessing_shape_trace(
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
            embeddings = self._forward_without_upstream_instancenorm_warning(tensor)
        array = embeddings.detach().cpu().numpy()
        return NativeEncoderOutput(
            embedding=array,
            components={"embedding": array},
            component_semantics={
                "embedding": "512-dimensional max-pooled encoder representation"
            },
        )

    def _forward_without_upstream_instancenorm_warning(self, tensor):
        # The released model constructs a non-affine InstanceNorm1d with the
        # final channel count. PyTorch explicitly ignores num_features when
        # affine=False, but warns because the raw input has one channel.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="input's size at dim=1 does not match num_features.*",
                category=UserWarning,
            )
            return self.model(tensor)

    def fingerprint(self) -> EncoderFingerprint:
        return EncoderFingerprint(
            contract_version=CONTRACT_VERSION,
            model_name="pulseppg",
            model_version="zenodo-17345536-v4",
            checkpoint_sha256=self.checkpoint_sha256,
            repository_commit=self.repository_commit,
            preprocessing={
                "name": "official_ppgbp_script_equivalent",
                "zscore_epsilon": 1e-7,
                "filter": "cheby2_bandpass",
                "filter_order": 4,
                "filter_stopband_attenuation_db": 20,
                "filter_low_hz": 0.5,
                "filter_high_hz": 12,
                "smoothing_window_ms": 50,
                "smoothing_applied_when_source_rate_hz_gte": 75,
                "resampling": "scipy.signal.resample_poly",
                "padding": "symmetric_zero_only_if_short_to_500",
                "dataset_specific_operations": {
                    PPGBP_DATASET_NAME: {
                        "final_raw_sample_trim": {
                            "samples_removed_from_end": 1,
                            "operation_order": (
                                "before_zscore_filter_resample_and_padding"
                            ),
                            "source": "pulseppg/data/process/PPGBP.py:90",
                        }
                    }
                },
                "pooling": "model_max_pool",
            },
            input_sampling_rate_hz=PULSEPPG_TARGET_RATE_HZ,
            input_samples=PULSEPPG_INPUT_SAMPLES,
            embedding_dimension=PULSEPPG_EMBEDDING_DIMENSION,
        )

    def fingerprint_for_dataset(self, dataset: str) -> EncoderFingerprint:
        fingerprint = self.fingerprint()
        if dataset != PPGBP_DATASET_NAME:
            return fingerprint
        preprocessing = dict(fingerprint.preprocessing)
        preprocessing["ppgbp_final_raw_sample_trim"] = (
            ppgbp_final_sample_trim_metadata(
                source="pulseppg/data/process/PPGBP.py:90"
            )
        )
        return replace(
            fingerprint,
            model_version=f"{fingerprint.model_version}-ppgbp-trim-v1",
            preprocessing=preprocessing,
        )
