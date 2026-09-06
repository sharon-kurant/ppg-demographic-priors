"""Versioned, source-audited contracts for the active BP benchmark.

The registry deliberately separates three things that previous iterations
blurred together:

1. the waveform operations required by a released encoder;
2. the native tensors returned by its checkpoint; and
3. the explicit downstream estimator that turns a selected representation into
   a blood-pressure prediction.

No tensor receives BP units merely because of its shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import numpy as np


CONTRACT_VERSION = "source-faithful-v2"
FidelityStatus = Literal[
    "released_exact",
    "paper_specified",
    "source_consistent_adaptation",
]
OutputScale = Literal["raw_mmhg", "z_score"]


@dataclass(frozen=True)
class NativeComponentContract:
    name: str
    shape: str
    meaning: str


@dataclass(frozen=True)
class ModelContract:
    key: str
    display_name: str
    source_commit: str
    checkpoint_sha256: str
    parameter_count: int
    architecture: str
    input_sampling_rate_hz: int
    input_duration_seconds: float
    input_shape: str
    preprocessing_operations: tuple[str, ...]
    native_outputs: tuple[NativeComponentContract, ...]
    selected_representation: str
    selected_dimension: int
    bp_estimator: str
    bp_target_scale: OutputScale
    bp_decoder: str
    bp_estimator_source: str
    direct_encoder_demographics: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BPTaskOutput:
    """Output of an explicit BP estimator, separate from encoder outputs."""

    target: Literal["sbp", "dbp"]
    estimator_output: np.ndarray
    output_scale: OutputScale
    decoder_metadata: Mapping[str, Any]
    mmhg_prediction: np.ndarray

    def checked(self, expected_rows: int | None = None) -> "BPTaskOutput":
        estimator = np.asarray(self.estimator_output, dtype=float).reshape(-1)
        prediction = np.asarray(self.mmhg_prediction, dtype=float).reshape(-1)
        if estimator.shape != prediction.shape:
            raise ValueError("Estimator and decoded BP outputs must have the same shape")
        if expected_rows is not None and estimator.shape != (expected_rows,):
            raise ValueError(
                f"BP estimator returned {estimator.shape}; expected ({expected_rows},)"
            )
        if not np.isfinite(estimator).all() or not np.isfinite(prediction).all():
            raise ValueError("BP task output contains non-finite values")
        if self.target not in {"sbp", "dbp"}:
            raise ValueError("BP target must be sbp or dbp")
        return BPTaskOutput(
            target=self.target,
            estimator_output=estimator,
            output_scale=self.output_scale,
            decoder_metadata=dict(self.decoder_metadata),
            mmhg_prediction=prediction,
        )


def decode_bp_output(
    values: np.ndarray,
    *,
    target: Literal["sbp", "dbp"],
    output_scale: OutputScale,
    decoder_metadata: Mapping[str, Any],
) -> BPTaskOutput:
    """Decode only an explicitly declared estimator scale.

    Active Ridge tasks use ``raw_mmhg`` with identity decoding.  The z-score
    branch exists for a future, source-declared adapter and refuses to operate
    without an explicit training-fold mean and standard deviation.
    """

    estimator = np.asarray(values, dtype=float).reshape(-1)
    if output_scale == "raw_mmhg":
        if decoder_metadata.get("type") != "identity":
            raise ValueError("raw_mmhg output requires an identity decoder declaration")
        prediction = estimator.copy()
    elif output_scale == "z_score":
        if decoder_metadata.get("type") != "affine_z_score":
            raise ValueError("z_score output requires an affine_z_score decoder")
        mean = float(decoder_metadata["mean_mmhg"])
        std = float(decoder_metadata["std_mmhg"])
        if not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError("z-score decoder requires finite mean and positive std")
        prediction = estimator * std + mean
    else:  # pragma: no cover - protected by the public type and explicit guard
        raise ValueError(f"Unsupported BP output scale: {output_scale}")
    return BPTaskOutput(
        target=target,
        estimator_output=estimator,
        output_scale=output_scale,
        decoder_metadata=dict(decoder_metadata),
        mmhg_prediction=prediction,
    ).checked()


MODEL_CONTRACTS: dict[str, ModelContract] = {
    "papagei_p": ModelContract(
        key="papagei_p",
        display_name="PaPaGei-P",
        source_commit="0c537dad4d2850e15b724260de820dd68d77f0b0",
        checkpoint_sha256="3a6850961af527cbb2e476d3ad0bb374ae86ed86ffdd55b0b0d7e2f49451518e",
        parameter_count=4_993_024,
        architecture="18-block 1-D ResNet participant-contrastive encoder",
        input_sampling_rate_hz=125,
        input_duration_seconds=10.0,
        input_shape="(B, 1, 1250)",
        preprocessing_operations=(
            "per-record z-normalization (epsilon 1e-7)",
            "0.5-12 Hz fourth-order Chebyshev-II band-pass, 20 dB attenuation",
            "50 ms zero-phase moving-average smoothing when source rate is at least 75 Hz",
            "polyphase resampling to 125 Hz",
            "symmetric zero-padding only when the source recording is shorter than 10 s",
        ),
        native_outputs=(
            NativeComponentContract(
                "projected_embedding", "(B, 512)", "downstream projected representation"
            ),
            NativeComponentContract(
                "pooled_embedding", "(B, 512)", "pooled backbone representation"
            ),
        ),
        selected_representation="projected_embedding",
        selected_dimension=512,
        bp_estimator="feature standardization followed by separate Ridge estimators",
        bp_target_scale="raw_mmhg",
        bp_decoder="identity",
        bp_estimator_source="released PaPaGei linear-probing regression code",
        direct_encoder_demographics=(),
    ),
    "papagei_s": ModelContract(
        key="papagei_s",
        display_name="PaPaGei-S",
        source_commit="0c537dad4d2850e15b724260de820dd68d77f0b0",
        checkpoint_sha256="79d68671e51bdc951fee853e5a8241099e38148ff26ed888b843849332e0b988",
        parameter_count=5_785_612,
        architecture="18-block 1-D ResNet mixture-of-experts morphology encoder",
        input_sampling_rate_hz=125,
        input_duration_seconds=10.0,
        input_shape="(B, 1, 1250)",
        preprocessing_operations=(
            "per-record z-normalization (epsilon 1e-7)",
            "0.5-12 Hz fourth-order Chebyshev-II band-pass, 20 dB attenuation",
            "50 ms zero-phase moving-average smoothing when source rate is at least 75 Hz",
            "polyphase resampling to 125 Hz",
            "symmetric zero-padding only when the source recording is shorter than 10 s",
        ),
        native_outputs=(
            NativeComponentContract(
                "projected_embedding", "(B, 512)", "downstream projected representation"
            ),
            NativeComponentContract("ipa", "(B, 1)", "inflection-point-area prediction"),
            NativeComponentContract("sqi", "(B, 1)", "signal-quality-index prediction"),
            NativeComponentContract(
                "pooled_embedding", "(B, 512)", "pooled backbone representation"
            ),
        ),
        selected_representation="projected_embedding",
        selected_dimension=512,
        bp_estimator="feature standardization followed by separate Ridge estimators",
        bp_target_scale="raw_mmhg",
        bp_decoder="identity",
        bp_estimator_source="released PaPaGei linear-probing regression code",
        direct_encoder_demographics=(),
    ),
    "pulseppg": ModelContract(
        key="pulseppg",
        display_name="Pulse-PPG",
        source_commit="716eaf9cf966e8f76436f2263872ef38b1f90166",
        checkpoint_sha256="485ade5033b3baa9b82e252fc131042dda897d7bdfc0d1a030d9746a1e98857c",
        parameter_count=28_497_920,
        architecture="12-block/26-convolution 1-D ResNet with released max pooling",
        input_sampling_rate_hz=50,
        input_duration_seconds=10.0,
        input_shape="(B, 1, 500)",
        preprocessing_operations=(
            "for PPG-BP only, discard the final raw sample as in released PPGBP.py",
            "per-record z-normalization (epsilon 1e-7)",
            "0.5-12 Hz fourth-order Chebyshev-II band-pass, 20 dB attenuation",
            "50 ms zero-phase moving-average smoothing",
            "polyphase resampling to 50 Hz",
            "symmetric zero-padding only when the source recording is shorter than 10 s",
        ),
        native_outputs=(
            NativeComponentContract(
                "embedding", "(B, 512)", "released max-pooled encoder representation"
            ),
        ),
        selected_representation="embedding",
        selected_dimension=512,
        bp_estimator="feature standardization followed by separate Ridge estimators",
        bp_target_scale="raw_mmhg",
        bp_decoder="identity",
        bp_estimator_source="released Pulse-PPG linear-probing regression code",
        direct_encoder_demographics=(),
    ),
    "anyppg": ModelContract(
        key="anyppg",
        display_name="AnyPPG",
        source_commit="661b877aa96eac3bba320a462cb2a3bfea991103",
        checkpoint_sha256="99b9bb0a3c2b83a1f5d8ca2963fbd25329b6530e8d337de8825722fc6fd5f4fa",
        parameter_count=4_044_272,
        architecture="multi-stage 1-D ResNet trained with ECG-guided contrastive learning",
        input_sampling_rate_hz=125,
        input_duration_seconds=10.0,
        input_shape="(B, 1, 1250)",
        preprocessing_operations=(
            "0.5-8 Hz third-order Butterworth IIR using MNE zero-phase semantics",
            "polyphase resampling to 125 Hz",
            "time-axis z-normalization (epsilon 1e-5)",
            "symmetric zero-padding only for the short PPG-BP adaptation",
        ),
        native_outputs=(
            NativeComponentContract(
                "embedding", "(B, 512)", "temporally mean-pooled encoder representation"
            ),
        ),
        selected_representation="embedding",
        selected_dimension=512,
        bp_estimator="feature standardization followed by separate Ridge estimators",
        bp_target_scale="raw_mmhg",
        bp_decoder="identity",
        bp_estimator_source="released AnyPPG linear-probing code; downstream data construction adapted",
        direct_encoder_demographics=(),
    ),
}


def source_fidelity(model: str, dataset: str) -> FidelityStatus:
    """Return fidelity of the active input/BP pipeline for one cohort."""

    if model in {"papagei_p", "papagei_s", "pulseppg"} and dataset == "PPG-BP":
        return "released_exact"
    if model == "anyppg" and dataset.startswith("PulseDB"):
        # Input preprocessing is paper/code specified, but BP use is an overlap
        # evaluation through the released linear-probe pattern.
        return "paper_specified"
    return "source_consistent_adaptation"


def contract_registry_payload() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "models": {key: contract.as_dict() for key, contract in MODEL_CONTRACTS.items()},
    }
