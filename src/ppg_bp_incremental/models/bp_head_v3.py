"""Trainable BP adapters that preserve every native foundation-model output.

This module is intentionally separate from the frozen encoder wrappers.  Those
wrappers remain the source of truth for checkpoint construction and waveform
preprocessing, while :class:`FoundationBPRegressorV3` exposes the wrapped
PyTorch module for either a frozen-head control or full end-to-end fine-tuning.

The regression head predicts a BP z-score fitted on an explicitly named data
pool.  Decoding to mmHg is an affine operation whose mean and standard
deviation are stored as model buffers.  Native encoder tensors never acquire
BP units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping
import warnings

import torch
from torch import nn

from ppg_bp_incremental.models.stabilization_v3 import (
    EmbeddingStandardizationStatistics,
    FixedEmbeddingStandardizer,
    zero_initialize_scalar_output_layer,
)


_NATIVE_NAMES = {
    "papagei_p": ("downstream_dense_embedding", "pooled_embedding"),
    "papagei_s": (
        "downstream_dense_embedding",
        "ipa",
        "sqi",
        "pooled_embedding",
    ),
    "pulseppg": ("embedding",),
    "anyppg": ("embedding",),
}

_NATIVE_SEMANTICS = {
    "papagei_p": {
        "downstream_dense_embedding": (
            "512-dimensional dense-transformed downstream representation"
        ),
        "pooled_embedding": "512-dimensional pooled backbone representation",
    },
    "papagei_s": {
        "downstream_dense_embedding": (
            "512-dimensional dense-transformed downstream representation"
        ),
        "ipa": "native inflection-point-area auxiliary prediction",
        "sqi": "native signal-quality-index auxiliary prediction",
        "pooled_embedding": "512-dimensional pooled backbone representation",
    },
    "pulseppg": {
        "embedding": "512-dimensional max-pooled encoder representation",
    },
    "anyppg": {
        "embedding": "512-dimensional temporally mean-pooled encoder representation",
    },
}


@dataclass(frozen=True)
class NativeTorchOutput:
    """Differentiable, named view of the checkpoint's unmodified output."""

    raw: Any
    embedding: torch.Tensor
    components: Mapping[str, torch.Tensor]
    component_semantics: Mapping[str, str]

    def component_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: tuple(value.shape) for name, value in self.components.items()}


@dataclass(frozen=True)
class BPTaskOutput:
    """Explicit BP-task output; native tensors remain available separately."""

    native: NativeTorchOutput
    prediction_zscore: torch.Tensor
    prediction_mmhg: torch.Tensor
    decoder_metadata: Mapping[str, float | str]


class TargetZScoreDecoder(nn.Module):
    """Explicitly scoped target transform and differentiable mmHg decoder."""

    def __init__(
        self,
        mean_mmhg: float,
        standard_deviation_mmhg: float,
        fit_pool_role: Literal[
            "outer_training_selection", "full_development_refit"
        ],
    ) -> None:
        super().__init__()
        if not float(standard_deviation_mmhg) > 0:
            raise ValueError("Target standard deviation must be positive")
        if not torch.isfinite(torch.tensor([mean_mmhg, standard_deviation_mmhg])).all():
            raise ValueError("Target z-score parameters must be finite")
        self.register_buffer("mean_mmhg", torch.tensor(float(mean_mmhg)))
        self.register_buffer(
            "standard_deviation_mmhg",
            torch.tensor(float(standard_deviation_mmhg)),
        )
        self.fit_pool_role = fit_pool_role

    def encode(self, target_mmhg: torch.Tensor) -> torch.Tensor:
        return (target_mmhg - self.mean_mmhg) / self.standard_deviation_mmhg

    def forward(self, prediction_zscore: torch.Tensor) -> torch.Tensor:
        return prediction_zscore * self.standard_deviation_mmhg + self.mean_mmhg

    def metadata(self) -> dict[str, float | str]:
        scale = {
            "outer_training_selection": "outer_training_pool_zscore",
            "full_development_refit": "full_development_pool_zscore",
        }[self.fit_pool_role]
        decoder = {
            "outer_training_selection": (
                "affine_outer_training_pool_inverse_zscore"
            ),
            "full_development_refit": (
                "affine_full_development_pool_inverse_zscore"
            ),
        }[self.fit_pool_role]
        return {
            "fit_pool_role": self.fit_pool_role,
            "estimator_output_scale": scale,
            "decoder": decoder,
            "target_mean_mmhg": float(self.mean_mmhg.detach().cpu()),
            "target_standard_deviation_mmhg": float(
                self.standard_deviation_mmhg.detach().cpu()
            ),
            "final_output_scale": "raw_mmhg",
        }


class BPRegressionHeadV3(nn.Module):
    """Standardized benchmark head: 512 -> 128 -> GELU -> one BP z-score."""

    def __init__(
        self,
        embedding_dimension: int = 512,
        demographic_dimension: int = 0,
        hidden_dimension: int = 128,
        zero_initialize_output_layer: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dimension < 1 or demographic_dimension < 0 or hidden_dimension < 1:
            raise ValueError("BP-head dimensions must be positive (demographics may be zero)")
        self.embedding_dimension = int(embedding_dimension)
        self.demographic_dimension = int(demographic_dimension)
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dimension + self.demographic_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )
        self.zero_initialized_output_layer = bool(zero_initialize_output_layer)
        if self.zero_initialized_output_layer:
            zero_initialize_scalar_output_layer(self.network[-1])

    def forward(
        self,
        embedding: torch.Tensor,
        demographics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embedding.ndim != 2 or embedding.shape[1] != self.embedding_dimension:
            raise ValueError(
                "BP head expected embedding shape "
                f"(B, {self.embedding_dimension}), got {tuple(embedding.shape)}"
            )
        if self.demographic_dimension:
            if demographics is None:
                raise ValueError("This BP head requires demographic features")
            expected = (embedding.shape[0], self.demographic_dimension)
            if tuple(demographics.shape) != expected:
                raise ValueError(
                    f"BP head expected demographics shape {expected}, "
                    f"got {tuple(demographics.shape)}"
                )
            embedding = torch.cat((embedding, demographics), dim=1)
        elif demographics is not None:
            raise ValueError("Demographics were supplied to a waveform-only BP head")
        output = self.network(embedding)
        if tuple(output.shape) != (embedding.shape[0], 1):
            raise RuntimeError(f"BP head produced unexpected shape {tuple(output.shape)}")
        return output


class FoundationBPRegressorV3(nn.Module):
    """Foundation encoder plus a standardized, single-target neural BP head.

    ``encoder`` is one of the audited frozen wrappers returned by
    :func:`ppg_bp_incremental.models.registry.create_encoder`.  The wrapper's
    checkpoint-loaded ``model`` is reused directly, so epoch-zero native
    outputs are identical in evaluation mode.
    """

    def __init__(
        self,
        encoder: Any,
        model_key: str,
        *,
        target_mean_mmhg: float,
        target_standard_deviation_mmhg: float,
        target_fit_pool_role: Literal[
            "outer_training_selection", "full_development_refit"
        ],
        use_demographics: bool = False,
        demographic_dimension: int = 0,
        hidden_dimension: int = 128,
        zero_initialize_output_layer: bool = False,
    ) -> None:
        super().__init__()
        if model_key not in _NATIVE_NAMES:
            raise ValueError(f"Unsupported foundation model {model_key!r}")
        if not hasattr(encoder, "model") or not isinstance(encoder.model, nn.Module):
            raise TypeError("Fine-tuning requires an audited PyTorch encoder wrapper")
        fingerprint = encoder.fingerprint()
        self.model_key = model_key
        self.encoder_model = encoder.model
        self.embedding_dimension = int(fingerprint.embedding_dimension)
        self.use_demographics = bool(use_demographics)
        effective_demographic_dimension = (
            int(demographic_dimension) if self.use_demographics else 0
        )
        if self.use_demographics and effective_demographic_dimension < 1:
            raise ValueError("Demographic fine-tuning requires a nonzero feature dimension")
        if not self.use_demographics and demographic_dimension:
            raise ValueError("demographic_dimension must be zero when demographics are disabled")
        self.head = BPRegressionHeadV3(
            embedding_dimension=self.embedding_dimension,
            demographic_dimension=effective_demographic_dimension,
            hidden_dimension=hidden_dimension,
            zero_initialize_output_layer=zero_initialize_output_layer,
        )
        self.embedding_standardizer = FixedEmbeddingStandardizer(
            self.embedding_dimension
        )
        self.decoder = TargetZScoreDecoder(
            target_mean_mmhg,
            target_standard_deviation_mmhg,
            target_fit_pool_role,
        )

    def native_forward(self, waveform: torch.Tensor) -> NativeTorchOutput:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(
                f"Foundation model input must have shape (B, 1, L), got {tuple(waveform.shape)}"
            )
        if self.model_key == "pulseppg":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="input's size at dim=1 does not match num_features.*",
                    category=UserWarning,
                )
                raw = self.encoder_model(waveform)
        else:
            raw = self.encoder_model(waveform)

        names = _NATIVE_NAMES[self.model_key]
        if self.model_key.startswith("papagei_"):
            if not isinstance(raw, tuple) or len(raw) != len(names):
                observed = len(raw) if isinstance(raw, tuple) else type(raw).__name__
                raise RuntimeError(
                    f"{self.model_key} native output contract mismatch: {observed}"
                )
            values = raw
        else:
            if not isinstance(raw, torch.Tensor):
                raise RuntimeError(
                    f"{self.model_key} native output must be one tensor, got {type(raw)!r}"
                )
            values = (raw,)
        components = dict(zip(names, values, strict=True))
        embedding_name = (
            "downstream_dense_embedding"
            if self.model_key.startswith("papagei_")
            else "embedding"
        )
        embedding = components[embedding_name]
        expected = (waveform.shape[0], self.embedding_dimension)
        if tuple(embedding.shape) != expected:
            raise RuntimeError(
                f"Selected embedding has shape {tuple(embedding.shape)}, expected {expected}"
            )
        for name, value in components.items():
            if not isinstance(value, torch.Tensor) or value.shape[0] != waveform.shape[0]:
                raise RuntimeError(f"Invalid native component {name!r}")
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite native component {name!r}")
        return NativeTorchOutput(
            raw=raw,
            embedding=embedding,
            components=components,
            component_semantics=_NATIVE_SEMANTICS[self.model_key],
        )

    def forward(
        self,
        waveform: torch.Tensor,
        demographics: torch.Tensor | None = None,
    ) -> BPTaskOutput:
        native = self.native_forward(waveform)
        if not self.use_demographics:
            demographics = None
        prediction_zscore = self.head(
            self.embedding_for_head(native.embedding), demographics
        )
        prediction_mmhg = self.decoder(prediction_zscore)
        return BPTaskOutput(
            native=native,
            prediction_zscore=prediction_zscore,
            prediction_mmhg=prediction_mmhg,
            decoder_metadata=self.decoder.metadata(),
        )

    def configure_embedding_standardization(
        self, statistics: EmbeddingStandardizationStatistics
    ) -> None:
        """Install fixed epoch-zero statistics used only at the BP head input."""

        self.embedding_standardizer.configure(statistics)

    def embedding_for_head(self, embedding: torch.Tensor) -> torch.Tensor:
        """Return the optional standardized head input; native output is unchanged."""

        return self.embedding_standardizer(embedding)

    def stabilization_summary(self) -> dict[str, Any]:
        return {
            "embedding_standardization": self.embedding_standardizer.summary(),
            "zero_initialize_output_layer": bool(
                self.head.zero_initialized_output_layer
            ),
        }

    @staticmethod
    def _is_papagei_s_auxiliary_parameter(name: str) -> bool:
        return name.startswith(("expert_layers_", "gating_network_"))

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Enable the complete selected-embedding path, excluding unused heads."""

        for name, parameter in self.encoder_model.named_parameters():
            unused_auxiliary = (
                self.model_key == "papagei_s"
                and self._is_papagei_s_auxiliary_parameter(name)
            )
            parameter.requires_grad_(bool(trainable) and not unused_auxiliary)

    def optimizer_groups(
        self,
        encoder_learning_rate: float,
        *,
        head_learning_rate_multiplier: float = 10.0,
    ) -> list[dict[str, object]]:
        if encoder_learning_rate <= 0 or head_learning_rate_multiplier <= 0:
            raise ValueError("Learning rates must be positive")
        encoder_parameters = [
            parameter
            for parameter in self.encoder_model.parameters()
            if parameter.requires_grad
        ]
        groups: list[dict[str, object]] = []
        if encoder_parameters:
            groups.append(
                {"params": encoder_parameters, "lr": float(encoder_learning_rate)}
            )
        groups.append(
            {
                "params": self.head.parameters(),
                "lr": float(encoder_learning_rate * head_learning_rate_multiplier),
            }
        )
        return groups

    def encoder_training_summary(self) -> dict[str, int]:
        selected = 0
        frozen = 0
        unused_auxiliary = 0
        for name, parameter in self.encoder_model.named_parameters():
            count = int(parameter.numel())
            if (
                self.model_key == "papagei_s"
                and self._is_papagei_s_auxiliary_parameter(name)
            ):
                unused_auxiliary += count
            elif parameter.requires_grad:
                selected += count
            else:
                frozen += count
        return {
            "trainable_embedding_path_parameters": selected,
            "frozen_embedding_path_parameters": frozen,
            "excluded_native_auxiliary_parameters": unused_auxiliary,
            "trainable_head_parameters": sum(p.numel() for p in self.head.parameters()),
        }
