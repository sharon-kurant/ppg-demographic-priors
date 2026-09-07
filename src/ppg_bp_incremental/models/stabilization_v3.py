"""Optional, fixed fine-tuning stabilization utilities.

The utilities in this module are deliberately inert until explicitly enabled.
They support a sensitivity in which the native 512-dimensional embedding is
standardized with statistics computed from epoch-zero development embeddings,
and a separate sensitivity in which encoder BatchNorm running buffers are held
fixed while ordinary trainable parameters remain trainable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


DEFAULT_EMBEDDING_STANDARD_DEVIATION_FLOOR = 1e-6


def _checked_weights(
    sample_weight: Sequence[float] | np.ndarray | None,
    length: int,
) -> np.ndarray:
    weights = (
        np.ones(length, dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    )
    if (
        len(weights) != length
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("Embedding-standardization weights must be positive and aligned")
    return weights / weights.sum()


def _row_identity_sha256(rows: pd.DataFrame) -> str:
    required = {"subject_id", "segment_id"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(
            "Embedding-standardization rows are missing identity columns: "
            f"{sorted(missing)}"
        )
    stable = rows[["subject_id", "segment_id"]].astype(str).sort_values(
        ["subject_id", "segment_id"], kind="stable"
    )
    return sha256(stable.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


@dataclass(frozen=True)
class EmbeddingStandardizationStatistics:
    """Train-only, per-feature statistics from native epoch-zero embeddings."""

    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    raw_standard_deviation_minimum: float
    raw_standard_deviation_maximum: float
    standard_deviation_floor: float
    floored_feature_count: int
    fitted_row_count: int
    fitted_subject_count: int
    fitted_row_identity_sha256: str
    statistics_sha256: str
    fitting_scope: str

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        rows: pd.DataFrame,
        *,
        sample_weight: Sequence[float] | np.ndarray | None = None,
        standard_deviation_floor: float = DEFAULT_EMBEDDING_STANDARD_DEVIATION_FLOOR,
        fitting_scope: str,
    ) -> "EmbeddingStandardizationStatistics":
        values = np.asarray(embeddings, dtype=np.float64)
        floor = float(standard_deviation_floor)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError("Epoch-zero embeddings must have shape (N, D)")
        if len(rows) != values.shape[0]:
            raise ValueError("Embedding rows and epoch-zero embeddings are misaligned")
        if not np.isfinite(values).all():
            raise ValueError("Epoch-zero embeddings contain non-finite values")
        if not np.isfinite(floor) or floor <= 0:
            raise ValueError("Embedding standard-deviation floor must be positive")
        if not str(fitting_scope).strip():
            raise ValueError("Embedding-standardization fitting scope is required")

        weights = _checked_weights(sample_weight, len(values))
        means = np.sum(values * weights[:, None], axis=0)
        variances = np.sum((values - means) ** 2 * weights[:, None], axis=0)
        raw_standard_deviations = np.sqrt(np.maximum(variances, 0.0))
        standard_deviations = np.maximum(raw_standard_deviations, floor)
        if not (
            np.isfinite(means).all() and np.isfinite(standard_deviations).all()
        ):
            raise ValueError("Embedding standardization produced non-finite statistics")

        statistics_bytes = b"".join(
            (
                np.ascontiguousarray(means, dtype="<f8").tobytes(),
                np.ascontiguousarray(standard_deviations, dtype="<f8").tobytes(),
            )
        )
        return cls(
            means=tuple(float(value) for value in means),
            standard_deviations=tuple(float(value) for value in standard_deviations),
            raw_standard_deviation_minimum=float(raw_standard_deviations.min()),
            raw_standard_deviation_maximum=float(raw_standard_deviations.max()),
            standard_deviation_floor=floor,
            floored_feature_count=int(np.sum(raw_standard_deviations < floor)),
            fitted_row_count=int(len(values)),
            fitted_subject_count=int(rows["subject_id"].astype(str).nunique()),
            fitted_row_identity_sha256=_row_identity_sha256(rows),
            statistics_sha256=sha256(statistics_bytes).hexdigest(),
            fitting_scope=str(fitting_scope),
        )

    @property
    def dimension(self) -> int:
        return len(self.means)

    def metadata(self) -> dict[str, Any]:
        return {"enabled": True, "dimension": self.dimension, **asdict(self)}


class FixedEmbeddingStandardizer(nn.Module):
    """Apply fixed per-feature statistics without changing native outputs."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError("Embedding dimension must be positive")
        self.dimension = int(dimension)
        # Persist the mode with the statistics so a saved standardized model
        # cannot silently reload as an identity transform.
        self.register_buffer("_enabled", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("means", torch.zeros(self.dimension, dtype=torch.float32))
        self.register_buffer(
            "standard_deviations", torch.ones(self.dimension, dtype=torch.float32)
        )

    def configure(self, statistics: EmbeddingStandardizationStatistics) -> None:
        if statistics.dimension != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} embedding features, got "
                f"{statistics.dimension}"
            )
        means = torch.as_tensor(
            statistics.means, dtype=self.means.dtype, device=self.means.device
        )
        standard_deviations = torch.as_tensor(
            statistics.standard_deviations,
            dtype=self.standard_deviations.dtype,
            device=self.standard_deviations.device,
        )
        if not torch.isfinite(means).all() or not torch.isfinite(
            standard_deviations
        ).all():
            raise ValueError("Embedding statistics must be finite")
        if torch.any(standard_deviations <= 0):
            raise ValueError("Embedding standard deviations must be positive")
        self.means.copy_(means)
        self.standard_deviations.copy_(standard_deviations)
        self._enabled.fill_(True)

    @property
    def enabled(self) -> bool:
        return bool(self._enabled.detach().cpu().item())

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 2 or embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding shape (B, {self.dimension}), got "
                f"{tuple(embedding.shape)}"
            )
        if not self.enabled:
            return embedding
        return (embedding - self.means) / self.standard_deviations

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "dimension": self.dimension,
            "mean_preview": self.means.detach().cpu()[:5].tolist(),
            "standard_deviation_preview": (
                self.standard_deviations.detach().cpu()[:5].tolist()
            ),
        }


@dataclass(frozen=True)
class BatchNormRunningStatsSnapshot:
    """Exact copies of encoder BatchNorm buffers for a no-drift assertion."""

    buffers: Mapping[str, torch.Tensor]
    module_count: int

    @classmethod
    def capture(cls, module: nn.Module) -> "BatchNormRunningStatsSnapshot":
        buffers: dict[str, torch.Tensor] = {}
        module_count = 0
        for module_name, child in module.named_modules():
            if not isinstance(child, nn.modules.batchnorm._BatchNorm):
                continue
            module_count += 1
            for buffer_name in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(child, buffer_name, None)
                if value is not None:
                    key = f"{module_name}.{buffer_name}" if module_name else buffer_name
                    buffers[key] = value.detach().cpu().clone()
        return cls(buffers=buffers, module_count=module_count)

    def compare(self, module: nn.Module) -> dict[str, Any]:
        current = self.capture(module)
        if set(current.buffers) != set(self.buffers):
            raise RuntimeError("Encoder BatchNorm running-buffer structure changed")
        changed: list[str] = []
        maximum_absolute_change = 0.0
        for name, before in self.buffers.items():
            after = current.buffers[name]
            if before.shape != after.shape:
                raise RuntimeError(f"BatchNorm running buffer {name!r} changed shape")
            if before.dtype.is_floating_point:
                difference = float(torch.max(torch.abs(after - before))) if before.numel() else 0.0
            else:
                difference = float(torch.max(torch.abs(after.long() - before.long()))) if before.numel() else 0.0
            maximum_absolute_change = max(maximum_absolute_change, difference)
            if not torch.equal(before, after):
                changed.append(name)
        return {
            "module_count": self.module_count,
            "running_buffer_count": len(self.buffers),
            "running_buffers_unchanged": not changed,
            "changed_running_buffers": changed,
            "maximum_absolute_change": maximum_absolute_change,
        }


def freeze_batchnorm_running_stats(module: nn.Module) -> dict[str, int]:
    """Put only BatchNorm modules in eval mode; do not freeze parameters."""

    module_count = 0
    affine_parameter_count = 0
    trainable_affine_parameter_count = 0
    for child in module.modules():
        if not isinstance(child, nn.modules.batchnorm._BatchNorm):
            continue
        module_count += 1
        child.eval()
        for parameter in (child.weight, child.bias):
            if parameter is None:
                continue
            affine_parameter_count += int(parameter.numel())
            if parameter.requires_grad:
                trainable_affine_parameter_count += int(parameter.numel())
    return {
        "module_count": module_count,
        "affine_parameter_count": affine_parameter_count,
        "trainable_affine_parameter_count": trainable_affine_parameter_count,
    }


def zero_initialize_scalar_output_layer(layer: nn.Linear) -> None:
    """Initialize a scalar regression layer at the training-target mean.

    The BP head predicts a training-pool z-score, so zero weight and bias make
    its epoch-zero decoded prediction exactly the training-pool BP mean.
    """

    if not isinstance(layer, nn.Linear) or layer.out_features != 1:
        raise TypeError("Zero-output initialization requires a scalar Linear layer")
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


__all__ = [
    "BatchNormRunningStatsSnapshot",
    "DEFAULT_EMBEDDING_STANDARD_DEVIATION_FLOOR",
    "EmbeddingStandardizationStatistics",
    "FixedEmbeddingStandardizer",
    "freeze_batchnorm_running_stats",
    "zero_initialize_scalar_output_layer",
]
