"""Model-independent encoder interface required by the benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Literal, Mapping

import numpy as np


PaddingPolicy = Literal["zero"]


@dataclass(frozen=True)
class NativeEncoderOutput:
    """Validated native model outputs plus the component used downstream.

    Foundation encoders do not predict blood pressure directly.  In particular,
    PaPaGei returns tuples containing embeddings and auxiliary morphology
    predictions.  Keeping every component named prevents callers from treating
    an arbitrary native tensor as a scalar mmHg prediction.
    """

    embedding: np.ndarray
    components: Mapping[str, np.ndarray]
    component_semantics: Mapping[str, str]

    def component_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            name: tuple(np.asarray(value).shape)
            for name, value in self.components.items()
        }


@dataclass(frozen=True)
class EncoderFingerprint:
    contract_version: str
    model_name: str
    model_version: str
    checkpoint_sha256: str
    repository_commit: str
    preprocessing: dict[str, Any]
    input_sampling_rate_hz: int
    input_samples: int
    embedding_dimension: int

    def stable_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return sha256(self.stable_json().encode("utf-8")).hexdigest()


class PPGEncoder(ABC):
    """A narrow contract shared by every foundation-model wrapper."""

    @abstractmethod
    def preprocess(self, signal: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
        """Return one model-ready one-dimensional signal."""

    @abstractmethod
    def encode(self, batch: np.ndarray) -> NativeEncoderOutput:
        """Return all named native outputs and the selected embedding."""

    @abstractmethod
    def fingerprint(self) -> EncoderFingerprint:
        """Describe the exact checkpoint and preprocessing implementation."""

    def preprocess_for_dataset(
        self,
        signal: np.ndarray,
        sampling_rate_hz: int,
        *,
        dataset: str,
    ) -> np.ndarray:
        """Apply an optional dataset-specific released preprocessing contract.

        Most encoders use the same waveform operations for every cohort.  This
        hook keeps the ordinary :meth:`preprocess` interface intact while
        allowing a wrapper to reproduce an operation that an upstream
        downstream script applies only to one named dataset.
        """

        del dataset
        return self.preprocess(signal, sampling_rate_hz)

    def fingerprint_for_dataset(self, dataset: str) -> EncoderFingerprint:
        """Return the effective fingerprint for one homogeneous cohort."""

        del dataset
        return self.fingerprint()

    def preprocessing_shape_trace(
        self, signal: np.ndarray, sampling_rate_hz: int
    ) -> list[dict[str, Any]]:
        """Describe observed input and final shapes for diagnostic artifacts."""
        return [
            {
                "name": "raw_waveform",
                "shape": list(np.asarray(signal).shape),
                "sampling_rate_hz": int(sampling_rate_hz),
            },
            {
                "name": "model_ready",
                "shape": [self.fingerprint().input_samples],
                "sampling_rate_hz": self.fingerprint().input_sampling_rate_hz,
            },
        ]

    def preprocessing_shape_trace_for_dataset(
        self,
        signal: np.ndarray,
        sampling_rate_hz: int,
        *,
        dataset: str,
    ) -> list[dict[str, Any]]:
        """Describe the effective preprocessing path for one dataset."""

        del dataset
        return self.preprocessing_shape_trace(signal, sampling_rate_hz)

    def preprocess_checked(
        self,
        signal: np.ndarray,
        sampling_rate_hz: int,
        *,
        dataset: str | None = None,
    ) -> np.ndarray:
        preprocessing = (
            self.preprocess(signal, sampling_rate_hz)
            if dataset is None
            else self.preprocess_for_dataset(
                signal,
                sampling_rate_hz,
                dataset=dataset,
            )
        )
        processed = np.asarray(
            preprocessing, dtype=np.float32
        )
        fingerprint = (
            self.fingerprint()
            if dataset is None
            else self.fingerprint_for_dataset(dataset)
        )
        if processed.ndim != 1:
            raise ValueError(
                f"Encoder preprocessing must return one dimension, got {processed.shape}"
            )
        if len(processed) != fingerprint.input_samples:
            raise ValueError(
                f"Encoder expected {fingerprint.input_samples} samples, "
                f"received {len(processed)}"
            )
        if not np.isfinite(processed).all():
            raise ValueError("Encoder preprocessing produced non-finite samples")
        return processed

    def encode_checked(self, batch: np.ndarray) -> NativeEncoderOutput:
        batch = np.asarray(batch, dtype=np.float32)
        fingerprint = self.fingerprint()
        if batch.ndim != 2 or batch.shape[1] != fingerprint.input_samples:
            raise ValueError(
                "Encoder input must have shape "
                f"(batch, {fingerprint.input_samples}), got {batch.shape}"
            )
        result = self.encode(batch)
        if not isinstance(result, NativeEncoderOutput):
            raise TypeError(
                "Encoder must return NativeEncoderOutput; native model outputs "
                "cannot be reduced to an assumed scalar contract"
            )
        embeddings = np.asarray(result.embedding, dtype=np.float32)
        expected = (len(batch), fingerprint.embedding_dimension)
        if embeddings.shape != expected:
            raise ValueError(
                f"Encoder returned shape {embeddings.shape}, expected {expected}"
            )
        if not result.components:
            raise ValueError("Encoder returned no named native components")
        checked_components: dict[str, np.ndarray] = {}
        for name, value in result.components.items():
            array = np.asarray(value, dtype=np.float32)
            if array.ndim < 1 or array.shape[0] != len(batch):
                raise ValueError(
                    f"Native component {name!r} has incompatible shape {array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"Native component {name!r} contains non-finite values")
            checked_components[name] = array
        if not np.isfinite(embeddings).all():
            raise ValueError("Encoder returned non-finite embeddings")
        if set(result.component_semantics) != set(checked_components):
            raise ValueError(
                "Every native component must have one declared semantic meaning"
            )
        return NativeEncoderOutput(
            embedding=embeddings,
            components=checked_components,
            component_semantics=dict(result.component_semantics),
        )

    def embedding_checked(self, batch: np.ndarray) -> np.ndarray:
        """Convenience accessor for representation-learning pipelines."""
        return self.encode_checked(batch).embedding
