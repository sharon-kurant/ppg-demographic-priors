"""Source-faithful, subject-disjoint end-to-end BP fine-tuning.

The frozen encoder wrappers own preprocessing and checkpoint construction.  A
run in this module performs model selection on the locked outer-training and
validation subjects, then initializes a fresh checkpoint and refits for the
selected number of epochs on the complete development pool.  Outer-test
subjects remain untouched until the final prediction pass.

The neural estimator is a standardized benchmark adaptation for every model;
it is not presented as released upstream BP code.  Its scalar output is a
training-pool BP z-score and is explicitly decoded to mmHg.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import random
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ppg_bp_incremental.data.benchmark import (
    available_demographic_fields,
    capped_subject_rows,
    load_record_waveform,
)
from ppg_bp_incremental.data.benchmark_splits import rows_for_role
from ppg_bp_incremental.models.bp_head_v3 import FoundationBPRegressorV3
from ppg_bp_incremental.models.contracts import CONTRACT_VERSION
from ppg_bp_incremental.models.registry import create_encoder, overlap_status
from ppg_bp_incremental.models.stabilization_v3 import (
    BatchNormRunningStatsSnapshot,
    DEFAULT_EMBEDDING_STANDARD_DEVIATION_FLOOR,
    EmbeddingStandardizationStatistics,
    freeze_batchnorm_running_stats,
)
from ppg_bp_incremental.training.ridge_utils import participant_measurement_weights


TargetFitPoolRole = Literal[
    "outer_training_selection",
    "full_development_refit",
]
TARGET_OUTPUT_SCALE_BY_FIT_ROLE: dict[TargetFitPoolRole, str] = {
    "outer_training_selection": "outer_training_pool_zscore",
    "full_development_refit": "full_development_pool_zscore",
}
TARGET_DECODER_BY_FIT_ROLE: dict[TargetFitPoolRole, str] = {
    "outer_training_selection": "affine_outer_training_pool_inverse_zscore",
    "full_development_refit": "affine_full_development_pool_inverse_zscore",
}


ENCODER_LEARNING_RATES = (1e-5, 3e-5, 1e-4)
DEFAULT_MAX_EPOCHS = 10
DEFAULT_PATIENCE = 3
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAXIMUM_PULSEDB_SEGMENTS_PER_SUBJECT = 20
EVALUATION_SEGMENT_SEED = 20260715


def _checked_sample_weight(
    sample_weight: Sequence[float] | np.ndarray | None,
    length: int,
) -> np.ndarray:
    weights = (
        np.ones(length, dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float).reshape(-1)
    )
    if (
        len(weights) != length
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("Sample weights must be positive, finite, and row-aligned")
    return weights * (length / weights.sum())


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = _checked_sample_weight(weights, len(values))
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, weights.sum() / 2, side="left"))
    return float(values[order[min(index, len(order) - 1)]])


@dataclass(frozen=True)
class FineTuneConfig:
    """Compute and optimization choices for the time-limited correction run."""

    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    gradient_clip_norm: float = 1.0
    head_learning_rate_multiplier: float = 10.0
    learning_rates: tuple[float, ...] = ENCODER_LEARNING_RATES
    maximum_pulsedb_segments_per_subject: int = (
        DEFAULT_MAXIMUM_PULSEDB_SEGMENTS_PER_SUBJECT
    )
    mixed_precision: bool = True
    encoder_trainable: bool = True
    num_workers: int = 0
    keep_final_checkpoint: bool = False
    standardize_embedding: bool = False
    freeze_batchnorm_running_stats: bool = False
    zero_initialize_output_layer: bool = False
    embedding_standard_deviation_floor: float = (
        DEFAULT_EMBEDDING_STANDARD_DEVIATION_FLOOR
    )

    def checked(self) -> "FineTuneConfig":
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 1 <= self.max_epochs <= DEFAULT_MAX_EPOCHS:
            raise ValueError(
                f"Correction runs are capped at {DEFAULT_MAX_EPOCHS} epochs"
            )
        if not 1 <= self.patience <= self.max_epochs:
            raise ValueError("patience must be between one and max_epochs")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.head_learning_rate_multiplier <= 0:
            raise ValueError("head_learning_rate_multiplier must be positive")
        if not self.learning_rates or any(rate <= 0 for rate in self.learning_rates):
            raise ValueError("At least one positive encoder learning rate is required")
        if self.maximum_pulsedb_segments_per_subject < 1:
            raise ValueError("PulseDB subject cap must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if (
            not np.isfinite(self.embedding_standard_deviation_floor)
            or self.embedding_standard_deviation_floor <= 0
        ):
            raise ValueError("embedding_standard_deviation_floor must be positive")
        return self


@dataclass(frozen=True)
class TargetZScoreTransform:
    """A target transform fitted only on the stated development rows."""

    mean_mmhg: float
    standard_deviation_mmhg: float
    fitted_row_count: int
    fitted_subject_sha256: str

    @classmethod
    def fit(
        cls,
        data: pd.DataFrame,
        target: str,
        sample_weight: Sequence[float] | np.ndarray | None = None,
    ) -> "TargetZScoreTransform":
        if target not in {"sbp", "dbp"}:
            raise ValueError("target must be sbp or dbp")
        values = pd.to_numeric(data[target], errors="coerce").to_numpy(dtype=float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("Target-transform fitting requires finite BP labels")
        weights = _checked_sample_weight(sample_weight, len(values))
        mean = float(np.average(values, weights=weights))
        standard_deviation = float(
            np.sqrt(np.average((values - mean) ** 2, weights=weights))
        )
        if not standard_deviation > 1e-8:
            raise ValueError("BP target has zero variance in the fitting pool")
        subjects = sorted(set(data["subject_id"].astype(str)))
        return cls(
            mean_mmhg=mean,
            standard_deviation_mmhg=standard_deviation,
            fitted_row_count=len(values),
            fitted_subject_sha256=sha256("\n".join(subjects).encode()).hexdigest(),
        )

    def encode(self, values_mmhg: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(values_mmhg, dtype=np.float32)
        output = (values - self.mean_mmhg) / self.standard_deviation_mmhg
        if not np.isfinite(output).all():
            raise ValueError("Target z-scoring produced non-finite values")
        return output.astype(np.float32)

    def decode(self, values_zscore: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(values_zscore, dtype=np.float32)
        output = values * self.standard_deviation_mmhg + self.mean_mmhg
        if not np.isfinite(output).all():
            raise ValueError("Target inverse z-scoring produced non-finite values")
        return output.astype(np.float32)

    def metadata(self, fit_pool_role: TargetFitPoolRole) -> dict[str, Any]:
        """Describe the fitted transform without obscuring its CV role."""

        return {
            **asdict(self),
            "fit_pool_role": fit_pool_role,
            "estimator_output_scale": TARGET_OUTPUT_SCALE_BY_FIT_ROLE[fit_pool_role],
            "final_output_scale": "raw_mmhg",
            "decoder": TARGET_DECODER_BY_FIT_ROLE[fit_pool_role],
        }


@dataclass(frozen=True)
class DemographicTransformV3:
    """Train-only age/sex/BMI transform with explicit missingness handling."""

    fields: tuple[str, ...]
    numeric_medians: dict[str, float]
    numeric_means: dict[str, float]
    numeric_standard_deviations: dict[str, float]
    missing_indicator_fields: tuple[str, ...]
    feature_names: tuple[str, ...]
    fitted_subject_sha256: str

    @classmethod
    def fit(
        cls,
        data: pd.DataFrame,
        fields: Sequence[str],
        *,
        missing_indicator_fields: Sequence[str] = (),
        sample_weight: Sequence[float] | np.ndarray | None = None,
    ) -> "DemographicTransformV3":
        fields = tuple(fields)
        unexpected = set(fields).difference({"age", "sex", "bmi"})
        if unexpected:
            raise ValueError(f"Unsupported demographic fields: {sorted(unexpected)}")
        if not fields:
            raise ValueError("At least one demographic field is required")
        indicators = tuple(
            field
            for field in ("age", "bmi")
            if field in fields and field in set(missing_indicator_fields)
        )
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        standard_deviations: dict[str, float] = {}
        feature_names: list[str] = []
        weights = _checked_sample_weight(sample_weight, len(data))
        for field in ("age", "bmi"):
            if field not in fields:
                continue
            values = pd.to_numeric(data[field], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if not len(finite):
                raise ValueError(f"Training demographic {field!r} is entirely missing")
            finite_mask = np.isfinite(values)
            median = _weighted_median(values[finite_mask], weights[finite_mask])
            imputed = np.where(np.isfinite(values), values, median)
            mean = float(np.average(imputed, weights=weights))
            standard_deviation = float(
                np.sqrt(np.average((imputed - mean) ** 2, weights=weights))
            )
            if not np.isfinite(standard_deviation) or standard_deviation <= 1e-8:
                standard_deviation = 1.0
            medians[field] = median
            means[field] = mean
            standard_deviations[field] = standard_deviation
            feature_names.append(f"{field}_zscore")
            if field in indicators:
                feature_names.append(f"{field}_missing")
        if "sex" in fields:
            feature_names.extend(("sex_female", "sex_male", "sex_unknown"))
        subjects = sorted(set(data["subject_id"].astype(str)))
        return cls(
            fields=fields,
            numeric_medians=medians,
            numeric_means=means,
            numeric_standard_deviations=standard_deviations,
            missing_indicator_fields=indicators,
            feature_names=tuple(feature_names),
            fitted_subject_sha256=sha256("\n".join(subjects).encode()).hexdigest(),
        )

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for field in ("age", "bmi"):
            if field not in self.fields:
                continue
            values = pd.to_numeric(data[field], errors="coerce").to_numpy(dtype=float)
            missing = ~np.isfinite(values)
            imputed = np.where(missing, self.numeric_medians[field], values)
            standardized = (
                imputed - self.numeric_means[field]
            ) / self.numeric_standard_deviations[field]
            columns.append(standardized)
            if field in self.missing_indicator_fields:
                columns.append(missing.astype(float))
        if "sex" in self.fields:
            sex = data["sex"].fillna("unknown").astype(str).str.strip().str.lower()
            sex = sex.where(sex.isin(("female", "male")), "unknown")
            columns.extend(
                (
                    sex.eq("female").to_numpy(dtype=float),
                    sex.eq("male").to_numpy(dtype=float),
                    sex.eq("unknown").to_numpy(dtype=float),
                )
            )
        output = np.column_stack(columns).astype(np.float32)
        if output.shape != (len(data), len(self.feature_names)):
            raise RuntimeError("Unexpected demographic design-matrix shape")
        if not np.isfinite(output).all():
            raise ValueError("Demographic transform produced non-finite values")
        return output

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedWaveforms:
    frame: pd.DataFrame
    waveforms: np.ndarray
    preprocessing_diagnostics: dict[str, Any]

    def checked(self) -> "PreparedWaveforms":
        if self.waveforms.ndim != 2 or len(self.waveforms) != len(self.frame):
            raise ValueError("Prepared waveform table and tensor are misaligned")
        if not np.isfinite(self.waveforms).all():
            raise ValueError("Prepared waveforms contain non-finite values")
        return self


def prepare_waveforms(data: pd.DataFrame, encoder: Any) -> PreparedWaveforms:
    """Apply the audited model preprocessing once, before repeated epochs."""

    frame = data.reset_index(drop=True).copy()
    datasets = frame["dataset"].astype(str).unique()
    if len(datasets) != 1:
        raise ValueError("A fine-tuning run must contain exactly one cohort")
    dataset = str(datasets[0])
    fingerprint = encoder.fingerprint_for_dataset(dataset)
    waveforms = np.empty(
        (len(frame), int(fingerprint.input_samples)), dtype=np.float32
    )
    diagnostics: dict[str, Any] = {}
    for position, (_, row) in enumerate(frame.iterrows()):
        raw = load_record_waveform(row)
        waveforms[position] = encoder.preprocess_checked(
            raw,
            int(row["sample_rate_hz"]),
            dataset=dataset,
        )
        if position == 0:
            diagnostics = {
                "segment_id": str(row["segment_id"]),
                "raw_shape": list(np.asarray(raw).shape),
                "raw_sampling_rate_hz": int(row["sample_rate_hz"]),
                "raw_preview": np.asarray(raw).reshape(-1)[:5].tolist(),
                "preprocessing_steps": encoder.preprocessing_shape_trace_for_dataset(
                    raw,
                    int(row["sample_rate_hz"]),
                    dataset=dataset,
                ),
                "model_ready_record_shape": list(waveforms[position].shape),
                "model_ready_preview": waveforms[position, :5].tolist(),
                "encoder_fingerprint": asdict(fingerprint),
            }
    return PreparedWaveforms(frame, waveforms, diagnostics).checked()


class _PreparedDataset(Dataset):
    def __init__(
        self,
        prepared: PreparedWaveforms,
        positions: np.ndarray,
        target_zscore: np.ndarray,
        demographics: np.ndarray,
        sample_weight: np.ndarray,
    ) -> None:
        positions = np.asarray(positions, dtype=np.int64)
        if positions.ndim != 1 or not len(positions):
            raise ValueError("Dataset positions must be a nonempty vector")
        if target_zscore.shape != (len(prepared.frame),):
            raise ValueError("Target vector does not align with prepared waveforms")
        if demographics.ndim != 2 or demographics.shape[0] != len(prepared.frame):
            raise ValueError("Demographics do not align with prepared waveforms")
        if sample_weight.shape != (len(prepared.frame),):
            raise ValueError("Sample weights do not align with prepared waveforms")
        self.waveforms = torch.from_numpy(prepared.waveforms)
        self.positions = torch.from_numpy(positions)
        self.target_zscore = torch.from_numpy(target_zscore.astype(np.float32))
        self.demographics = torch.from_numpy(demographics.astype(np.float32))
        self.sample_weight = torch.from_numpy(sample_weight.astype(np.float32))

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int):
        position = self.positions[index]
        return (
            self.waveforms[position].unsqueeze(0),
            self.target_zscore[position].reshape(1),
            self.demographics[position],
            self.sample_weight[position].reshape(1),
            position,
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _loader(
    prepared: PreparedWaveforms,
    positions: np.ndarray,
    target_zscore: np.ndarray,
    demographics: np.ndarray,
    sample_weight: np.ndarray | None = None,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    weights = (
        np.ones(len(prepared.frame), dtype=np.float32)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32)
    )
    return DataLoader(
        _PreparedDataset(
            prepared, positions, target_zscore, demographics, weights
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _positions(frame: pd.DataFrame, selected: pd.DataFrame) -> np.ndarray:
    identifiers = set(selected["segment_id"].astype(str))
    output = np.flatnonzero(frame["segment_id"].astype(str).isin(identifiers).to_numpy())
    if len(output) != len(selected):
        raise ValueError("Split rows could not be aligned by unique segment_id")
    return output.astype(np.int64)


def _capped_positions(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    *,
    maximum_segments: int,
    seed: int,
    epoch: int | None,
) -> np.ndarray:
    selected = capped_subject_rows(
        source,
        maximum_segments=maximum_segments,
        seed=seed,
        epoch=epoch,
    )
    return _positions(frame, selected)


def _participant_balanced_weight_vector(
    frame: pd.DataFrame, positions: np.ndarray
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    output = np.ones(len(frame), dtype=np.float32)
    output[positions] = participant_measurement_weights(
        frame.iloc[positions]
    ).astype(np.float32)
    return output


def _parameter_sample(model: FoundationBPRegressorV3) -> dict[str, torch.Tensor]:
    """Small deterministic snapshots used to prove encoder weights changed."""

    sample: dict[str, torch.Tensor] = {}
    for name, parameter in model.encoder_model.named_parameters():
        if parameter.requires_grad:
            sample[name] = parameter.detach().reshape(-1)[:32].cpu().clone()
    return sample


def _parameter_change_audit(
    before: dict[str, torch.Tensor],
    model: FoundationBPRegressorV3,
) -> dict[str, Any]:
    changed_names: list[str] = []
    maximum_absolute_change = 0.0
    current = dict(model.encoder_model.named_parameters())
    for name, old in before.items():
        new = current[name].detach().reshape(-1)[: len(old)].cpu()
        difference = float(torch.max(torch.abs(new - old))) if len(old) else 0.0
        maximum_absolute_change = max(maximum_absolute_change, difference)
        if difference > 0:
            changed_names.append(name)
    return {
        "sampled_parameter_tensors": len(before),
        "changed_sampled_parameter_tensors": len(changed_names),
        "changed_parameter_name_preview": changed_names[:10],
        "maximum_sampled_absolute_change": maximum_absolute_change,
    }


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json_save(payload: Mapping[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return sha256(encoded).hexdigest()


def _native_diagnostics(
    model: FoundationBPRegressorV3,
    loader: DataLoader,
    device: torch.device,
    use_demographics: bool,
) -> dict[str, Any]:
    model.eval()
    waveform, target_zscore, demographics, _, positions = next(iter(loader))
    waveform = waveform.to(device)
    demographics_device = demographics.to(device) if use_demographics else None
    with torch.inference_mode():
        output = model(waveform, demographics_device)
        head_embedding = model.embedding_for_head(output.native.embedding)
    return {
        "final_model_input_shape": list(waveform.shape),
        "native_output_shapes": {
            name: list(value.shape) for name, value in output.native.components.items()
        },
        "native_output_previews": {
            name: value[0].detach().cpu().reshape(-1)[:5].tolist()
            for name, value in output.native.components.items()
        },
        "selected_embedding_shape": list(output.native.embedding.shape),
        "selected_embedding_preview": (
            output.native.embedding[0].detach().cpu().reshape(-1)[:5].tolist()
        ),
        "bp_head_embedding_shape": list(head_embedding.shape),
        "bp_head_embedding_preview": (
            head_embedding[0].detach().cpu().reshape(-1)[:5].tolist()
        ),
        "stabilization": model.stabilization_summary(),
        "demographic_input_shape": list(demographics.shape) if use_demographics else None,
        "bp_head_output_zscore_shape": list(output.prediction_zscore.shape),
        "bp_head_output_zscore_preview": (
            output.prediction_zscore.detach().cpu().reshape(-1)[:5].tolist()
        ),
        "decoded_bp_output_mmhg_shape": list(output.prediction_mmhg.shape),
        "decoded_bp_output_mmhg_preview": (
            output.prediction_mmhg.detach().cpu().reshape(-1)[:5].tolist()
        ),
        "target_zscore_preview": target_zscore.reshape(-1)[:5].tolist(),
        "row_position_preview": positions.reshape(-1)[:5].tolist(),
        "decoder": dict(output.decoder_metadata),
    }


def _cohort_mae_mmhg(
    evaluation_rows: pd.DataFrame,
    target: str,
    prediction_mmhg: Sequence[float] | np.ndarray,
) -> float:
    """Apply the benchmark's cohort-specific unit of analysis."""

    prediction = np.asarray(prediction_mmhg, dtype=float).reshape(-1)
    if len(evaluation_rows) != len(prediction) or not len(prediction):
        raise ValueError("Evaluation rows and predictions must be nonempty and aligned")
    if not np.isfinite(prediction).all():
        raise ValueError("Evaluation predictions contain non-finite values")
    required = {"dataset", "subject_id", "measurement_id", target}
    missing = required.difference(evaluation_rows.columns)
    if missing:
        raise ValueError(f"Evaluation rows are missing columns: {sorted(missing)}")
    evaluation = evaluation_rows[
        ["dataset", "subject_id", "measurement_id", target]
    ].reset_index(drop=True).copy()
    evaluation["truth"] = pd.to_numeric(evaluation[target], errors="raise")
    evaluation["prediction"] = prediction
    dataset = str(evaluation["dataset"].iloc[0])
    if dataset == "PPG-BP":
        # The repeated waveforms share one cuff label: average predictions first.
        participants = evaluation.groupby("subject_id", as_index=False).agg(
            truth=("truth", "mean"), prediction=("prediction", "mean")
        )
        return float(
            np.mean(np.abs(participants["prediction"] - participants["truth"]))
        )
    if dataset.startswith("PulseDB"):
        # Multiple segments may represent one measurement. Average their
        # predictions, calculate measurement error, then weight subjects equally.
        measurements = evaluation.groupby(
            ["subject_id", "measurement_id"], as_index=False
        ).agg(truth=("truth", "mean"), prediction=("prediction", "mean"))
        measurements["absolute_error"] = np.abs(
            measurements["prediction"] - measurements["truth"]
        )
        return float(
            measurements.groupby("subject_id")["absolute_error"].mean().mean()
        )
    return float(np.mean(np.abs(evaluation["prediction"] - evaluation["truth"])))


def _evaluate(
    model: FoundationBPRegressorV3,
    loader: DataLoader,
    device: torch.device,
    use_demographics: bool,
    frame: pd.DataFrame,
    target: str,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions_zscore: list[np.ndarray] = []
    predictions_mmhg: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    with torch.inference_mode():
        for waveform, target_zscore, demographics, _, position in loader:
            waveform = waveform.to(device, non_blocking=True)
            demographic_input = (
                demographics.to(device, non_blocking=True)
                if use_demographics
                else None
            )
            output = model(waveform, demographic_input)
            predictions_zscore.append(
                output.prediction_zscore.detach().cpu().numpy().reshape(-1)
            )
            predictions_mmhg.append(
                output.prediction_mmhg.detach().cpu().numpy().reshape(-1)
            )
            positions.append(position.numpy().reshape(-1))
    prediction_z = np.concatenate(predictions_zscore)
    prediction_mmhg = np.concatenate(predictions_mmhg)
    ordered_positions = np.concatenate(positions).astype(np.int64)
    mae_mmhg = _cohort_mae_mmhg(
        frame.iloc[ordered_positions], target, prediction_mmhg
    )
    return mae_mmhg, prediction_z, prediction_mmhg, ordered_positions


@dataclass
class _FitResult:
    model: FoundationBPRegressorV3
    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_mae_mmhg: float | None
    native_diagnostics: dict[str, Any]
    weight_change_audit: dict[str, Any]
    maximum_gradient_norm: float
    embedding_standardization: dict[str, Any]
    batchnorm_running_stats_audit: dict[str, Any]
    batchnorm_training_mode: dict[str, int]


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older supported torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, enabled: bool):
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except AttributeError:  # pragma: no cover - older supported torch
        return torch.cuda.amp.autocast(enabled=enabled)


def _fit_epoch_zero_embedding_standardization(
    model: FoundationBPRegressorV3,
    prepared: PreparedWaveforms,
    fitting_rows: pd.DataFrame,
    *,
    device: torch.device,
    batch_size: int,
    standard_deviation_floor: float,
    fitting_scope: str,
) -> EmbeddingStandardizationStatistics:
    """Fit fixed head-input statistics from epoch-zero native embeddings."""

    positions = _positions(prepared.frame, fitting_rows)
    ordered_rows = prepared.frame.iloc[positions].reset_index(drop=True)
    embeddings: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start : start + batch_size]
            waveform = torch.from_numpy(
                prepared.waveforms[batch_positions]
            ).unsqueeze(1).to(device)
            native = model.native_forward(waveform)
            embeddings.append(native.embedding.detach().cpu().numpy())
    values = np.concatenate(embeddings, axis=0)
    weights = participant_measurement_weights(ordered_rows)
    statistics = EmbeddingStandardizationStatistics.fit(
        values,
        ordered_rows,
        sample_weight=weights,
        standard_deviation_floor=standard_deviation_floor,
        fitting_scope=fitting_scope,
    )
    if statistics.dimension != model.embedding_dimension:
        raise RuntimeError(
            "Epoch-zero embedding statistics disagree with the audited dimension"
        )
    model.configure_embedding_standardization(statistics)
    return statistics


def _embedding_standardization_metadata(
    statistics: EmbeddingStandardizationStatistics | None,
    *,
    standard_deviation_floor: float,
    fitting_scope: str,
) -> dict[str, Any]:
    if statistics is not None:
        return statistics.metadata()
    return {
        "enabled": False,
        "dimension": 512,
        "standard_deviation_floor": float(standard_deviation_floor),
        "fitting_scope": fitting_scope,
    }


def _train_model(
    *,
    model_key: str,
    encoder_factory: Callable[[str, str], Any],
    prepared: PreparedWaveforms,
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame | None,
    embedding_standardization_rows: pd.DataFrame,
    embedding_standardization_scope: str,
    target: Literal["sbp", "dbp"],
    target_transform: TargetZScoreTransform,
    target_fit_pool_role: TargetFitPoolRole,
    demographic_transform: DemographicTransformV3 | None,
    encoder_learning_rate: float,
    seed: int,
    device: torch.device,
    config: FineTuneConfig,
    epochs: int,
    checkpoint_path: Path,
) -> _FitResult:
    _set_seed(seed)
    encoder = encoder_factory(model_key, str(device))
    demographics = (
        demographic_transform.transform(prepared.frame)
        if demographic_transform is not None
        else np.empty((len(prepared.frame), 0), dtype=np.float32)
    )
    target_zscore = target_transform.encode(
        pd.to_numeric(prepared.frame[target], errors="raise").to_numpy(float)
    )
    model = FoundationBPRegressorV3(
        encoder,
        model_key,
        target_mean_mmhg=target_transform.mean_mmhg,
        target_standard_deviation_mmhg=target_transform.standard_deviation_mmhg,
        target_fit_pool_role=target_fit_pool_role,
        use_demographics=demographic_transform is not None,
        demographic_dimension=demographics.shape[1],
        zero_initialize_output_layer=config.zero_initialize_output_layer,
    ).to(device)
    model.set_encoder_trainable(config.encoder_trainable)
    embedding_statistics = (
        _fit_epoch_zero_embedding_standardization(
            model,
            prepared,
            embedding_standardization_rows,
            device=device,
            batch_size=config.batch_size,
            standard_deviation_floor=(
                config.embedding_standard_deviation_floor
            ),
            fitting_scope=embedding_standardization_scope,
        )
        if config.standardize_embedding
        else None
    )
    embedding_standardization = _embedding_standardization_metadata(
        embedding_statistics,
        standard_deviation_floor=config.embedding_standard_deviation_floor,
        fitting_scope=embedding_standardization_scope,
    )
    optimizer = torch.optim.Adam(
        model.optimizer_groups(
            encoder_learning_rate,
            head_learning_rate_multiplier=config.head_learning_rate_multiplier,
        )
    )
    use_amp = bool(config.mixed_precision and device.type == "cuda")
    scaler = _make_scaler(use_amp)
    loss_function = torch.nn.MSELoss(reduction="none")
    before = _parameter_sample(model)
    batchnorm_snapshot = BatchNormRunningStatsSnapshot.capture(model.encoder_model)
    batchnorm_training_mode = {
        "module_count": batchnorm_snapshot.module_count,
        "affine_parameter_count": 0,
        "trainable_affine_parameter_count": 0,
    }

    checkpoint_identity = {
        "model_key": model_key,
        "target": target,
        "seed": seed,
        "encoder_learning_rate": encoder_learning_rate,
        "target_transform": target_transform.metadata(target_fit_pool_role),
        "demographic_features": (
            list(demographic_transform.feature_names)
            if demographic_transform is not None
            else []
        ),
        "stabilization": {
            "standardize_embedding": bool(config.standardize_embedding),
            "freeze_batchnorm_running_stats": bool(
                config.freeze_batchnorm_running_stats
            ),
            "zero_initialize_output_layer": bool(
                config.zero_initialize_output_layer
            ),
            "embedding_standard_deviation_floor": float(
                config.embedding_standard_deviation_floor
            ),
            "embedding_statistics_sha256": (
                embedding_statistics.statistics_sha256
                if embedding_statistics is not None
                else None
            ),
        },
    }

    fixed_validation_positions: np.ndarray | None = None
    validation_loader: DataLoader | None = None
    if validation_rows is not None:
        if str(prepared.frame["dataset"].iloc[0]).startswith("PulseDB"):
            fixed_validation_positions = _capped_positions(
                prepared.frame,
                validation_rows,
                maximum_segments=config.maximum_pulsedb_segments_per_subject,
                seed=EVALUATION_SEGMENT_SEED,
                epoch=None,
            )
        else:
            fixed_validation_positions = _positions(prepared.frame, validation_rows)
        validation_loader = _loader(
            prepared,
            fixed_validation_positions,
            target_zscore,
            demographics,
            batch_size=config.batch_size,
            shuffle=False,
            seed=seed,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )

    diagnostic_rows = validation_rows if validation_rows is not None else train_rows
    diagnostic_positions = _positions(prepared.frame, diagnostic_rows)
    diagnostic_loader = _loader(
        prepared,
        diagnostic_positions,
        target_zscore,
        demographics,
        batch_size=config.batch_size,
        shuffle=False,
        seed=seed,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    diagnostics = _native_diagnostics(
        model,
        diagnostic_loader,
        device,
        demographic_transform is not None,
    )

    history: list[dict[str, float | int]] = []
    best_epoch = 1
    best_validation_mae = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    maximum_gradient_norm = 0.0
    completed_epochs = 0

    if checkpoint_path.exists():
        try:
            state = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:  # PyTorch 1.12 in the CUDA 11.3 cluster environment
            state = torch.load(checkpoint_path, map_location="cpu")
        if state.get("identity") != checkpoint_identity:
            raise ValueError(f"Refusing incompatible resume checkpoint {checkpoint_path}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history = list(state["history"])
        best_epoch = int(state["best_epoch"])
        best_validation_mae = float(state["best_validation_mae"])
        best_state = state["best_state"]
        stale_epochs = int(state["stale_epochs"])
        maximum_gradient_norm = float(state["maximum_gradient_norm"])
        completed_epochs = int(state["completed_epochs"])

    for epoch in range(completed_epochs + 1, epochs + 1):
        if str(prepared.frame["dataset"].iloc[0]).startswith("PulseDB"):
            train_positions = _capped_positions(
                prepared.frame,
                train_rows,
                maximum_segments=config.maximum_pulsedb_segments_per_subject,
                seed=seed,
                epoch=epoch,
            )
        else:
            train_positions = _positions(prepared.frame, train_rows)
        train_loader = _loader(
            prepared,
            train_positions,
            target_zscore,
            demographics,
            _participant_balanced_weight_vector(prepared.frame, train_positions),
            batch_size=config.batch_size,
            shuffle=True,
            seed=seed + epoch,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        model.train()
        if config.freeze_batchnorm_running_stats:
            batchnorm_training_mode = freeze_batchnorm_running_stats(
                model.encoder_model
            )
        loss_sum = 0.0
        training_weight_sum = 0.0
        observation_count = 0
        epoch_maximum_gradient_norm = 0.0
        for waveform, target_batch, demographic_batch, sample_weight, _ in train_loader:
            waveform = waveform.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            demographic_input = (
                demographic_batch.to(device, non_blocking=True)
                if demographic_transform is not None
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                output = model(waveform, demographic_input)
                squared_error = loss_function(
                    output.prediction_zscore, target_batch
                )
                # participant_measurement_weights has global mean one, so this
                # is an unbiased mini-batch estimate of the balanced objective.
                weighted_squared_error = torch.sum(squared_error * sample_weight)
                batch_weight_sum = torch.sum(sample_weight)
                loss = torch.mean(squared_error * sample_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                ).detach().cpu()
            )
            if not np.isfinite(gradient_norm):
                raise RuntimeError("Fine-tuning produced a non-finite gradient norm")
            epoch_maximum_gradient_norm = max(
                epoch_maximum_gradient_norm, gradient_norm
            )
            scaler.step(optimizer)
            scaler.update()
            batch_count = len(waveform)
            loss_sum += float(weighted_squared_error.detach().cpu())
            training_weight_sum += float(batch_weight_sum.detach().cpu())
            observation_count += batch_count
        maximum_gradient_norm = max(maximum_gradient_norm, epoch_maximum_gradient_norm)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "training_zscore_mse": loss_sum / training_weight_sum,
            "training_rows": observation_count,
            "training_weight_sum": training_weight_sum,
            "maximum_preclip_gradient_norm": epoch_maximum_gradient_norm,
        }
        if validation_loader is not None:
            validation_mae, _, _, _ = _evaluate(
                model,
                validation_loader,
                device,
                demographic_transform is not None,
                prepared.frame,
                target,
            )
            record["validation_mae_mmhg"] = validation_mae
            if validation_mae < best_validation_mae:
                best_validation_mae = validation_mae
                best_epoch = epoch
                best_state = _cpu_state_dict(model)
                stale_epochs = 0
            else:
                stale_epochs += 1
        else:
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
        history.append(record)
        _atomic_torch_save(
            {
                "identity": checkpoint_identity,
                "model": _cpu_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                "best_epoch": best_epoch,
                "best_validation_mae": best_validation_mae,
                "best_state": best_state,
                "stale_epochs": stale_epochs,
                "maximum_gradient_norm": maximum_gradient_norm,
                "completed_epochs": epoch,
            },
            checkpoint_path,
        )
        if validation_loader is not None and stale_epochs >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Fine-tuning completed without a model state")
    model.load_state_dict(best_state)
    change_audit = _parameter_change_audit(before, model)
    if config.encoder_trainable:
        if maximum_gradient_norm <= 0:
            raise RuntimeError("No gradients reached the trainable model")
        if change_audit["changed_sampled_parameter_tensors"] == 0:
            raise RuntimeError("Encoder fine-tuning did not change sampled encoder weights")
    batchnorm_audit = batchnorm_snapshot.compare(model.encoder_model)
    if (
        config.freeze_batchnorm_running_stats
        and not batchnorm_audit["running_buffers_unchanged"]
    ):
        raise RuntimeError(
            "Encoder BatchNorm running buffers changed despite frozen-stat mode"
        )
    checkpoint_path.unlink(missing_ok=True)
    return _FitResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_mae_mmhg=(
            best_validation_mae if validation_loader is not None else None
        ),
        native_diagnostics=diagnostics,
        weight_change_audit=change_audit,
        maximum_gradient_norm=maximum_gradient_norm,
        embedding_standardization=embedding_standardization,
        batchnorm_running_stats_audit=batchnorm_audit,
        batchnorm_training_mode=batchnorm_training_mode,
    )


def _factory(model_key: str, device: str):
    return create_encoder(model_key, device=device)


def selection_artifact_path(
    output_root: str | Path,
    *,
    cohort: str,
    model_key: str,
    condition: str,
    target: str,
    fold: int,
) -> Path:
    """Stable cross-seed location for seed-17 LR/epoch selection."""

    return (
        Path(output_root)
        / cohort.lower().replace(" ", "_").replace("-", "_")
        / model_key
        / condition
        / target
        / f"fold_{fold}"
        / "selection.json"
    )


def _load_reusable_selection(
    source: str | Path | Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    *,
    maximum_epoch: int,
) -> tuple[dict[str, Any], str | None, str]:
    if isinstance(source, Mapping):
        artifact = dict(source)
        source_path = None
        encoded = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
    else:
        path = Path(source)
        encoded = path.read_bytes()
        artifact = json.loads(encoded)
        source_path = str(path)
    if artifact.get("schema_version") != "finetuning-v3-selection":
        raise ValueError("Reusable selection artifact has an unsupported schema")
    if _normalized_selection_identity(artifact.get("identity")) != (
        _normalized_selection_identity(expected_identity)
    ):
        raise ValueError("Reusable selection artifact does not match this run")
    if int(artifact.get("selection_seed", -1)) != 17:
        raise ValueError("Cross-seed hyperparameters must come from seed 17")
    learning_rate = float(artifact.get("selected_encoder_learning_rate", 0))
    epoch = int(artifact.get("selected_epoch", 0))
    if learning_rate <= 0 or not 1 <= epoch <= maximum_epoch:
        raise ValueError("Reusable selection contains invalid hyperparameters")
    return artifact, source_path, sha256(encoded).hexdigest()


def _normalized_selection_identity(
    identity: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize pre-stabilization default artifacts without weakening identity."""

    if not isinstance(identity, Mapping):
        return None
    output = dict(identity)
    stabilization = dict(output.get("stabilization", {}))
    stabilization.setdefault("standardize_embedding", False)
    stabilization.setdefault("freeze_batchnorm_running_stats", False)
    stabilization.setdefault("zero_initialize_output_layer", False)
    stabilization.setdefault(
        "embedding_standard_deviation_floor",
        DEFAULT_EMBEDDING_STANDARD_DEVIATION_FLOOR,
    )
    output["stabilization"] = stabilization
    return output


def run_finetuning_v3(
    data: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    model_key: str,
    condition: Literal["finetuned", "finetuned_demographics"],
    target: Literal["sbp", "dbp"],
    fold: int,
    seed: int,
    output_root: str | Path,
    checkpoint_root: str | Path | None = None,
    device: str = "auto",
    config: FineTuneConfig = FineTuneConfig(),
    prepared_waveforms: PreparedWaveforms | None = None,
    encoder_factory: Callable[[str, str], Any] = _factory,
    reuse_selection: str | Path | Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Select LR, refit on the full development pool, and predict outer test.

    The three split roles must already be locked and subject-disjoint.  This
    function never uses outer-test labels for preprocessing, hyperparameter
    selection, epoch selection, or fitting.
    """

    config = config.checked()
    if condition not in {"finetuned", "finetuned_demographics"}:
        raise ValueError("Unsupported fine-tuning condition")
    if target not in {"sbp", "dbp"}:
        raise ValueError("target must be sbp or dbp")
    if data.empty:
        raise ValueError("Fine-tuning data cannot be empty")
    cohort_names = data["dataset"].astype(str).unique()
    if len(cohort_names) != 1:
        raise ValueError("Fine-tuning requires one cohort at a time")
    cohort = str(cohort_names[0])
    train_rows = rows_for_role(data, splits, fold, "train").reset_index(drop=True)
    validation_rows = rows_for_role(data, splits, fold, "validation").reset_index(drop=True)
    test_rows = rows_for_role(data, splits, fold, "test").reset_index(drop=True)
    role_subjects = {
        role: set(frame["subject_id"].astype(str))
        for role, frame in (
            ("train", train_rows),
            ("validation", validation_rows),
            ("test", test_rows),
        )
    }
    if (
        role_subjects["train"] & role_subjects["validation"]
        or role_subjects["train"] & role_subjects["test"]
        or role_subjects["validation"] & role_subjects["test"]
    ):
        raise ValueError("Fine-tuning roles are not subject-disjoint")

    torch_device = _device(device)
    if prepared_waveforms is None:
        preprocessing_encoder = encoder_factory(model_key, str(torch_device))
        prepared_waveforms = prepare_waveforms(data, preprocessing_encoder)
        del preprocessing_encoder
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
    prepared_waveforms.checked()
    if set(prepared_waveforms.frame["segment_id"].astype(str)) != set(
        data["segment_id"].astype(str)
    ):
        raise ValueError("Prepared waveform cache does not match the run manifest")

    use_demographics = condition == "finetuned_demographics"
    demographic_fields = available_demographic_fields(data) if use_demographics else ()
    missing_indicator_fields = tuple(
        field
        for field in ("age", "bmi")
        if field in demographic_fields and data[field].isna().any()
    )
    selection_transform_rows = (
        capped_subject_rows(
            train_rows,
            config.maximum_pulsedb_segments_per_subject,
            EVALUATION_SEGMENT_SEED,
        )
        if cohort.startswith("PulseDB")
        else train_rows
    )
    selection_transform_weights = participant_measurement_weights(
        selection_transform_rows
    )
    selection_demographics = (
        DemographicTransformV3.fit(
            selection_transform_rows,
            demographic_fields,
            missing_indicator_fields=missing_indicator_fields,
            sample_weight=selection_transform_weights,
        )
        if use_demographics
        else None
    )
    selection_target = TargetZScoreTransform.fit(
        selection_transform_rows, target, selection_transform_weights
    )

    root = (
        Path(output_root)
        / cohort.lower().replace(" ", "_").replace("-", "_")
        / model_key
        / condition
        / target
        / f"fold_{fold}"
        / f"seed_{seed}"
    )
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = (
        root
        if checkpoint_root is None
        else (
            Path(checkpoint_root)
            / cohort.lower().replace(" ", "_").replace("-", "_")
            / model_key
            / condition
            / target
            / f"fold_{fold}"
            / f"seed_{seed}"
        )
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    split_fingerprint = (
        str(splits["dataset_fingerprint"].iloc[0])
        if "dataset_fingerprint" in splits
        else "unrecorded"
    )
    selection_identity = {
        "dataset": cohort,
        "model": model_key,
        "condition": condition,
        "target": target,
        "fold": int(fold),
        "split_fingerprint": split_fingerprint,
        "demographic_fields": list(demographic_fields),
        "stabilization": {
            "standardize_embedding": bool(config.standardize_embedding),
            "freeze_batchnorm_running_stats": bool(
                config.freeze_batchnorm_running_stats
            ),
            "zero_initialize_output_layer": bool(
                config.zero_initialize_output_layer
            ),
            "embedding_standard_deviation_floor": float(
                config.embedding_standard_deviation_floor
            ),
        },
    }
    reusable_path = selection_artifact_path(
        output_root,
        cohort=cohort,
        model_key=model_key,
        condition=condition,
        target=target,
        fold=fold,
    )
    selection: list[dict[str, Any]] = []
    selection_diagnostics: dict[str, Any] = {}
    selection_reused = reuse_selection is not None
    selection_source_path: str | None = None
    if reuse_selection is not None:
        selection_artifact, selection_source_path, selection_sha256 = (
            _load_reusable_selection(
                reuse_selection,
                selection_identity,
                maximum_epoch=config.max_epochs,
            )
        )
        selection = list(selection_artifact["learning_rate_selection"])
        selected_learning_rate = float(
            selection_artifact["selected_encoder_learning_rate"]
        )
        selected_epoch = int(selection_artifact["selected_epoch"])
        best_selection = {
            "encoder_learning_rate": selected_learning_rate,
            "best_epoch": selected_epoch,
            "validation_mae_mmhg": float(
                selection_artifact["selected_validation_mae_mmhg"]
            ),
            "embedding_standardization": selection_artifact.get(
                "selection_epoch_zero_embedding_standardization"
            ),
        }
    else:
        if seed != 17:
            raise ValueError(
                "Hyperparameters must be selected once with seed 17; pass "
                "reuse_selection for refit seeds 23 and 42"
            )
        for learning_rate in config.learning_rates:
            token = f"{learning_rate:.0e}".replace("-", "m")
            result = _train_model(
                model_key=model_key,
                encoder_factory=encoder_factory,
                prepared=prepared_waveforms,
                train_rows=train_rows,
                validation_rows=validation_rows,
                embedding_standardization_rows=selection_transform_rows,
                embedding_standardization_scope=(
                    "selection_outer_train_epoch_zero"
                ),
                target=target,
                target_transform=selection_target,
                target_fit_pool_role="outer_training_selection",
                demographic_transform=selection_demographics,
                encoder_learning_rate=learning_rate,
                seed=seed,
                device=torch_device,
                config=config,
                epochs=config.max_epochs,
                checkpoint_path=(
                    checkpoint_directory / f"selection_lr_{token}_last.pt"
                ),
            )
            selection.append(
                {
                    "encoder_learning_rate": learning_rate,
                    "best_epoch": result.best_epoch,
                    "validation_mae_mmhg": result.best_validation_mae_mmhg,
                    "history": result.history,
                    "weight_change_audit": result.weight_change_audit,
                    "maximum_gradient_norm": result.maximum_gradient_norm,
                    "embedding_standardization": (
                        result.embedding_standardization
                    ),
                    "batchnorm_running_stats_audit": (
                        result.batchnorm_running_stats_audit
                    ),
                    "batchnorm_training_mode": result.batchnorm_training_mode,
                }
            )
            selection_diagnostics[str(learning_rate)] = result.native_diagnostics
            del result
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
        best_selection = min(
            selection, key=lambda row: float(row["validation_mae_mmhg"])
        )
        selected_learning_rate = float(best_selection["encoder_learning_rate"])
        selected_epoch = int(best_selection["best_epoch"])
        selection_artifact = {
            "schema_version": "finetuning-v3-selection",
            "identity": selection_identity,
            "selection_seed": 17,
            "selected_encoder_learning_rate": selected_learning_rate,
            "selected_epoch": selected_epoch,
            "selected_validation_mae_mmhg": float(
                best_selection["validation_mae_mmhg"]
            ),
            "learning_rate_selection": selection,
            "selection_target_transform": selection_target.metadata(
                "outer_training_selection"
            ),
            "selection_demographic_transform": (
                selection_demographics.metadata()
                if selection_demographics is not None
                else None
            ),
            "selection_epoch_zero_embedding_standardization": (
                best_selection.get("embedding_standardization")
            ),
            "stabilization": selection_identity["stabilization"],
        }
        selection_sha256 = _atomic_json_save(selection_artifact, reusable_path)
        selection_source_path = str(reusable_path)

    development_rows = pd.concat(
        (train_rows, validation_rows), ignore_index=True
    ).drop_duplicates("segment_id")
    development_transform_rows = (
        capped_subject_rows(
            development_rows,
            config.maximum_pulsedb_segments_per_subject,
            EVALUATION_SEGMENT_SEED,
        )
        if cohort.startswith("PulseDB")
        else development_rows
    )
    development_transform_weights = participant_measurement_weights(
        development_transform_rows
    )
    final_target = TargetZScoreTransform.fit(
        development_transform_rows, target, development_transform_weights
    )
    final_demographics = (
        DemographicTransformV3.fit(
            development_transform_rows,
            demographic_fields,
            missing_indicator_fields=missing_indicator_fields,
            sample_weight=development_transform_weights,
        )
        if use_demographics
        else None
    )
    final_result = _train_model(
        model_key=model_key,
        encoder_factory=encoder_factory,
        prepared=prepared_waveforms,
        train_rows=development_rows,
        validation_rows=None,
        embedding_standardization_rows=development_transform_rows,
        embedding_standardization_scope=(
            "refit_full_development_epoch_zero"
        ),
        target=target,
        target_transform=final_target,
        target_fit_pool_role="full_development_refit",
        demographic_transform=final_demographics,
        encoder_learning_rate=selected_learning_rate,
        seed=seed,
        device=torch_device,
        config=config,
        epochs=selected_epoch,
        checkpoint_path=checkpoint_directory / "development_refit_last.pt",
    )

    all_demographics = (
        final_demographics.transform(prepared_waveforms.frame)
        if final_demographics is not None
        else np.empty((len(prepared_waveforms.frame), 0), dtype=np.float32)
    )
    all_target_zscore = final_target.encode(
        pd.to_numeric(prepared_waveforms.frame[target], errors="raise").to_numpy(float)
    )
    if cohort.startswith("PulseDB"):
        test_positions = _capped_positions(
            prepared_waveforms.frame,
            test_rows,
            maximum_segments=config.maximum_pulsedb_segments_per_subject,
            seed=EVALUATION_SEGMENT_SEED,
            epoch=None,
        )
    else:
        test_positions = _positions(prepared_waveforms.frame, test_rows)
    test_loader = _loader(
        prepared_waveforms,
        test_positions,
        all_target_zscore,
        all_demographics,
        batch_size=config.batch_size,
        shuffle=False,
        seed=seed,
        num_workers=config.num_workers,
        pin_memory=torch_device.type == "cuda",
    )
    test_mae, prediction_zscore, prediction_mmhg, ordered_positions = _evaluate(
        final_result.model,
        test_loader,
        torch_device,
        use_demographics,
        prepared_waveforms.frame,
        target,
    )
    ordered = prepared_waveforms.frame.iloc[ordered_positions].reset_index(drop=True)
    for column in ("age", "sex", "bmi", "height_cm", "weight_kg"):
        if column not in ordered:
            ordered[column] = np.nan
    retained_columns = [
        column
        for column in (
            "dataset",
            "source",
            "subject_id",
            "measurement_id",
            "segment_id",
            "age",
            "sex",
            "bmi",
            "height_cm",
            "weight_kg",
        )
        if column in ordered
    ]
    predictions = ordered[retained_columns].copy()
    predictions["fold"] = fold
    predictions["seed"] = seed
    predictions["model"] = model_key
    predictions["condition"] = condition
    predictions["target"] = target
    predictions["y_true_mmhg"] = pd.to_numeric(ordered[target]).to_numpy(float)
    predictions["y_true_zscore_development"] = all_target_zscore[ordered_positions]
    predictions["y_pred_zscore"] = prediction_zscore
    predictions["estimator_output"] = prediction_zscore
    predictions["y_pred_mmhg"] = prediction_mmhg
    predictions["estimator_output_scale"] = "full_development_pool_zscore"
    predictions["decoder"] = "affine_full_development_pool_inverse_zscore"
    predictions["demographic_fields"] = (
        "/".join("BMI" if field == "bmi" else field for field in demographic_fields)
        if use_demographics
        else "none"
    )
    encoder_preprocessing = prepared_waveforms.preprocessing_diagnostics.get(
        "encoder_fingerprint", {}
    ).get("preprocessing", {})
    predictions["preprocessing_policy"] = encoder_preprocessing.get(
        "name", "audited_model_specific"
    )
    predictions["padding_policy"] = "zero" if cohort == "PPG-BP" else "none"
    predictions["padding_required"] = cohort == "PPG-BP"
    predictions["pretraining_overlap"] = overlap_status(model_key, cohort)
    predictions["source_fidelity"] = "source_consistent_adaptation"
    predictions["contract_version"] = CONTRACT_VERSION
    predictions["ridge_input_dimension"] = 0
    predictions["calibration"] = "none"
    predictions["embedding_standardization"] = (
        "fixed_epoch_zero_training_statistics"
        if config.standardize_embedding
        else "none"
    )
    predictions["batchnorm_running_stats"] = (
        "frozen" if config.freeze_batchnorm_running_stats else "train_mode"
    )
    predictions["output_layer_initialization"] = (
        "zero" if config.zero_initialize_output_layer else "framework_default"
    )
    prediction_path = root / "predictions.csv"
    predictions.to_csv(prediction_path, index=False)

    checkpoint_path: str | None = None
    if config.keep_final_checkpoint:
        final_checkpoint = root / "model.pt"
        _atomic_torch_save(
            {
                "model": _cpu_state_dict(final_result.model),
                "target_transform": final_target.metadata(
                    "full_development_refit"
                ),
                "demographic_transform": (
                    final_demographics.metadata() if final_demographics else None
                ),
                "model_key": model_key,
                "condition": condition,
                "target": target,
                "fold": fold,
                "seed": seed,
                "embedding_standardization": (
                    final_result.embedding_standardization
                ),
                "batchnorm_running_stats_audit": (
                    final_result.batchnorm_running_stats_audit
                ),
                "zero_initialize_output_layer": bool(
                    config.zero_initialize_output_layer
                ),
            },
            final_checkpoint,
        )
        checkpoint_path = str(final_checkpoint)

    metadata = {
        "schema_version": "finetuning-v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": cohort,
        "model": model_key,
        "condition": condition,
        "target": target,
        "fold": fold,
        "seed": seed,
        "method_fidelity": "standardized_benchmark_adaptation",
        "preprocessing_policy": encoder_preprocessing.get(
            "name", "audited_model_specific"
        ),
        "source_fidelity": "source_consistent_adaptation",
        "encoder_update": (
            "complete_selected_embedding_path"
            if config.encoder_trainable
            else "frozen_neural_head_control"
        ),
        "loss": (
            "participant_measurement_weighted_mean_squared_error_"
            "on_training_pool_target_zscore"
        ),
        "calibration": "none",
        "selection_protocol": "outer_train_fit_outer_validation_mae",
        "refit_protocol": "fresh_checkpoint_full_development_pool_selected_epochs",
        "test_protocol": "single_pass_untouched_outer_test_subjects",
        "selected_encoder_learning_rate": selected_learning_rate,
        "selected_epoch": selected_epoch,
        "selected_validation_mae_mmhg": best_selection["validation_mae_mmhg"],
        "hyperparameter_selection_reused": selection_reused,
        "hyperparameter_selection_seed": 17,
        "selection_artifact_path": selection_source_path,
        "selection_artifact_sha256": selection_sha256,
        "outer_test_protocol_mae_mmhg": test_mae,
        "learning_rate_selection": selection,
        "selection_target_transform": selection_target.metadata(
            "outer_training_selection"
        ),
        "development_target_transform": final_target.metadata(
            "full_development_refit"
        ),
        "selection_demographic_transform": (
            selection_demographics.metadata() if selection_demographics else None
        ),
        "development_demographic_transform": (
            final_demographics.metadata() if final_demographics else None
        ),
        "stabilization": {
            "standardize_embedding": bool(config.standardize_embedding),
            "freeze_batchnorm_running_stats": bool(
                config.freeze_batchnorm_running_stats
            ),
            "zero_initialize_output_layer": bool(
                config.zero_initialize_output_layer
            ),
            "embedding_standard_deviation_floor": float(
                config.embedding_standard_deviation_floor
            ),
            "selection_epoch_zero_embedding_standardization": (
                best_selection.get("embedding_standardization")
            ),
            "development_epoch_zero_embedding_standardization": (
                final_result.embedding_standardization
            ),
            "development_batchnorm_running_stats_audit": (
                final_result.batchnorm_running_stats_audit
            ),
            "development_batchnorm_training_mode": (
                final_result.batchnorm_training_mode
            ),
        },
        "demographic_fields": list(demographic_fields),
        "demographic_feature_names": (
            list(final_demographics.feature_names) if final_demographics else []
        ),
        "config": asdict(config),
        "preprocessing_diagnostics": prepared_waveforms.preprocessing_diagnostics,
        "selection_native_diagnostics": selection_diagnostics,
        "development_refit_native_diagnostics": final_result.native_diagnostics,
        "development_refit_history": final_result.history,
        "encoder_training_summary": final_result.model.encoder_training_summary(),
        "encoder_weight_change_audit": final_result.weight_change_audit,
        "maximum_gradient_norm": final_result.maximum_gradient_norm,
        "checkpoint": checkpoint_path,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": str(torch_device),
        },
    }
    metadata_path = root / "run.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    diagnostics_path = root / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "preprocessing": prepared_waveforms.preprocessing_diagnostics,
                "selection": selection_diagnostics,
                "development_refit": final_result.native_diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return prediction_path, metadata_path


__all__ = [
    "DEFAULT_MAX_EPOCHS",
    "DemographicTransformV3",
    "FineTuneConfig",
    "PreparedWaveforms",
    "TargetZScoreTransform",
    "prepare_waveforms",
    "run_finetuning_v3",
    "selection_artifact_path",
]
