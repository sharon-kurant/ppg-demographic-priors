"""Frozen embedding extraction over a harmonized waveform manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ppg_bp_incremental.data.ppgbp import file_digest
from ppg_bp_incremental.data.benchmark import load_record_waveform
from ppg_bp_incremental.models.encoders.cache import (
    embedding_cache_key,
    input_manifest_fingerprint,
    load_embedding_artifact,
    save_embedding_artifact,
)
from ppg_bp_incremental.models.encoders.base import PPGEncoder


def extract_waveform_embeddings(
    input_csv: str | Path,
    output_root: str | Path,
    encoder: PPGEncoder,
    batch_size: int,
    limit: int | None = None,
) -> Path:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    data = pd.read_csv(input_csv)
    required = {
        "dataset",
        "subject_id",
        "segment_id",
        "waveform_path",
        "waveform_sha256",
        "sample_rate_hz",
        "n_samples",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Embedding input is missing columns: {sorted(missing)}")
    if "valid_for_analysis" in data.columns:
        valid = data["valid_for_analysis"]
        if valid.dtype != bool:
            valid = valid.astype(str).str.lower().eq("true")
        data = data[valid].copy()
    if data["dataset"].astype(str).str.startswith("PulseDB").all():
        from ppg_bp_incremental.data.pulsedb import (
            validate_pulsedb_raw_waveform_contract,
        )

        validate_pulsedb_raw_waveform_contract(data)
    data = data.sort_values("segment_id").reset_index(drop=True)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        data = data.head(limit).copy()
    if data.empty:
        raise ValueError("No valid segments are available for embedding extraction")

    datasets = data["dataset"].astype(str).unique().tolist()
    if len(datasets) != 1:
        raise ValueError(
            "One embedding artifact must contain exactly one dataset; received "
            f"{sorted(datasets)}"
        )
    dataset = datasets[0]
    encoder_fingerprint = encoder.fingerprint_for_dataset(dataset)

    fingerprint = input_manifest_fingerprint(data)
    key = embedding_cache_key(encoder_fingerprint, fingerprint)
    artifact = Path(output_root) / encoder_fingerprint.model_name / key
    if artifact.exists():
        load_embedding_artifact(artifact)
        return artifact

    processed = np.empty(
        (len(data), encoder_fingerprint.input_samples), dtype=np.float32
    )
    verified_files: dict[str, str] = {}
    diagnostics: dict = {}
    for row_index, row in data.iterrows():
        waveform_path = Path(row["waveform_path"])
        if not waveform_path.exists():
            raise FileNotFoundError(f"Waveform not found: {waveform_path}")
        expected_digest = str(row["waveform_sha256"])
        cache_key = str(waveform_path.resolve())
        actual_digest = verified_files.get(cache_key)
        if actual_digest is None:
            actual_digest = file_digest(waveform_path)
            verified_files[cache_key] = actual_digest
        if actual_digest != expected_digest:
            raise ValueError(f"Waveform checksum mismatch: {waveform_path}")
        signal = load_record_waveform(row)
        processed[row_index] = encoder.preprocess_checked(
            signal,
            int(row["sample_rate_hz"]),
            dataset=dataset,
        )
        if row_index == 0:
            diagnostics = {
                "segment_id": str(row["segment_id"]),
                "original_sampling_rate_hz": int(row["sample_rate_hz"]),
                "original_waveform_shape": list(np.asarray(signal).shape),
                "original_waveform_preview": np.asarray(signal)[:5].tolist(),
                "preprocessing_dataset": dataset,
                "effective_encoder_fingerprint": encoder_fingerprint.digest(),
                "preprocessing_steps": encoder.preprocessing_shape_trace_for_dataset(
                    signal,
                    int(row["sample_rate_hz"]),
                    dataset=dataset,
                ),
                "preprocessed_waveform_shape": list(processed[row_index].shape),
                "preprocessed_waveform_preview": processed[row_index, :5].tolist(),
            }

    embeddings = np.empty(
        (len(data), encoder_fingerprint.embedding_dimension), dtype=np.float32
    )
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        result = encoder.encode_checked(processed[start:stop])
        embeddings[start:stop] = result.embedding
        if start == 0:
            diagnostics.update(
                {
                    "preprocessed_batch_shape": list(processed[start:stop].shape),
                    "final_model_input_shape": (
                        [len(result.embedding), 1, processed.shape[1]]
                        if encoder_fingerprint.input_sampling_rate_hz > 0
                        else list(processed[start:stop].shape)
                    ),
                    "native_output_shapes": {
                        name: list(shape)
                        for name, shape in result.component_shapes().items()
                    },
                    "native_output_previews": {
                        name: np.asarray(value)[0].reshape(-1)[:5].tolist()
                        for name, value in result.components.items()
                    },
                    "native_output_semantics": dict(result.component_semantics),
                    "selected_embedding_shape": list(result.embedding.shape),
                    "selected_embedding_preview": result.embedding[0, :5].tolist(),
                }
            )
            print("encoder diagnostics:", json.dumps(diagnostics, sort_keys=True))

    index_columns = [
        column
        for column in (
            "dataset",
            "dataset_version",
            "source",
            "subject_id",
            "measurement_id",
            "segment_id",
            "waveform_sha256",
            "sample_rate_hz",
            "n_samples",
        )
        if column in data.columns
    ]
    return save_embedding_artifact(
        output_root,
        encoder_fingerprint,
        fingerprint,
        data[index_columns],
        embeddings,
        observed_diagnostics=diagnostics,
    )
