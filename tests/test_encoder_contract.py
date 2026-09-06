from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ppg_bp_incremental.models.encoders.base import (
    EncoderFingerprint,
    NativeEncoderOutput,
    PPGEncoder,
)
from ppg_bp_incremental.models.encoders.cache import (
    input_manifest_fingerprint,
    load_embedding_artifact,
    save_embedding_artifact,
)
from ppg_bp_incremental.data.ppgbp import file_digest
from ppg_bp_incremental.models.contracts import CONTRACT_VERSION
from ppg_bp_incremental.training.embeddings import extract_waveform_embeddings


class DummyEncoder(PPGEncoder):
    def preprocess(self, signal: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
        return np.asarray(signal[:4], dtype=np.float32)

    def encode(self, batch: np.ndarray) -> NativeEncoderOutput:
        values = np.column_stack([batch.mean(axis=1), batch.std(axis=1)])
        return NativeEncoderOutput(
            embedding=values,
            components={"embedding": values},
            component_semantics={"embedding": "test representation"},
        )

    def fingerprint(self) -> EncoderFingerprint:
        return EncoderFingerprint(
            contract_version=CONTRACT_VERSION,
            model_name="dummy",
            model_version="1",
            checkpoint_sha256="abc",
            repository_commit="def",
            preprocessing={"normalization": "none"},
            input_sampling_rate_hz=125,
            input_samples=4,
            embedding_dimension=2,
        )


class DatasetAwareDummyEncoder(DummyEncoder):
    def preprocess_for_dataset(
        self,
        signal: np.ndarray,
        sampling_rate_hz: int,
        *,
        dataset: str,
    ) -> np.ndarray:
        assert dataset == "PPG-BP"
        return np.asarray(signal[-4:], dtype=np.float32)

    def fingerprint_for_dataset(self, dataset: str) -> EncoderFingerprint:
        assert dataset == "PPG-BP"
        base = self.fingerprint()
        return replace(
            base,
            model_version="1-ppgbp",
            preprocessing={**base.preprocessing, "ppgbp_test_operation": True},
        )

    def preprocessing_shape_trace_for_dataset(
        self,
        signal: np.ndarray,
        sampling_rate_hz: int,
        *,
        dataset: str,
    ) -> list[dict[str, object]]:
        assert dataset == "PPG-BP"
        return [
            {
                "name": "dataset_aware_test_operation",
                "shape": [4],
                "sampling_rate_hz": sampling_rate_hz,
            }
        ]

def test_encoder_checks_shapes_and_embedding_cache_round_trip(tmp_path):
    encoder = DummyEncoder()
    first = encoder.preprocess_checked(np.arange(8), 125)
    second = encoder.preprocess_checked(np.arange(8) + 1, 125)
    batch = np.stack([first, second])
    output = encoder.encode_checked(batch)
    embeddings = output.embedding
    assert output.component_shapes() == {"embedding": (2, 2)}
    index = pd.DataFrame(
        {
            "subject_id": ["s1", "s2"],
            "segment_id": ["a", "b"],
            "waveform_sha256": ["one", "two"],
            "sample_rate_hz": [125, 125],
            "n_samples": [4, 4],
        }
    )
    input_fingerprint = input_manifest_fingerprint(index)
    artifact = save_embedding_artifact(
        tmp_path, encoder.fingerprint(), input_fingerprint, index, embeddings
    )
    loaded_embeddings, loaded_index, metadata = load_embedding_artifact(artifact)

    np.testing.assert_array_equal(loaded_embeddings, embeddings)
    pd.testing.assert_frame_equal(loaded_index, index)
    assert metadata["encoder_fingerprint"] == encoder.fingerprint().digest()
    with pytest.raises(FileExistsError):
        save_embedding_artifact(
            tmp_path, encoder.fingerprint(), input_fingerprint, index, embeddings
        )


def test_encoder_rejects_wrong_preprocessed_length():
    encoder = DummyEncoder()
    with pytest.raises(ValueError, match="expected 4 samples"):
        encoder.preprocess_checked(np.arange(3), 125)


def test_extraction_saves_observed_shape_and_native_output_diagnostics(tmp_path):
    waveform = tmp_path / "waveform.npy"
    np.save(waveform, np.arange(8, dtype=np.float32), allow_pickle=False)
    manifest = pd.DataFrame(
        {
            "dataset": ["Synthetic"],
            "source": ["fixture"],
            "subject_id": ["s1"],
            "measurement_id": ["m1"],
            "segment_id": ["segment-1"],
            "waveform_path": [str(waveform)],
            "waveform_sha256": [file_digest(waveform)],
            "waveform_format": ["npy"],
            "sample_rate_hz": [125],
            "n_samples": [8],
            "valid_for_analysis": [True],
        }
    )
    input_csv = tmp_path / "manifest.csv"
    manifest.to_csv(input_csv, index=False)

    artifact = extract_waveform_embeddings(
        input_csv, tmp_path / "embeddings", DummyEncoder(), batch_size=1
    )
    _, _, metadata = load_embedding_artifact(artifact)
    diagnostics = metadata["observed_diagnostics"]

    assert diagnostics["original_waveform_shape"] == [8]
    assert diagnostics["preprocessed_waveform_shape"] == [4]
    assert diagnostics["final_model_input_shape"] == [1, 1, 4]
    assert diagnostics["native_output_shapes"] == {"embedding": [1, 2]}
    assert diagnostics["selected_embedding_shape"] == [1, 2]


def test_extraction_uses_dataset_specific_preprocessing_and_fingerprint(tmp_path):
    waveform = tmp_path / "waveform.npy"
    np.save(waveform, np.arange(8, dtype=np.float32), allow_pickle=False)
    manifest = pd.DataFrame(
        {
            "dataset": ["PPG-BP"],
            "source": ["PPG-BP"],
            "subject_id": ["s1"],
            "measurement_id": ["m1"],
            "segment_id": ["segment-1"],
            "waveform_path": [str(waveform)],
            "waveform_sha256": [file_digest(waveform)],
            "waveform_format": ["npy"],
            "sample_rate_hz": [125],
            "n_samples": [8],
            "valid_for_analysis": [True],
        }
    )
    input_csv = tmp_path / "manifest.csv"
    manifest.to_csv(input_csv, index=False)

    encoder = DatasetAwareDummyEncoder()
    artifact = extract_waveform_embeddings(
        input_csv,
        tmp_path / "embeddings",
        encoder,
        batch_size=1,
    )
    embeddings, _, metadata = load_embedding_artifact(artifact)

    np.testing.assert_allclose(embeddings, [[5.5, np.std([4.0, 5.0, 6.0, 7.0])]])
    assert metadata["encoder"]["model_version"] == "1-ppgbp"
    assert metadata["encoder"]["preprocessing"]["ppgbp_test_operation"] is True
    assert metadata["observed_diagnostics"]["preprocessing_dataset"] == "PPG-BP"
    assert metadata["observed_diagnostics"]["preprocessing_steps"][0]["name"] == (
        "dataset_aware_test_operation"
    )
