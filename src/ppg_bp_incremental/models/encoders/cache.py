"""Content-addressed, validated storage for frozen segment embeddings."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
import pandas as pd

from ppg_bp_incremental.models.contracts import CONTRACT_VERSION
from ppg_bp_incremental.models.encoders.base import EncoderFingerprint


def _file_sha256(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _software_environment() -> dict:
    packages = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def input_manifest_fingerprint(
    manifest: pd.DataFrame,
    segment_column: str = "segment_id",
) -> str:
    """Fingerprint the ordered segment identity and waveform content metadata."""
    columns = [segment_column]
    for optional in (
        "waveform_sha256",
        "waveform_dataset",
        "waveform_index",
        "waveform_axis",
        "source_waveform_dataset",
        "sample_rate_hz",
        "n_samples",
    ):
        if optional in manifest.columns:
            columns.append(optional)
    if segment_column not in manifest.columns:
        raise ValueError(f"Manifest is missing {segment_column}")
    if manifest[segment_column].duplicated().any():
        raise ValueError("Input manifest contains duplicate segment identifiers")
    canonical = manifest[columns].copy()
    canonical["__sort"] = canonical[segment_column].astype(str)
    canonical = canonical.sort_values("__sort").drop(columns="__sort")
    hashed = pd.util.hash_pandas_object(canonical, index=False)
    return sha256(hashed.to_numpy().tobytes()).hexdigest()


def embedding_cache_key(
    encoder: EncoderFingerprint,
    input_fingerprint: str,
) -> str:
    payload = f"{encoder.digest()}:{input_fingerprint}"
    return sha256(payload.encode("utf-8")).hexdigest()


def save_embedding_artifact(
    output_root: str | Path,
    encoder: EncoderFingerprint,
    input_fingerprint: str,
    index: pd.DataFrame,
    embeddings: np.ndarray,
    observed_diagnostics: dict | None = None,
) -> Path:
    """Atomically save an embedding matrix and its segment index."""
    output_root = Path(output_root)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if "segment_id" not in index.columns or "subject_id" not in index.columns:
        raise ValueError("Embedding index requires segment_id and subject_id")
    if index["segment_id"].duplicated().any():
        raise ValueError("Embedding index contains duplicate segment identifiers")
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional matrix")
    expected_shape = (len(index), encoder.embedding_dimension)
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Embedding shape {embeddings.shape} does not match {expected_shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values")

    key = embedding_cache_key(encoder, input_fingerprint)
    model_root = output_root / encoder.model_name
    artifact_path = model_root / key
    model_root.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists():
        raise FileExistsError(f"Embedding artifact already exists: {artifact_path}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=model_root))
    try:
        embedding_path = temporary / "embeddings.npy"
        index_path = temporary / "index.csv"
        metadata_path = temporary / "metadata.json"
        np.save(embedding_path, embeddings, allow_pickle=False)
        index.reset_index(drop=True).to_csv(index_path, index=False)
        metadata = {
            "cache_key": key,
            "encoder": asdict(encoder),
            "encoder_fingerprint": encoder.digest(),
            "input_fingerprint": input_fingerprint,
            "rows": len(index),
            "embedding_dimension": encoder.embedding_dimension,
            "embeddings_sha256": _file_sha256(embedding_path),
            "index_sha256": _file_sha256(index_path),
            "observed_diagnostics": observed_diagnostics or {},
            "software_environment": _software_environment(),
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.rename(artifact_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return artifact_path


def load_embedding_artifact(
    path: str | Path,
    *,
    allow_stale_source_faithful: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Load and validate a representation artifact.

    Source-faithful benchmark artifacts are contract-bound.  After a waveform
    contract changes, the old bytes remain available for historical inspection
    through an explicit opt-in, but active callers cannot silently consume them
    under the current benchmark version.
    """

    path = Path(path)
    embedding_path = path / "embeddings.npy"
    index_path = path / "index.csv"
    metadata_path = path / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    recorded_contract = str(
        metadata.get("encoder", {}).get("contract_version", "")
    )
    if (
        recorded_contract.startswith("source-faithful-")
        and recorded_contract != CONTRACT_VERSION
        and not allow_stale_source_faithful
    ):
        raise ValueError(
            "Stale source-faithful embedding artifact: artifact uses "
            f"{recorded_contract!r}, active contract is {CONTRACT_VERSION!r}. "
            "Regenerate the embedding or explicitly opt in for historical inspection."
        )
    if _file_sha256(embedding_path) != metadata["embeddings_sha256"]:
        raise ValueError("Embedding matrix checksum mismatch")
    if _file_sha256(index_path) != metadata["index_sha256"]:
        raise ValueError("Embedding index checksum mismatch")
    embeddings = np.load(embedding_path, allow_pickle=False)
    index = pd.read_csv(index_path)
    expected_shape = (metadata["rows"], metadata["embedding_dimension"])
    if embeddings.shape != expected_shape or len(index) != metadata["rows"]:
        raise ValueError("Embedding artifact shape does not match its metadata")
    if index["segment_id"].duplicated().any():
        raise ValueError("Loaded embedding index contains duplicate segment identifiers")
    if not np.isfinite(embeddings).all():
        raise ValueError("Loaded embeddings contain non-finite values")
    return embeddings, index, metadata
