"""Locked foundation-model registry and construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ppg_bp_incremental.models.contracts import MODEL_CONTRACTS
from ppg_bp_incremental.models.encoders.base import PPGEncoder


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    repository_commit: str
    parameters: int
    repository: Path
    checkpoint: Path


MODEL_SPECS = {
    "papagei_p": ModelSpec(
        "papagei_p",
        "PaPaGei-P",
        "0c537dad4d2850e15b724260de820dd68d77f0b0",
        4_993_024,
        Path("external/papagei/repository"),
        Path("external/papagei/weights/papagei_p.pt"),
    ),
    "papagei_s": ModelSpec(
        "papagei_s",
        "PaPaGei-S",
        "0c537dad4d2850e15b724260de820dd68d77f0b0",
        5_785_612,
        Path("external/papagei/repository"),
        Path("external/papagei/weights/papagei_s.pt"),
    ),
    "pulseppg": ModelSpec(
        "pulseppg",
        "Pulse-PPG",
        "716eaf9cf966e8f76436f2263872ef38b1f90166",
        28_497_920,
        Path("external/pulseppg/repository"),
        Path("external/pulseppg/weights/pulseppg/experiments/out/pulseppg/checkpoint_best.pkl"),
    ),
    "anyppg": ModelSpec(
        "anyppg",
        "AnyPPG",
        "661b877aa96eac3bba320a462cb2a3bfea991103",
        4_044_272,
        Path("external/anyppg/repository"),
        Path("external/anyppg/repository/load_anyppg/anyppg_ckpt.pth"),
    ),
}

if set(MODEL_SPECS) != set(MODEL_CONTRACTS):  # pragma: no cover - import invariant
    raise RuntimeError("Model construction and source-contract registries differ")


OVERLAP_STATUS = {
    ("papagei_p", "PPG-BP"): "unknown",
    ("papagei_s", "PPG-BP"): "unknown",
    ("pulseppg", "PPG-BP"): "none",
    ("anyppg", "PPG-BP"): "none",
    ("papagei_p", "PulseDB-Vital"): "source_overlap",
    ("papagei_s", "PulseDB-Vital"): "source_overlap",
    ("pulseppg", "PulseDB-Vital"): "none",
    ("anyppg", "PulseDB-Vital"): "exact_dataset_overlap",
    ("papagei_p", "PulseDB-MIMIC"): "source_overlap",
    ("papagei_s", "PulseDB-MIMIC"): "source_overlap",
    ("pulseppg", "PulseDB-MIMIC"): "none",
    ("anyppg", "PulseDB-MIMIC"): "exact_dataset_overlap",
    ("papagei_p", "BUT PPG"): "none",
    ("papagei_s", "BUT PPG"): "none",
    ("pulseppg", "BUT PPG"): "none",
    ("anyppg", "BUT PPG"): "none",
}


def overlap_status(model: str, dataset: str) -> str:
    return OVERLAP_STATUS.get((model, dataset), "not_applicable")


def create_encoder(
    model: str,
    device: str = "auto",
    repository: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> PPGEncoder:
    if model not in MODEL_SPECS:
        raise ValueError(f"Unknown model {model!r}; choose from {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[model]
    repository = Path(repository) if repository is not None else spec.repository
    checkpoint = Path(checkpoint) if checkpoint is not None else spec.checkpoint
    if model.startswith("papagei_"):
        from ppg_bp_incremental.models.encoders.papagei import PapageiEncoder

        return PapageiEncoder(model[-1], repository, checkpoint, device=device)
    if model == "pulseppg":
        from ppg_bp_incremental.models.encoders.pulseppg import PulsePPGEncoder

        return PulsePPGEncoder(repository, checkpoint, device=device)
    from ppg_bp_incremental.models.encoders.anyppg import AnyPPGEncoder

    return AnyPPGEncoder(repository, checkpoint, device=device)
