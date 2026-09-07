"""Contract tests for the intentionally narrow public CLI."""

from __future__ import annotations

import argparse

import pytest

from ppg_bp_incremental.cli import (
    PUBLIC_COMMANDS,
    _parser,
    _validate_embedding_identity,
)


EXPECTED_PUBLIC_COMMANDS = {
    "prepare",
    "validate",
    "stage",
    "make-splits",
    "audit-models",
    "extract",
    "run-ridge",
    "run-finetune",
    "aggregate",
    "plot",
    "data-eda",
}


def _command_choices(parser: argparse.ArgumentParser) -> set[str]:
    subparser_actions = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(subparser_actions) == 1
    return set(subparser_actions[0].choices)


def test_release_cli_exposes_exact_public_command_set() -> None:
    parser = _parser()
    assert PUBLIC_COMMANDS == EXPECTED_PUBLIC_COMMANDS
    assert _command_choices(parser) == EXPECTED_PUBLIC_COMMANDS


@pytest.mark.parametrize(
    ("command", "required_arguments"),
    [
        ("prepare", ["--dataset", "ppgbp", "--raw-root", "raw"]),
        ("validate", ["--input-csv", "manifest.csv"]),
        (
            "stage",
            [
                "--input-csv",
                "manifest.csv",
                "--output-csv",
                "staged.csv",
                "--stage-root",
                "stage",
            ],
        ),
        (
            "make-splits",
            ["--input-csv", "manifest.csv", "--output", "splits.csv"],
        ),
        ("audit-models", []),
        (
            "extract",
            ["--model", "handcrafted_ppg", "--input-csv", "manifest.csv"],
        ),
        (
            "run-ridge",
            [
                "--input-csv",
                "manifest.csv",
                "--splits",
                "splits.csv",
                "--condition",
                "demographics",
                "--model",
                "demographics",
                "--output",
                "predictions.csv",
            ],
        ),
        (
            "run-finetune",
            [
                "--input-csv",
                "manifest.csv",
                "--splits",
                "splits.csv",
                "--model",
                "pulseppg",
                "--condition",
                "finetuned_demographics",
                "--target",
                "sbp",
                "--fold",
                "0",
                "--seed",
                "17",
                "--output-root",
                "finetuning",
            ],
        ),
        (
            "aggregate",
            ["--predictions", "predictions.csv", "--output-root", "aggregate"],
        ),
        (
            "plot",
            [
                "--aggregated-predictions",
                "units.csv",
                "--metrics",
                "metrics.csv",
                "--output-root",
                "figures",
            ],
        ),
        ("data-eda", ["--input-csv", "manifest.csv", "--output-root", "eda"]),
    ],
)
def test_every_public_command_constructs(command: str, required_arguments: list[str]) -> None:
    args = _parser().parse_args([command, *required_arguments])
    assert args.command == command


def test_release_defaults_match_the_reported_run() -> None:
    parser = _parser()
    extract = parser.parse_args(
        ["extract", "--model", "handcrafted_ppg", "--input-csv", "manifest.csv"]
    )
    aggregate = parser.parse_args(
        ["aggregate", "--predictions", "predictions.csv", "--output-root", "out"]
    )
    finetune = parser.parse_args(
        [
            "run-finetune",
            "--input-csv",
            "manifest.csv",
            "--splits",
            "splits.csv",
            "--model",
            "pulseppg",
            "--condition",
            "finetuned",
            "--target",
            "dbp",
            "--fold",
            "0",
            "--seed",
            "17",
            "--output-root",
            "finetuning",
        ]
    )
    assert extract.batch_size == 256
    assert aggregate.bootstrap_replicates == 10_000
    assert finetune.batch_size == 64
    assert finetune.max_epochs == 10
    assert finetune.patience == 3
    assert finetune.gradient_clip_norm == 1.0
    assert finetune.standardize_embedding is None
    assert finetune.zero_initialize_output_layer is None


def test_release_cli_rejects_embedding_model_mismatch() -> None:
    with pytest.raises(ValueError, match="artifact/model mismatch"):
        _validate_embedding_identity(
            {"encoder": {"model_name": "pulseppg"}},
            "anyppg",
        )
