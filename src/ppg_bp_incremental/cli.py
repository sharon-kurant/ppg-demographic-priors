"""Public command line interface for the reported demographic-prior benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import shutil

import pandas as pd

from ppg_bp_incremental import __version__
from ppg_bp_incremental.data.benchmark import load_benchmark_manifest
from ppg_bp_incremental.data.benchmark_splits import (
    make_benchmark_splits,
    refresh_benchmark_split_fingerprint,
    validate_benchmark_splits,
)
from ppg_bp_incremental.models.contracts import CONTRACT_VERSION
from ppg_bp_incremental.models.registry import MODEL_SPECS, create_encoder


DATASETS = ("ppgbp", "pulsedb-vital", "pulsedb-mimic", "butppg")
RIDGE_CONDITIONS = (
    "demographics",
    "ppg_features",
    "ppg_features_demographics",
    "frozen",
    "frozen_demographics",
)
PUBLIC_COMMANDS = frozenset(
    {
        "prepare",
        "validate",
        "stage",
        "make-splits",
        "audit-models",
        "extract",
        "run-ridge",
        "aggregate",
        "plot",
        "data-eda",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ppg-bp",
        description=(
            "Reproduce the subject-disjoint SBP/DBP benchmark for four frozen "
            "PPG foundation models"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="prepare one supported benchmark cohort"
    )
    prepare.add_argument("--dataset", choices=DATASETS, required=True)
    prepare.add_argument("--raw-root", type=Path, required=True)
    prepare.add_argument("--output-csv", type=Path)
    prepare.add_argument("--provenance-json", type=Path)
    prepare.add_argument("--qc-manifest", type=Path)
    prepare.add_argument("--source-archive", type=Path)
    prepare.add_argument("--sample-rate-hz", type=int, default=125)
    prepare.add_argument(
        "--dataset-version",
        help="source release version (PulseDB defaults to 2.0)",
    )
    prepare.add_argument(
        "--subject-limit",
        type=int,
        help="deterministically sample this many eligible PulseDB subjects",
    )
    prepare.add_argument(
        "--segments-per-subject",
        type=int,
        help="retain this many eligible segments per selected subject",
    )
    prepare.add_argument("--selection-seed", type=int, default=42)
    prepare.add_argument("--subject-selection-csv", type=Path)
    prepare.add_argument(
        "--waveform-cache-hdf5",
        type=Path,
        help="materialize selected PulseDB PPG arrays in a compact HDF5 file",
    )

    validate = subparsers.add_parser(
        "validate", help="validate a common-contract cohort manifest"
    )
    validate.add_argument("--input-csv", type=Path, required=True)

    stage = subparsers.add_parser(
        "stage", help="copy waveform source files to node-local storage"
    )
    stage.add_argument("--input-csv", type=Path, required=True)
    stage.add_argument("--output-csv", type=Path, required=True)
    stage.add_argument("--stage-root", type=Path, required=True)

    splits = subparsers.add_parser(
        "make-splits", help="create locked subject-disjoint cross-validation splits"
    )
    splits.add_argument("--input-csv", type=Path, required=True)
    splits.add_argument("--output", type=Path, required=True)
    splits.add_argument("--outer-folds", type=int, default=5)
    splits.add_argument("--validation-fraction", type=float, default=0.2)
    splits.add_argument("--seed", type=int, default=20260715)
    splits.add_argument(
        "--preserve-assignments-from",
        type=Path,
        help="reuse subject/fold/role assignments and refresh only the data fingerprint",
    )
    splits.add_argument("--force", action="store_true")

    audit = subparsers.add_parser(
        "audit-models", help="load real checkpoints and verify native contracts"
    )
    audit.add_argument(
        "--output-json", type=Path, default=Path("artifacts/audit/models.json")
    )
    audit.add_argument("--device", default="cpu")

    extract = subparsers.add_parser(
        "extract", help="extract cached frozen embeddings or handcrafted features"
    )
    extract.add_argument(
        "--model", choices=(*MODEL_SPECS, "handcrafted_ppg"), required=True
    )
    extract.add_argument("--input-csv", type=Path, required=True)
    extract.add_argument(
        "--output-root", type=Path, default=Path("artifacts/embeddings")
    )
    extract.add_argument("--device", default="auto")
    extract.add_argument("--batch-size", type=int, default=256)
    extract.add_argument("--limit", type=int)
    extract.add_argument("--repository", type=Path)
    extract.add_argument("--checkpoint", type=Path)

    ridge = subparsers.add_parser(
        "run-ridge", help="fit one frozen-representation or baseline Ridge condition"
    )
    ridge.add_argument("--input-csv", type=Path, required=True)
    ridge.add_argument("--splits", type=Path, required=True)
    ridge.add_argument("--condition", choices=RIDGE_CONDITIONS, required=True)
    ridge.add_argument(
        "--model",
        choices=("demographics", "handcrafted_ppg", *MODEL_SPECS),
        required=True,
    )
    ridge.add_argument("--embedding-artifact", type=Path)
    ridge.add_argument(
        "--targets", nargs="+", choices=("sbp", "dbp"), default=("sbp", "dbp")
    )
    ridge.add_argument("--folds", nargs="+", type=int)
    ridge.add_argument("--seed", type=int, default=20260715)
    ridge.add_argument(
        "--ridge-alphas",
        nargs="+",
        type=float,
        default=(0.1, 1, 10, 100, 1000),
        help="positive alpha grid selected by inner subject-disjoint MAE",
    )
    ridge.add_argument("--output", type=Path, required=True)

    aggregate = subparsers.add_parser(
        "aggregate", help="combine predictions and calculate participant-balanced metrics"
    )
    aggregate.add_argument("--predictions", nargs="+", type=Path, required=True)
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--bootstrap-replicates", type=int, default=10000)
    aggregate.add_argument("--bootstrap-confidence", type=float, default=0.95)
    aggregate.add_argument("--seed", type=int, default=20260715)
    aggregate.add_argument(
        "--require-complete-matrix",
        action="store_true",
        help="reject missing methods/cohorts and non-identical paired rows",
    )

    plot = subparsers.add_parser(
        "plot", help="generate benchmark comparison visualizations"
    )
    plot.add_argument("--aggregated-predictions", type=Path, required=True)
    plot.add_argument("--metrics", type=Path, required=True)
    plot.add_argument("--output-root", type=Path, required=True)

    data_eda = subparsers.add_parser(
        "data-eda", help="summarize cohort contents and descriptive distributions"
    )
    data_eda.add_argument("--input-csv", nargs="+", type=Path, required=True)
    data_eda.add_argument("--output-root", type=Path, required=True)

    return parser


def _dataset_defaults(dataset: str) -> tuple[Path, Path, Path]:
    token = dataset.replace("-", "_")
    return (
        Path(f"data/processed/{token}.csv"),
        Path(f"data/manifests/{token}_provenance.json"),
        Path(f"data/manifests/{token}_quality_control.csv"),
    )


def _load_splits(path: Path, data: pd.DataFrame) -> pd.DataFrame:
    splits = pd.read_csv(path)
    validate_benchmark_splits(data, splits)
    return splits


def _write_run_metadata(path: Path, payload: dict) -> None:
    packages: dict[str, str] = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    complete_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        **payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(complete_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_embedding_identity(metadata: dict, requested_model: str) -> None:
    artifact_model = str(metadata.get("encoder", {}).get("model_name", ""))
    if artifact_model != requested_model:
        raise ValueError(
            "Embedding artifact/model mismatch: artifact records "
            f"{artifact_model!r}, command requested {requested_model!r}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "prepare":
        default_csv, default_provenance, default_qc = _dataset_defaults(args.dataset)
        output_csv = args.output_csv or default_csv
        provenance_json = args.provenance_json or default_provenance
        if args.dataset == "ppgbp":
            from ppg_bp_incremental.data.ppgbp import prepare_ppgbp

            data, _, _ = prepare_ppgbp(
                args.raw_root,
                output_csv,
                args.qc_manifest or default_qc,
                provenance_json,
                args.source_archive,
            )
        elif args.dataset == "butppg":
            from ppg_bp_incremental.data.butppg import prepare_butppg

            data, _, _ = prepare_butppg(
                args.raw_root,
                output_csv,
                args.qc_manifest or default_qc,
                provenance_json,
                args.source_archive,
                args.segments_per_subject or 20,
                args.selection_seed,
            )
        else:
            from ppg_bp_incremental.data.pulsedb import prepare_pulsedb

            cohort = "vital" if args.dataset.endswith("vital") else "mimic"
            data, _ = prepare_pulsedb(
                args.raw_root,
                cohort,
                output_csv,
                provenance_json,
                args.sample_rate_hz,
                args.dataset_version or "2.0",
                args.subject_limit,
                args.segments_per_subject,
                args.selection_seed,
                args.subject_selection_csv,
                None,
                args.waveform_cache_hdf5,
            )
        print(
            f"Prepared {len(data):,} segments from "
            f"{data['subject_id'].nunique():,} subjects"
        )
        print(f"Manifest: {output_csv}")
        print(f"Provenance: {provenance_json}")
        return 0

    if args.command == "validate":
        data = load_benchmark_manifest(args.input_csv)
        print(
            f"Valid {data['dataset'].iloc[0]} manifest: {len(data):,} segments, "
            f"{data['subject_id'].nunique():,} subjects, shape={list(data.shape)}"
        )
        return 0

    if args.command == "stage":
        data = load_benchmark_manifest(args.input_csv)
        args.stage_root.mkdir(parents=True, exist_ok=True)
        mapping: dict[str, str] = {}
        for source in sorted(set(data["waveform_path"].astype(str))):
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"Waveform source not found: {source_path}")
            prefix = sha256(str(source_path.resolve()).encode()).hexdigest()[:12]
            destination = args.stage_root / f"{prefix}-{source_path.name}"
            if not destination.exists():
                shutil.copy2(source_path, destination)
            mapping[source] = str(destination.resolve())
        data["waveform_path"] = data["waveform_path"].astype(str).map(mapping)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.output_csv, index=False)
        print(f"Staged {len(mapping):,} waveform source files to {args.stage_root}")
        print(f"Staged manifest: {args.output_csv}")
        return 0

    if args.command == "make-splits":
        if args.output.exists() and not args.force:
            raise FileExistsError(
                f"Refusing to overwrite locked split manifest: {args.output}"
            )
        data = load_benchmark_manifest(args.input_csv)
        if args.preserve_assignments_from:
            splits = refresh_benchmark_split_fingerprint(
                data, pd.read_csv(args.preserve_assignments_from)
            )
        else:
            splits = make_benchmark_splits(
                data, args.outer_folds, args.validation_fraction, args.seed
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        splits.to_csv(args.output, index=False)
        print(f"Wrote {args.outer_folds} subject-disjoint folds to {args.output}")
        return 0

    if args.command == "audit-models":
        from ppg_bp_incremental.evaluation.model_audit import audit_models

        payload = audit_models(args.output_json, args.device)
        print(
            f"Validated {len(payload['models'])} real checkpoint contracts: "
            f"{args.output_json}"
        )
        return 0

    if args.command == "extract":
        from ppg_bp_incremental.training.embeddings import (
            extract_waveform_embeddings,
        )

        if args.model == "handcrafted_ppg":
            from ppg_bp_incremental.models.handcrafted_ppg import (
                HandcraftedFeatureEncoder,
            )

            encoder = HandcraftedFeatureEncoder()
        else:
            encoder = create_encoder(
                args.model, args.device, args.repository, args.checkpoint
            )
        artifact = extract_waveform_embeddings(
            args.input_csv,
            args.output_root,
            encoder,
            args.batch_size,
            args.limit,
        )
        print(f"Validated representation artifact: {artifact}")
        return 0

    if args.command == "run-ridge":
        from ppg_bp_incremental.models.encoders.cache import load_embedding_artifact
        from ppg_bp_incremental.training.ridge_benchmark import (
            run_ridge_benchmark,
            save_ridge_predictions,
        )

        data = load_benchmark_manifest(args.input_csv)
        splits = _load_splits(args.splits, data)
        representation = index = metadata = None
        if args.embedding_artifact is not None:
            representation, index, metadata = load_embedding_artifact(
                args.embedding_artifact
            )
            _validate_embedding_identity(metadata, args.model)
        preprocessing_policy = "none"
        padding_policy = "none"
        padding_required = False
        if metadata is not None:
            preprocessing = metadata["encoder"].get("preprocessing", {})
            preprocessing_policy = str(preprocessing.get("name", "cached_native"))
            steps = metadata.get("observed_diagnostics", {}).get(
                "preprocessing_steps", []
            )
            padding_required = any(
                bool(step.get("padding_required", False)) for step in steps
            )
            if padding_required:
                padding_policy = "zero"
        predictions = run_ridge_benchmark(
            data,
            splits,
            args.condition,
            args.model,
            index,
            representation,
            tuple(args.targets),
            tuple(args.folds) if args.folds else None,
            args.seed,
            preprocessing_policy=preprocessing_policy,
            padding_policy=padding_policy,
            padding_required=padding_required,
            ridge_alphas=tuple(args.ridge_alphas),
        )
        save_ridge_predictions(predictions, args.output)
        print(
            "ridge diagnostics:",
            json.dumps(
                {
                    "rows": len(predictions),
                    "representation_shape": (
                        list(representation.shape)
                        if representation is not None
                        else None
                    ),
                    "ridge_input_dimensions": sorted(
                        predictions["ridge_input_dimension"].unique().tolist()
                    ),
                    "estimator_output_shape": [len(predictions)],
                    "target_scale": "raw_mmhg",
                    "decoder": "identity",
                    "prediction_preview": predictions["y_pred_mmhg"]
                    .head(5)
                    .tolist(),
                },
                sort_keys=True,
            ),
        )
        _write_run_metadata(
            args.output.with_suffix(".run.json"),
            {
                "dataset": str(data["dataset"].iloc[0]),
                "model": args.model,
                "condition": args.condition,
                "contract_version": CONTRACT_VERSION,
                "demographic_fields": sorted(
                    set(predictions["demographic_fields"].astype(str))
                ),
                "target_scale": "raw_mmhg",
                "target_transform": "none",
                "decoder": "identity",
                "split_fingerprint": str(splits["dataset_fingerprint"].iloc[0]),
                "embedding_metadata": metadata,
                "selected_hyperparameters": predictions[
                    ["fold", "target", "selected_alpha"]
                ]
                .drop_duplicates()
                .sort_values(["target", "fold"])
                .to_dict("records"),
                "ridge_alpha_grid": list(args.ridge_alphas),
                "observed_shapes": {
                    "representation": (
                        list(representation.shape)
                        if representation is not None
                        else None
                    ),
                    "ridge_input_dimensions": sorted(
                        predictions["ridge_input_dimension"].unique().tolist()
                    ),
                    "bp_task_output": [len(predictions)],
                },
                "ridge_fit_diagnostics": predictions.attrs.get(
                    "fit_diagnostics", []
                ),
            },
        )
        print(f"Predictions: {args.output}")
        return 0

    if args.command == "aggregate":
        from ppg_bp_incremental.evaluation.benchmark import (
            incremental_effects,
            summarize_benchmark,
            validate_complete_benchmark_matrix,
        )

        predictions = pd.concat(
            [pd.read_csv(path) for path in args.predictions], ignore_index=True
        )
        if args.require_complete_matrix:
            validate_complete_benchmark_matrix(predictions)
        units, metrics = summarize_benchmark(
            predictions,
            args.bootstrap_replicates,
            args.bootstrap_confidence,
            args.seed,
            set(),
        )
        effects = incremental_effects(metrics)
        args.output_root.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output_root / "predictions.csv", index=False)
        units.to_csv(args.output_root / "aggregated_predictions.csv", index=False)
        metrics.to_csv(args.output_root / "metrics.csv", index=False)
        effects.to_csv(args.output_root / "incremental_effects.csv", index=False)
        _write_run_metadata(
            args.output_root / "aggregation.run.json",
            {
                "prediction_files": [str(path) for path in args.predictions],
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_confidence": args.bootstrap_confidence,
                "complete_matrix_required": bool(args.require_complete_matrix),
                "target_scale": "raw_mmhg",
                "contract_version": CONTRACT_VERSION,
            },
        )
        print(
            f"Aggregated {len(predictions):,} prediction rows into {args.output_root}"
        )
        return 0

    if args.command == "plot":
        from ppg_bp_incremental.evaluation.benchmark import generate_visualizations

        paths = generate_visualizations(
            pd.read_csv(args.aggregated_predictions),
            pd.read_csv(args.metrics),
            args.output_root,
        )
        print(f"Generated {len(paths)} visualization files in {args.output_root}")
        return 0

    if args.command == "data-eda":
        from ppg_bp_incremental.evaluation.data_eda import generate_data_eda

        outputs = generate_data_eda(args.input_csv, args.output_root)
        print(f"Data EDA summary: {outputs['summary']}")
        return 0

    raise AssertionError(f"Unhandled public command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
