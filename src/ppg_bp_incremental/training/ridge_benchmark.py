"""Subject-disjoint Ridge conditions for frozen and handcrafted features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ppg_bp_incremental.data.benchmark import (
    available_demographic_fields,
    capped_subject_rows,
)
from ppg_bp_incremental.data.benchmark_splits import rows_for_role
from ppg_bp_incremental.training.ridge_utils import (
    FeatureTransformer,
    nested_ridge_predict,
    participant_measurement_weights,
)
from ppg_bp_incremental.models.contracts import (
    CONTRACT_VERSION,
    MODEL_CONTRACTS,
    decode_bp_output,
    source_fidelity,
)
from ppg_bp_incremental.models.registry import overlap_status


RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
EVALUATION_SEGMENT_SEED = 20260715
RIDGE_CONDITIONS = {
    "demographics",
    "ppg_features",
    "ppg_features_demographics",
    "frozen",
    "frozen_demographics",
}


def _validate_condition_model(
    condition: str,
    model: str,
    *,
    representation_supplied: bool,
) -> None:
    """Reject mislabeled or structurally incompatible benchmark runs."""
    if condition not in RIDGE_CONDITIONS:
        raise ValueError(f"Unknown Ridge condition: {condition}")
    if condition == "demographics":
        if model != "demographics":
            raise ValueError("The demographics condition requires model='demographics'")
        if representation_supplied:
            raise ValueError("The demographics condition must not receive an embedding artifact")
        return
    if condition in {"ppg_features", "ppg_features_demographics"}:
        if model != "handcrafted_ppg":
            raise ValueError(
                f"Condition {condition} requires model='handcrafted_ppg'"
            )
        return
    if model not in MODEL_CONTRACTS:
        raise ValueError(
            f"Condition {condition} requires a registered foundation model; "
            f"received {model!r}"
        )


def _attach_representation(
    data: pd.DataFrame,
    representation_index: pd.DataFrame | None,
    representation: np.ndarray | None,
) -> tuple[pd.DataFrame, list[str]]:
    if representation is None:
        return data.copy(), []
    if representation_index is None or len(representation_index) != len(representation):
        raise ValueError("Representation matrix and index are incompatible")
    if "segment_id" not in representation_index:
        raise ValueError("Representation index requires segment_id")
    if representation.ndim != 2 or not np.isfinite(representation).all():
        raise ValueError("Representation must be a finite two-dimensional matrix")
    columns = [f"representation_{index:04d}" for index in range(representation.shape[1])]
    values = pd.concat(
        [
            representation_index[["segment_id"]].reset_index(drop=True),
            pd.DataFrame(representation, columns=columns),
        ],
        axis=1,
    )
    merged = data.merge(values, on="segment_id", how="left", validate="one_to_one")
    if merged[columns].isna().any().any():
        raise ValueError("Some benchmark segments lack the requested representation")
    return merged, columns


def _features_for(
    condition: str,
    representation_columns: list[str],
    demographic_columns: tuple[str, ...],
) -> tuple[list[str], tuple[str, ...]]:
    if condition not in RIDGE_CONDITIONS:
        raise ValueError(f"Unknown Ridge condition: {condition}")
    uses_representation = condition in {
        "ppg_features", "ppg_features_demographics", "frozen", "frozen_demographics"
    }
    uses_demographics = condition in {
        "demographics", "ppg_features_demographics", "frozen_demographics"
    }
    if uses_representation and not representation_columns:
        raise ValueError(f"Condition {condition} requires a representation artifact")
    features = list(representation_columns) if uses_representation else []
    if uses_demographics:
        features.extend(demographic_columns)
    categorical = tuple(column for column in demographic_columns if column == "sex")
    return features, (categorical if uses_demographics else ())


def run_ridge_benchmark(
    data: pd.DataFrame,
    splits: pd.DataFrame,
    condition: str,
    model: str,
    representation_index: pd.DataFrame | None = None,
    representation: np.ndarray | None = None,
    targets: tuple[str, ...] = ("sbp", "dbp"),
    folds: tuple[int, ...] | None = None,
    seed: int = 20260715,
    maximum_segments_per_subject: int = 20,
    preprocessing_policy: str = "none",
    padding_policy: str = "none",
    padding_required: bool = False,
    demographic_columns: tuple[str, ...] | None = None,
    ridge_alphas: tuple[float, ...] = RIDGE_ALPHAS,
) -> pd.DataFrame:
    """Generate test predictions while fitting all transforms on outer training data."""
    if not ridge_alphas or any((not np.isfinite(alpha) or alpha <= 0) for alpha in ridge_alphas):
        raise ValueError("Ridge alphas must be finite positive values")
    _validate_condition_model(
        condition,
        model,
        representation_supplied=(
            representation is not None or representation_index is not None
        ),
    )
    table, representation_columns = _attach_representation(
        data, representation_index, representation
    )
    demographic_columns = demographic_columns or available_demographic_fields(table)
    features, categorical = _features_for(
        condition, representation_columns, demographic_columns
    )
    if any(column not in table for column in demographic_columns):
        raise ValueError("Requested demographic columns are absent from the dataset")
    uses_demographics = condition in {
        "demographics", "ppg_features_demographics", "frozen_demographics"
    }
    if uses_demographics and not demographic_columns:
        raise ValueError("No demographic field is available for this cohort")
    condition_label = condition
    demographic_label = "/".join(
        "BMI" if column == "bmi" else column for column in demographic_columns
    )
    folds = folds or tuple(sorted(int(value) for value in splits["fold"].unique()))
    records: list[pd.DataFrame] = []
    fit_diagnostics: list[dict[str, object]] = []
    dataset = str(table["dataset"].iloc[0])
    for fold in folds:
        train = pd.concat(
            [rows_for_role(table, splits, fold, "train"), rows_for_role(table, splits, fold, "validation")],
            ignore_index=True,
        )
        test = rows_for_role(table, splits, fold, "test")
        if dataset.startswith("PulseDB"):
            train = capped_subject_rows(train, maximum_segments_per_subject, seed, epoch=fold)
            test = capped_subject_rows(
                test,
                maximum_segments_per_subject,
                EVALUATION_SEGMENT_SEED,
                epoch=None,
            )
        n_subjects = train["subject_id"].nunique()
        inner_folds = min(3, n_subjects)
        if inner_folds < 2:
            raise ValueError("At least two training subjects are required for Ridge")
        predicted, selected, tuning = nested_ridge_predict(
            train,
            test,
            features,
            categorical,
            targets,
            alphas=ridge_alphas,
            inner_folds=inner_folds,
        )
        # Reconstruct the exact weighted outer-development transformer used by
        # nested_ridge_predict so shape/value diagnostics describe the fitted
        # estimator rather than an unweighted approximation.
        fit_weights = participant_measurement_weights(train)
        transformer = FeatureTransformer.fit(
            train, features, categorical, fit_weights
        )
        transformed_train = transformer.transform(train)
        transformed_test = transformer.transform(test)
        ridge_input_dimension = int(transformed_train.shape[1])
        for target_index, target in enumerate(targets):
            selected_target = selected.loc[selected["target"].eq(target)]
            if len(selected_target) != 1:
                raise RuntimeError(f"Ridge selection is not unique for {target}")
            selected_alpha = float(selected_target["selected_alpha"].iloc[0])
            task_output = decode_bp_output(
                predicted[:, target_index],
                target=target,
                output_scale="raw_mmhg",
                decoder_metadata={"type": "identity"},
            ).checked(len(test))
            fit_diagnostics.append(
                {
                    "fold": int(fold),
                    "target": target,
                    "declared_input_columns": list(features),
                    "raw_train_input_shape": list(train[features].shape),
                    "raw_test_input_shape": list(test[features].shape),
                    "transformed_train_input_shape": list(transformed_train.shape),
                    "transformed_test_input_shape": list(transformed_test.shape),
                    "transformed_test_input_preview": (
                        transformed_test[0, :8].astype(float).tolist()
                        if len(transformed_test)
                        else []
                    ),
                    "estimator_output_shape": list(task_output.estimator_output.shape),
                    "estimator_output_scale": task_output.output_scale,
                    "decoder": task_output.decoder_metadata["type"],
                    "fit_weighting": (
                        "equal_participant_then_equal_measurement_then_equal_source_row"
                    ),
                    "alpha_selection_metric": "pooled_participant_macro_oof_mae",
                    "selected_alpha": selected_alpha,
                    "inner_validation_mae": float(
                        selected_target["inner_validation_mae"].iloc[0]
                    ),
                    "inner_tuning_rows": int(
                        len(tuning.loc[tuning["target"].eq(target)])
                    ),
                    "estimator_output_preview": (
                        task_output.estimator_output[:5].astype(float).tolist()
                    ),
                    "mmhg_prediction_preview": (
                        task_output.mmhg_prediction[:5].astype(float).tolist()
                    ),
                }
            )
            frame = test[
                [
                    "dataset", "source", "subject_id", "measurement_id", "segment_id",
                    "age", "sex", "bmi", "height_cm", "weight_kg",
                ]
            ].copy()
            frame["fold"] = fold
            frame["seed"] = seed
            frame["model"] = model
            frame["condition"] = condition_label
            frame["demographic_fields"] = (
                demographic_label if uses_demographics else "none"
            )
            frame["target"] = target
            frame["y_true_mmhg"] = test[target].to_numpy(dtype=float)
            frame["estimator_output"] = task_output.estimator_output
            frame["estimator_output_scale"] = task_output.output_scale
            frame["decoder"] = task_output.decoder_metadata["type"]
            frame["y_pred_mmhg"] = task_output.mmhg_prediction
            frame["selected_alpha"] = selected_alpha
            frame["ridge_fit_weighting"] = (
                "equal_participant_then_equal_measurement_then_equal_source_row"
            )
            frame["alpha_selection_metric"] = (
                "pooled_participant_macro_oof_mae"
            )
            frame["ridge_input_dimension"] = ridge_input_dimension
            frame["preprocessing_policy"] = preprocessing_policy
            frame["padding_policy"] = padding_policy
            frame["padding_required"] = bool(padding_required)
            frame["pretraining_overlap"] = overlap_status(model, dataset)
            frame["source_fidelity"] = (
                source_fidelity(model, dataset)
                if model in MODEL_CONTRACTS
                else "not_applicable"
            )
            frame["contract_version"] = CONTRACT_VERSION
            records.append(frame)
    predictions = pd.concat(records, ignore_index=True)
    key = ["segment_id", "fold", "seed", "model", "condition", "target"]
    if predictions.duplicated(key).any():
        raise RuntimeError("Ridge benchmark produced duplicate predictions")
    if not np.isfinite(predictions[["y_true_mmhg", "y_pred_mmhg"]]).all().all():
        raise RuntimeError("Ridge benchmark produced non-finite BP values")
    result = predictions.sort_values(
        ["target", "fold", "subject_id", "segment_id"], key=lambda value: value.astype(str)
    ).reset_index(drop=True)
    result.attrs["fit_diagnostics"] = fit_diagnostics
    return result


def save_ridge_predictions(predictions: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    return path
