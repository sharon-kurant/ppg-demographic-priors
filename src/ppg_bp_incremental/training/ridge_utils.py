"""Leakage-safe preprocessing, weighting, and nested Ridge utilities.

This module contains the estimator primitives used by the released benchmark.
It is intentionally independent of the exploratory demographic-ablation suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0 or len(values) != len(weights):
        raise ValueError("weighted median requires aligned nonempty arrays")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
    ):
        raise ValueError(
            "weighted median inputs must be finite with nonnegative weights"
        )
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, weights.sum() / 2, side="left"))
    return float(values[order[min(index, len(order) - 1)]])


def _checked_weights(weights: np.ndarray, length: int) -> np.ndarray:
    checked = np.asarray(weights, dtype=float).reshape(-1)
    if (
        len(checked) != length
        or not np.isfinite(checked).all()
        or (checked <= 0).any()
    ):
        raise ValueError("sample weights must be positive, finite, and row-aligned")
    # A mean row weight of one keeps the registered alpha grid on the same
    # scale across cohorts.
    return checked * (length / checked.sum())


def _canonical_measurement_ids(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.empty(0, dtype=str)
    if set(frame["dataset"].astype(str)) == {"PPG-BP"}:
        return frame["subject_id"].astype(str).to_numpy(dtype=str)
    return frame["measurement_id"].astype(str).to_numpy(dtype=str)


def participant_measurement_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give participants equal weight, divided across measurements and rows."""

    if frame.empty:
        raise ValueError("cannot weight an empty frame")
    # NumPy 1.26 requires homogeneous unicode arrays for ``np.char`` while
    # pandas otherwise returns object dtype here.
    subject = frame["subject_id"].astype(str).to_numpy(dtype=str)
    measurement = np.asarray(_canonical_measurement_ids(frame), dtype=str)
    keys = pd.DataFrame({"subject": subject, "measurement": measurement})
    rows_per_measurement = keys.groupby(
        ["subject", "measurement"], sort=False
    )["measurement"].transform("size").to_numpy(float)
    measurements_per_subject = (
        keys.drop_duplicates(["subject", "measurement"])
        .groupby("subject")["measurement"]
        .size()
        .reindex(keys["subject"])
        .to_numpy(float)
    )
    raw = 1.0 / (rows_per_measurement * measurements_per_subject)
    return _checked_weights(raw, len(frame))


@dataclass(frozen=True)
class FeatureTransformer:
    """Train-only numeric standardization and fixed sex one-hot encoding."""

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    missing_indicator_columns: tuple[int, ...]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
        categorical_columns: Sequence[str],
        sample_weight: np.ndarray,
    ) -> "FeatureTransformer":
        categorical = tuple(
            column for column in feature_columns if column in set(categorical_columns)
        )
        numeric = tuple(column for column in feature_columns if column not in categorical)
        weights = _checked_weights(sample_weight, len(frame))
        if numeric:
            values = frame.loc[:, numeric].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(float)
            medians = np.asarray(
                [
                    _weighted_median(
                        column[np.isfinite(column)], weights[np.isfinite(column)]
                    )
                    for column in values.T
                ],
                dtype=float,
            )
            missing_columns = tuple(
                int(index) for index in np.flatnonzero(np.isnan(values).any(axis=0))
            )
            filled = np.where(np.isfinite(values), values, medians[None, :])
            if missing_columns:
                indicators = np.isnan(values[:, missing_columns]).astype(float)
                filled = np.column_stack([filled, indicators])
            means = np.average(filled, axis=0, weights=weights)
            variances = np.average((filled - means) ** 2, axis=0, weights=weights)
            scales = np.sqrt(np.maximum(variances, 0.0))
            scales[scales <= np.finfo(float).eps] = 1.0
        else:
            medians = np.empty(0, dtype=float)
            means = np.empty(0, dtype=float)
            scales = np.empty(0, dtype=float)
            missing_columns = ()
        return cls(numeric, categorical, medians, means, scales, missing_columns)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.numeric_columns:
            values = frame.loc[:, self.numeric_columns].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(float)
            filled = np.where(np.isfinite(values), values, self.medians[None, :])
            if self.missing_indicator_columns:
                indicators = np.isnan(
                    values[:, self.missing_indicator_columns]
                ).astype(float)
                filled = np.column_stack([filled, indicators])
            parts.append((filled - self.means) / self.scales)
        for column in self.categorical_columns:
            values = (
                frame[column]
                .fillna("unknown")
                .astype(str)
                .str.strip()
                .str.lower()
                .to_numpy()
            )
            parts.append(
                np.column_stack(
                    [values == category for category in ("female", "male", "unknown")]
                ).astype(float)
            )
        if not parts:
            raise ValueError("at least one feature is required")
        transformed = np.column_stack(parts)
        if transformed.ndim != 2 or not np.isfinite(transformed).all():
            raise ValueError("feature transformation produced invalid values")
        return transformed


def _participant_macro_mae(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    predicted = np.asarray(predictions, dtype=float)
    truth = np.asarray(y_true, dtype=float)
    if predicted.ndim != 3 or truth.ndim != 2:
        raise ValueError("prediction and truth arrays have invalid ranks")
    if predicted.shape[1:] != truth.shape:
        raise ValueError("prediction and truth arrays are not row aligned")
    subject = frame["subject_id"].astype(str).to_numpy(dtype=str)
    measurement = np.asarray(_canonical_measurement_ids(frame), dtype=str)
    measurement_keys = np.char.add(np.char.add(subject, "\x1f"), measurement)
    unique_measurements, measurement_code = np.unique(
        measurement_keys, return_inverse=True
    )
    counts = np.bincount(measurement_code).astype(float)
    n_candidates, _, n_targets = predicted.shape
    aggregated_prediction = np.zeros(
        (n_candidates, len(unique_measurements), n_targets), dtype=float
    )
    aggregated_truth = np.zeros((len(unique_measurements), n_targets), dtype=float)
    for row, code in enumerate(measurement_code):
        aggregated_prediction[:, code, :] += predicted[:, row, :]
        aggregated_truth[code, :] += truth[row, :]
    aggregated_prediction /= counts[None, :, None]
    aggregated_truth /= counts[:, None]
    measurement_subject = np.asarray(
        [key.split("\x1f", 1)[0] for key in unique_measurements], dtype=str
    )
    _, subject_code = np.unique(measurement_subject, return_inverse=True)
    subject_counts = np.bincount(subject_code).astype(float)
    absolute = np.abs(aggregated_prediction - aggregated_truth[None, :, :])
    subject_error = np.zeros(
        (n_candidates, len(subject_counts), n_targets), dtype=float
    )
    for row, code in enumerate(subject_code):
        subject_error[:, code, :] += absolute[:, row, :]
    subject_error /= subject_counts[None, :, None]
    return subject_error.mean(axis=1)


def _ridge_path_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_evaluate: np.ndarray,
    alphas: Sequence[float],
    sample_weight: np.ndarray,
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_evaluate = np.asarray(x_evaluate, dtype=float)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    weights = _checked_weights(sample_weight, len(x_train))
    if (
        x_train.ndim != 2
        or x_evaluate.ndim != 2
        or x_train.shape[1] != x_evaluate.shape[1]
    ):
        raise ValueError("Ridge matrices have incompatible shapes")
    if y_train.shape[0] != x_train.shape[0]:
        raise ValueError("Ridge targets are not row aligned")
    if not all(np.isfinite(array).all() for array in (x_train, y_train, x_evaluate)):
        raise ValueError("Ridge inputs must be finite")
    weight_sum = weights.sum()
    x_mean = np.sum(x_train * weights[:, None], axis=0) / weight_sum
    y_mean = np.sum(y_train * weights[:, None], axis=0) / weight_sum
    square_root_weight = np.sqrt(weights)[:, None]
    centered_x = (x_train - x_mean) * square_root_weight
    centered_y = (y_train - y_mean) * square_root_weight
    gram = centered_x.T @ centered_x
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    right = eigenvectors.T @ (centered_x.T @ centered_y)
    coefficient_path = np.stack(
        [
            eigenvectors @ (right / (eigenvalues[:, None] + float(alpha)))
            for alpha in alphas
        ],
        axis=0,
    )
    intercept_path = y_mean[None, :] - np.einsum(
        "d,adt->at", x_mean, coefficient_path
    )
    result = np.einsum("nd,adt->ant", x_evaluate, coefficient_path)
    return result + intercept_path[:, None, :]


def nested_ridge_predict(
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
    *,
    alphas: Sequence[float],
    inner_folds: int,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Select alpha by participant-macro inner MAE, refit, and predict."""

    if not feature_columns:
        raise ValueError("nested Ridge requires at least one feature")
    target_columns = tuple(target_columns)
    y = train.loc[:, target_columns].to_numpy(float)
    groups = train["subject_id"].astype(str).to_numpy()
    n_splits = min(int(inner_folds), len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("nested Ridge requires at least two participants")
    oof_predictions = np.full(
        (len(alphas), len(train), len(target_columns)), np.nan, dtype=float
    )
    tuning_records: list[dict[str, object]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for inner_fold, (fit_index, held_index) in enumerate(
        splitter.split(train, groups=groups)
    ):
        fit = train.iloc[fit_index]
        held = train.iloc[held_index]
        if set(fit["subject_id"].astype(str)) & set(
            held["subject_id"].astype(str)
        ):
            raise RuntimeError("inner Ridge split is not subject-disjoint")
        fit_weight = participant_measurement_weights(fit)
        transformer = FeatureTransformer.fit(
            fit, feature_columns, categorical_columns, fit_weight
        )
        predicted = _ridge_path_predict(
            transformer.transform(fit),
            y[fit_index],
            transformer.transform(held),
            alphas,
            fit_weight,
        )
        oof_predictions[:, held_index, :] = predicted
        scores = _participant_macro_mae(held, y[held_index], predicted)
        for alpha_index, alpha in enumerate(alphas):
            for target_index, target in enumerate(target_columns):
                tuning_records.append(
                    {
                        "inner_fold": int(inner_fold),
                        "target": target,
                        "alpha": float(alpha),
                        "validation_mae": float(scores[alpha_index, target_index]),
                        "n_fit_subjects": int(fit["subject_id"].nunique()),
                        "n_validation_subjects": int(
                            held["subject_id"].nunique()
                        ),
                    }
                )
    if not np.isfinite(oof_predictions).all():
        raise RuntimeError("inner Ridge OOF predictions do not cover every row")
    mean_scores = _participant_macro_mae(train, y, oof_predictions)
    best_indices: list[int] = []
    for target_index in range(len(target_columns)):
        column = mean_scores[:, target_index]
        minimum = float(np.min(column))
        tied = np.flatnonzero(np.isclose(column, minimum, rtol=0, atol=1e-12))
        best_indices.append(int(tied[-1]))
    full_weight = participant_measurement_weights(train)
    transformer = FeatureTransformer.fit(
        train, feature_columns, categorical_columns, full_weight
    )
    x_train = transformer.transform(train)
    x_evaluate = transformer.transform(evaluate)
    path_evaluate = _ridge_path_predict(
        x_train, y, x_evaluate, alphas, full_weight
    )
    path_train = _ridge_path_predict(x_train, y, x_train, alphas, full_weight)
    predictions = np.column_stack(
        [
            path_evaluate[index, :, target]
            for target, index in enumerate(best_indices)
        ]
    )
    development_predictions = np.column_stack(
        [path_train[index, :, target] for target, index in enumerate(best_indices)]
    )
    selected: list[dict[str, object]] = []
    for target_index, target in enumerate(target_columns):
        best = best_indices[target_index]
        development_mae = _participant_macro_mae(
            train,
            y[:, [target_index]],
            development_predictions[:, [target_index]][None, :, :],
        )[0, 0]
        selected.append(
            {
                "target": target,
                "selected_alpha": float(alphas[best]),
                "inner_validation_mae": float(mean_scores[best, target_index]),
                "alpha_selection_metric": "pooled_participant_macro_oof_mae",
                "fit_weighting": (
                    "equal_participant_then_equal_measurement_then_equal_source_row"
                ),
                "development_in_sample_mae": float(development_mae),
                "input_dimension": int(x_train.shape[1]),
                "n_development_rows": int(len(train)),
                "n_development_subjects": int(train["subject_id"].nunique()),
                "n_evaluation_rows": int(len(evaluate)),
                "n_evaluation_subjects": int(evaluate["subject_id"].nunique()),
            }
        )
    return predictions, pd.DataFrame(selected), pd.DataFrame(tuning_records)
