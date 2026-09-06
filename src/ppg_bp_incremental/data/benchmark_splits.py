"""Deterministic subject-disjoint train/validation/test manifests."""

from __future__ import annotations

from hashlib import sha256

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def _subject_table(data: pd.DataFrame) -> pd.DataFrame:
    table = data.groupby("subject_id", as_index=False, sort=True).agg(
        sex=("sex", "first"),
        sbp=("sbp", "median"),
        dbp=("dbp", "median"),
    )
    for column in ("sbp", "dbp"):
        q = min(4, int(table[column].nunique()))
        table[f"{column}_bin"] = pd.qcut(
            table[column], q=q, labels=False, duplicates="drop"
        ).astype("Int64").astype(str)
    table["stratum"] = (
        table["sex"].fillna("unknown").astype(str)
        + "|"
        + table["sbp_bin"]
        + "|"
        + table["dbp_bin"]
    )
    return table


def _usable_labels(labels: pd.Series, minimum: int) -> pd.Series | None:
    candidates = [labels, labels.str.rsplit("|", n=1).str[0], labels.str.split("|").str[0]]
    for candidate in candidates:
        if candidate.value_counts().min() >= minimum:
            return candidate
    return None


def manifest_fingerprint(data: pd.DataFrame) -> str:
    columns = [
        column
        for column in ("dataset", "source", "subject_id", "segment_id", "sbp", "dbp", "age", "sex", "waveform_sha256")
        if column in data
    ]
    canonical = data[columns].sort_values(
        ["subject_id", "segment_id"], key=lambda value: value.astype(str)
    ).copy()
    # CSV staging may change the least significant bits of a floating-point
    # label while leaving the scientific value unchanged. Hash a stable decimal
    # representation so dataset identity is independent of storage location and
    # harmless CSV round trips.
    for column in ("sbp", "dbp", "age"):
        if column in canonical:
            values = pd.to_numeric(canonical[column], errors="coerce")
            canonical[column] = values.map(
                lambda value: "" if pd.isna(value) else f"{float(value):.12g}"
            )
    for column in canonical.columns:
        if column not in {"sbp", "dbp", "age"}:
            canonical[column] = canonical[column].fillna("").astype(str)
    hashed = pd.util.hash_pandas_object(canonical, index=False)
    return sha256(hashed.to_numpy().tobytes()).hexdigest()


def make_benchmark_splits(
    data: pd.DataFrame,
    outer_folds: int = 5,
    validation_fraction: float = 0.2,
    seed: int = 20260715,
) -> pd.DataFrame:
    if outer_folds < 2:
        raise ValueError("outer_folds must be at least two")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    subjects = _subject_table(data)
    if len(subjects) < outer_folds:
        raise ValueError("Not enough subjects for the requested folds")
    labels = _usable_labels(subjects["stratum"], outer_folds)
    if labels is None:
        splitter = KFold(outer_folds, shuffle=True, random_state=seed)
        iterator = splitter.split(subjects)
        outer_method = "kfold"
    else:
        splitter = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
        iterator = splitter.split(subjects, labels)
        outer_method = "stratified"
    outer_assignment = np.empty(len(subjects), dtype=int)
    for fold, (_, test_index) in enumerate(iterator):
        outer_assignment[test_index] = fold
    subjects["outer_assignment"] = outer_assignment

    fingerprint = manifest_fingerprint(data)
    rows: list[dict[str, object]] = []
    for fold in range(outer_folds):
        test = subjects[subjects["outer_assignment"] == fold]
        remaining = subjects[subjects["outer_assignment"] != fold]
        remaining_labels = _usable_labels(remaining["stratum"], 2)
        indices = np.arange(len(remaining))
        stratify = remaining_labels if remaining_labels is not None else None
        train_index, validation_index = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=seed + fold,
            stratify=stratify,
        )
        roles = {
            **{subject: "train" for subject in remaining.iloc[train_index]["subject_id"]},
            **{subject: "validation" for subject in remaining.iloc[validation_index]["subject_id"]},
            **{subject: "test" for subject in test["subject_id"]},
        }
        for subject in subjects["subject_id"]:
            rows.append(
                {
                    "subject_id": subject,
                    "fold": fold,
                    "role": roles[subject],
                    "split_seed": seed,
                    "outer_method": outer_method,
                    "validation_method": "stratified" if stratify is not None else "random",
                    "dataset_fingerprint": fingerprint,
                }
            )
    manifest = pd.DataFrame(rows)
    validate_benchmark_splits(data, manifest, outer_folds)
    return manifest


def validate_benchmark_splits(
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    outer_folds: int | None = None,
) -> None:
    required = {"subject_id", "fold", "role", "dataset_fingerprint"}
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    folds = sorted(manifest["fold"].unique())
    if outer_folds is not None and folds != list(range(outer_folds)):
        raise ValueError("Split manifest fold set does not match requested folds")
    subjects = set(data["subject_id"].astype(str))
    for fold, group in manifest.groupby("fold"):
        if set(group["subject_id"].astype(str)) != subjects:
            raise ValueError(f"Fold {fold} does not assign every subject exactly once")
        if group["subject_id"].duplicated().any():
            raise ValueError(f"Fold {fold} contains duplicate subject assignments")
        if set(group["role"]) != {"train", "validation", "test"}:
            raise ValueError(f"Fold {fold} must contain train, validation, and test")
    test_counts = manifest[manifest["role"] == "test"].groupby("subject_id").size()
    if not (test_counts == 1).all() or set(test_counts.index.astype(str)) != subjects:
        raise ValueError("Every subject must occur in exactly one outer test fold")
    if set(manifest["dataset_fingerprint"].astype(str)) != {manifest_fingerprint(data)}:
        raise ValueError("Split manifest fingerprint does not match the dataset")


def refresh_benchmark_split_fingerprint(
    data: pd.DataFrame,
    existing_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Preserve every subject/role assignment while updating dataset identity."""

    refreshed = existing_manifest.copy()
    if set(refreshed["subject_id"].astype(str)) != set(data["subject_id"].astype(str)):
        raise ValueError("Cannot preserve split roles because the subject set changed")
    refreshed["dataset_fingerprint"] = manifest_fingerprint(data)
    validate_benchmark_splits(data, refreshed)
    return refreshed


def rows_for_role(
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    fold: int,
    role: str,
) -> pd.DataFrame:
    subjects = manifest.loc[
        (manifest["fold"] == fold) & (manifest["role"] == role), "subject_id"
    ].astype(str)
    return data[data["subject_id"].astype(str).isin(set(subjects))].copy()
