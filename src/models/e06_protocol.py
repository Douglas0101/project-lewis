"""Evaluation contract for the reopened E06 representation ablations."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import StratifiedGroupKFold


class E06EvaluationContract(BaseModel):
    """Lock every variable except the feature representation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sampling: Literal["natural"] = "natural"
    loss: Literal["sparse_categorical_crossentropy"] = "sparse_categorical_crossentropy"
    class_weight: None = None
    decision: Literal["raw_softmax_argmax"] = "raw_softmax_argmax"
    architecture: Literal["minimal_mlp_128"] = "minimal_mlp_128"
    outer_test_used_for_early_stopping: Literal[False] = False
    imputation: Literal["outer_train_median"] = "outer_train_median"
    scaling: Literal["outer_train_standard_scaler"] = "outer_train_standard_scaler"
    n_splits: int = Field(default=5, ge=2)
    inner_splits: int = Field(default=4, ge=2)
    random_seed: int = Field(default=42, ge=0)


def _validated_targets_groups(
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_values = np.asarray(labels, dtype=np.int64)
    group_values = np.asarray(groups)
    if y_values.ndim != 1 or group_values.ndim != 1:
        raise ValueError("labels and groups must be one-dimensional")
    if y_values.shape[0] != group_values.shape[0]:
        raise ValueError("labels and groups must have equal length")
    if y_values.shape[0] == 0:
        raise ValueError("labels and groups must not be empty")
    if not set(np.unique(y_values)).issubset({0, 1, 2}):
        raise ValueError("Stage 2 labels must be encoded as S=0, V=1, F=2")
    return y_values, group_values


def build_outer_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    contract: E06EvaluationContract,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic patient-disjoint outer folds."""
    y_values, group_values = _validated_targets_groups(labels, groups)
    splitter = StratifiedGroupKFold(
        n_splits=contract.n_splits,
        shuffle=True,
        random_state=contract.random_seed,
    )
    placeholder = np.zeros(y_values.shape[0], dtype=np.uint8)
    return [
        (train.astype(np.int64), test.astype(np.int64))
        for train, test in splitter.split(placeholder, y_values, group_values)
    ]


def select_inner_split(
    outer_train_indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    contract: E06EvaluationContract,
    *,
    fold_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose early-stopping groups exclusively inside one outer train fold."""
    y_values, group_values = _validated_targets_groups(labels, groups)
    outer_train = np.asarray(outer_train_indices, dtype=np.int64)
    if outer_train.ndim != 1 or outer_train.size == 0:
        raise ValueError("outer_train_indices must be a non-empty vector")
    if np.any(outer_train < 0) or np.any(outer_train >= y_values.shape[0]):
        raise ValueError("outer_train_indices contain an out-of-range index")

    splitter = StratifiedGroupKFold(
        n_splits=contract.inner_splits,
        shuffle=True,
        random_state=contract.random_seed + fold_index + 1,
    )
    local_placeholder = np.zeros(outer_train.shape[0], dtype=np.uint8)
    local_train, local_val = next(
        splitter.split(
            local_placeholder,
            y_values[outer_train],
            group_values[outer_train],
        )
    )
    inner_train = outer_train[local_train].astype(np.int64)
    inner_val = outer_train[local_val].astype(np.int64)
    if not set(group_values[inner_train]).isdisjoint(set(group_values[inner_val])):
        raise RuntimeError("inner train and validation groups overlap")
    return inner_train, inner_val
