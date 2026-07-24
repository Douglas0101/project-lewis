"""E06 reopened: representation-only evaluation protocol contracts."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from src.models.e06_protocol import (
    E06EvaluationContract,
    build_outer_splits,
    select_inner_split,
)


def _grouped_fixture() -> tuple[np.ndarray, np.ndarray]:
    groups = np.repeat(np.arange(20), 9)
    labels = np.tile(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), 20)
    return labels.astype(np.int64), groups.astype(np.int64)


def test_e06_contract_changes_representation_only() -> None:
    contract = E06EvaluationContract()

    assert contract.sampling == "natural"
    assert contract.loss == "sparse_categorical_crossentropy"
    assert contract.class_weight is None
    assert contract.decision == "raw_softmax_argmax"
    assert contract.architecture == "minimal_mlp_128"
    assert not contract.outer_test_used_for_early_stopping
    assert contract.imputation == "outer_train_median"
    assert contract.scaling == "outer_train_standard_scaler"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sampling", "smote"),
        ("loss", "focal"),
        ("class_weight", {2: 8.0}),
        ("decision", "calibrated_threshold"),
        ("outer_test_used_for_early_stopping", True),
    ],
)
def test_e06_contract_rejects_prohibited_protocol(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        E06EvaluationContract.model_validate({field: value})


def test_outer_splits_are_patient_disjoint_and_deterministic() -> None:
    labels, groups = _grouped_fixture()
    contract = E06EvaluationContract(n_splits=5, random_seed=42)

    first = build_outer_splits(labels, groups, contract)
    second = build_outer_splits(labels, groups, contract)

    assert len(first) == 5
    for (train_a, test_a), (train_b, test_b) in zip(first, second, strict=True):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(test_a, test_b)
        assert set(groups[train_a]).isdisjoint(set(groups[test_a]))


def test_inner_split_never_uses_outer_test() -> None:
    labels, groups = _grouped_fixture()
    contract = E06EvaluationContract(n_splits=5, inner_splits=4, random_seed=42)
    outer_train, outer_test = build_outer_splits(labels, groups, contract)[0]

    inner_train, inner_val = select_inner_split(
        outer_train,
        labels,
        groups,
        contract,
        fold_index=0,
    )

    assert set(inner_train).issubset(set(outer_train))
    assert set(inner_val).issubset(set(outer_train))
    assert set(inner_train).isdisjoint(set(outer_test))
    assert set(inner_val).isdisjoint(set(outer_test))
    assert set(groups[inner_train]).isdisjoint(set(groups[inner_val]))
