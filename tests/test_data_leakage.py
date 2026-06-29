"""Validação de ausência de vazamento de dados entre treino e teste."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.stratified_group_split import StratifiedGroupKFold


@pytest.fixture
def leakage_data():
    """Dados com múltiplos pacientes e classes para testar leakage."""
    groups = np.array([10, 10, 20, 20, 30, 30, 40, 40, 50, 50])
    y = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1])
    return groups, y


def test_stratified_group_kfold_no_leakage(leakage_data):
    """StratifiedGroupKFold nunca mistura grupos entre treino e teste."""
    groups, y = leakage_data
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in splitter.split(X=[0] * len(y), y=y, groups=groups):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)
