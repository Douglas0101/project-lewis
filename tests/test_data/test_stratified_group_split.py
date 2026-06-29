"""Testes para src.data.stratified_group_split."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.stratified_group_split import StratifiedGroupKFold


@pytest.fixture
def toy_data():
    """Dados sintéticos com 4 pacientes e 2 classes desbalanceadas."""
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    return groups, y


def test_no_patient_across_folds(toy_data):
    """Nenhum paciente pode aparecer em mais de um fold de teste."""
    groups, y = toy_data
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)

    test_patients_per_fold = []
    for train_idx, test_idx in splitter.split(X=[0] * len(y), y=y, groups=groups):
        test_patients = set(groups[test_idx])
        test_patients_per_fold.append(test_patients)

    all_test_patients = set().union(*test_patients_per_fold)
    assert len(all_test_patients) == sum(len(p) for p in test_patients_per_fold)


def test_export_json_creates_files(tmp_path, toy_data):
    """JSONs de folds devem ser criados e DataFrame de proporções retornado."""
    groups, y = toy_data
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    output_dir = tmp_path / "splits"

    df = splitter.export_json(groups=groups, y=y, output_dir=output_dir)

    assert output_dir.exists()
    for fold in range(2):
        path = output_dir / f"fold_{fold}.json"
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "train" in payload and "test" in payload
        assert len(payload["train"]) + len(payload["test"]) == len(y)

    assert not df.empty
    assert {"fold", "class", "prop"}.issubset(df.columns)
    assert df["prop"].between(0.0, 1.0).all()


def test_stratified_group_split_reproducible(toy_data):
    """Mesmos dados e seed devem gerar folds idênticos."""
    groups, y = toy_data
    splitter_a = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    splitter_b = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)

    folds_a = list(splitter_a.split(X=[0] * len(y), y=y, groups=groups))
    folds_b = list(splitter_b.split(X=[0] * len(y), y=y, groups=groups))

    assert len(folds_a) == len(folds_b)
    for (train_a, test_a), (train_b, test_b) in zip(folds_a, folds_b):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)
