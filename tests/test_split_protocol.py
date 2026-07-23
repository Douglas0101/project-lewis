"""Testes de integridade do protocolo de split (E03)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.models.split_protocol import SplitConfig, SplitProtocol, SplitterName

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E03_DIR = PROJECT_ROOT / "experiments" / "stage2_v2.4_research" / "E03_split_protocol"


def _load_distribution():
    dist_path = PROJECT_ROOT / "experiments" / "stage2_v2.4_research" / "E01_patient_distribution"
    if not dist_path.exists():
        pytest.skip("E01 patient distribution audit ainda nao foi executado")
    import pandas as pd

    return pd.read_csv(dist_path / "patient_class_distribution.csv")


@pytest.mark.requires_artifacts
def test_groupkfold_no_overlap():
    npz = np.load(PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features.npz")
    X, y, groups = npz["X"], npz["y"], npz["groups"]
    protocol = SplitProtocol(SplitConfig(SplitterName.GROUP_K_FOLD, n_splits=5))
    for train_idx, test_idx in protocol.split(X=X, y=y, groups=groups):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)


@pytest.mark.requires_artifacts
def test_stratified_groupkfold_no_overlap():
    npz = np.load(PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features.npz")
    X, y, groups = npz["X"], npz["y"], npz["groups"]
    protocol = SplitProtocol(
        SplitConfig(SplitterName.STRATIFIED_GROUP_K_FOLD, n_splits=5, shuffle=True, random_state=42)
    )
    for train_idx, test_idx in protocol.split(X=X, y=y, groups=groups):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)


def test_split_manifest_exists():
    if not E03_DIR.exists():
        pytest.skip("E03 split audit ainda nao foi executado")
    assert (E03_DIR / "split_manifest_GroupKFold.json").exists()
    assert (E03_DIR / "split_manifest_StratifiedGroupKFold.json").exists()
    assert (E03_DIR / "split_diagnostics.csv").exists()
    assert (E03_DIR / "split_diagnostics_report.json").exists()


def _load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        raise AssertionError(f"Falha ao carregar {path}: {exc}") from exc


def test_split_manifest_zero_overlap():
    if not E03_DIR.exists():
        pytest.skip("E03 split audit ainda nao foi executado")
    for name in ["GroupKFold", "StratifiedGroupKFold"]:
        manifest = _load_json(E03_DIR / f"split_manifest_{name}.json")
        for fold in manifest["folds"]:
            assert (
                fold["overlap_groups"] == []
            ), f"{name} fold {fold['fold']} possui overlap de grupos"


def test_split_manifest_f_presence():
    """Todos os folds devem conter ao menos um exemplo da classe F no teste."""
    if not E03_DIR.exists():
        pytest.skip("E03 split audit ainda nao foi executado")
    for name in ["GroupKFold", "StratifiedGroupKFold"]:
        manifest = _load_json(E03_DIR / f"split_manifest_{name}.json")
        for fold in manifest["folds"]:
            assert (
                fold["test_counts"].get("2", 0) > 0
            ), f"{name} fold {fold['fold']} nao possui F no teste"


def test_split_manifest_hashes_unique():
    if not E03_DIR.exists():
        pytest.skip("E03 split audit ainda nao foi executado")
    for name in ["GroupKFold", "StratifiedGroupKFold"]:
        manifest = _load_json(E03_DIR / f"split_manifest_{name}.json")
        test_hashes = [f["test_idx_hash"] for f in manifest["folds"]]
        assert len(test_hashes) == len(
            set(test_hashes)
        ), f"{name} possui folds de teste com hashes duplicados"


def test_selected_splitter_is_justified():
    if not E03_DIR.exists():
        pytest.skip("E03 split audit ainda nao foi executado")
    report = _load_json(E03_DIR / "split_diagnostics_report.json")
    assert report["selected_splitter"] in ["GroupKFold", "StratifiedGroupKFold"]
    assert report["selection_reason"] != ""


def test_split_manifest_matches_config():
    if not E03_DIR.exists():
        pytest.skip("E03 split audit ainda nao foi executado")
    for name in ["GroupKFold", "StratifiedGroupKFold"]:
        manifest = _load_json(E03_DIR / f"split_manifest_{name}.json")
        assert manifest["split_config"]["splitter"] == name
        assert manifest["n_samples"] == 55161
        assert manifest["n_groups"] == 197
        assert len(manifest["folds"]) == 5


def test_records_208_213_in_some_test_fold():
    """Registros 208 e 213 devem aparecer em test sets de algum fold."""
    if not E03_DIR.exists():
        pytest.skip("E03 split audit ainda nao foi executado")
    manifest = _load_json(E03_DIR / "split_manifest_GroupKFold.json")
    all_test_groups = set()
    for fold in manifest["folds"]:
        all_test_groups.update(fold["test_groups"])
    assert 27 in all_test_groups, "registro 208 (group 27) nunca aparece no teste"
    assert 30 in all_test_groups, "registro 213 (group 30) nunca aparece no teste"
