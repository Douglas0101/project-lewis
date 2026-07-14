"""Testes da engenharia de features E06."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E06_DIR = PROJECT_ROOT / "experiments" / "stage2_v2.4_research" / "E06_feature_engineering"


def _load_enhanced():
    if not (E06_DIR / "baseline_enhanced_metrics.json").exists():
        pytest.skip("E06 baseline enhanced ainda nao foi executado")
    try:
        with open(E06_DIR / "baseline_enhanced_metrics.json") as f:
            return json.load(f)
    except Exception as exc:
        raise AssertionError(f"Falha ao carregar metrics enhanced: {exc}") from exc


def _load_original():
    original_path = E06_DIR / "baseline_original" / "baseline_enhanced_metrics.json"
    if not original_path.exists():
        pytest.skip("E06 baseline original ainda nao foi executado")
    try:
        with open(original_path) as f:
            return json.load(f)
    except Exception as exc:
        raise AssertionError(f"Falha ao carregar metrics original: {exc}") from exc


def test_enhanced_artifacts_exist():
    npz_path = PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features_enhanced_e06_v1.npz"
    json_path = (
        PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features_enhanced_e06_v1.json"
    )
    assert npz_path.exists()
    assert json_path.exists()


def test_enhanced_has_more_features():
    json_path = (
        PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features_enhanced_e06_v1.json"
    )
    try:
        with open(json_path) as f:
            meta = json.load(f)
    except Exception as exc:
        raise AssertionError(f"Falha ao carregar metadata enhanced: {exc}") from exc
    assert meta["n_features"] >= 16
    assert "derived_features" in meta
    assert len(meta["derived_features"]) >= 10


def test_enhanced_baseline_ran():
    enhanced = _load_enhanced()
    assert "mean" in enhanced
    assert "f1_F" in enhanced["mean"]
    assert "fold_metrics" in enhanced
    assert len(enhanced["fold_metrics"]) == 5


def test_original_baseline_ran():
    original = _load_original()
    assert "mean" in original
    assert "f1_F" in original["mean"]


def test_hypothesis_not_supported_by_f1_f():
    """As features enhanced nao superam o baseline original em F1(F)."""
    enhanced = _load_enhanced()
    original = _load_original()
    f1_f_enhanced = enhanced["mean"]["f1_F"]
    f1_f_original = original["mean"]["f1_F"]
    # A hipotese de melhoria nao e suportada; a diferenca deve ser pequena ou negativa.
    assert f1_f_enhanced - f1_f_original < 0.05, (
        f"F1(F) enhanced={f1_f_enhanced:.4f} nao supera original={f1_f_original:.4f} "
        "de forma significativa; hipotese rejeitada."
    )
