"""Testes da reescrita de rótulos e reamostragem de F (E07)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E07_DIR = PROJECT_ROOT / "experiments" / "stage2_v2.4_research" / "E07_label_audit"


def _load_audit():
    if not (E07_DIR / "label_audit_report.json").exists():
        pytest.skip("E07 label audit ainda nao foi executado")
    try:
        with open(E07_DIR / "label_audit_report.json") as f:
            return json.load(f)
    except Exception as exc:
        raise AssertionError(f"Falha ao carregar audit E07: {exc}") from exc


def _load_resampled():
    if not (E07_DIR / "baseline_resampled" / "baseline_enhanced_metrics.json").exists():
        pytest.skip("E07 baseline resampled ainda nao foi executado")
    try:
        with open(E07_DIR / "baseline_resampled" / "baseline_enhanced_metrics.json") as f:
            return json.load(f)
    except Exception as exc:
        raise AssertionError(f"Falha ao carregar metrics resampled: {exc}") from exc


def _load_original():
    p = (
        PROJECT_ROOT
        / "experiments"
        / "stage2_v2.4_research"
        / "E06_feature_engineering"
        / "baseline_original"
    )
    if not (p / "baseline_enhanced_metrics.json").exists():
        pytest.skip("E06 baseline original nao encontrado")
    try:
        with open(p / "baseline_enhanced_metrics.json") as f:
            return json.load(f)
    except Exception as exc:
        raise AssertionError(f"Falha ao carregar metrics original: {exc}") from exc


def test_label_audit_artifacts_exist():
    assert (E07_DIR / "label_audit_report.json").exists()
    assert (E07_DIR / "label_audit_report.md").exists()
    assert (E07_DIR / "label_cooccurrence.csv").exists()


def test_f_occurs_in_multiple_records():
    audit = _load_audit()
    assert audit["records_with_f"] >= 10
    assert audit["records_with_f"] <= audit["records_total"]


def test_label_mapping_is_consistent():
    """AAMI label F mapeia 1:1 para y=2."""
    audit = _load_audit()
    assert audit["class_counts"]["F"] == 1044
    assert audit["class_counts"]["S"] == 16934
    assert audit["class_counts"]["V"] == 37183


def test_resampled_artifact_exists():
    assert (
        PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features_resampled_e07_v1.npz"
    ).exists()
    assert (
        PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features_resampled_e07_v1.json"
    ).exists()


def test_resampled_improves_f1_f():
    """Reamostragem por paciente melhora F1(F) inter-paciente."""
    resampled = _load_resampled()
    original = _load_original()
    f1_f_resampled = resampled["mean"]["f1_F"]
    f1_f_original = original["mean"]["f1_F"]
    assert (
        f1_f_resampled > f1_f_original + 0.05
    ), f"F1(F) resampled={f1_f_resampled:.4f} nao supera original={f1_f_original:.4f}"


def test_resampled_f1_f_towards_target():
    resampled = _load_resampled()
    f1_f = resampled["mean"]["f1_F"]
    assert f1_f >= 0.35, f"F1(F) resampled={f1_f:.4f} muito abaixo de 0.35"
