"""Testes da auditoria de separabilidade E05."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E05_DIR = PROJECT_ROOT / "experiments" / "stage2_v2.4_research" / "E05_feature_separability"


def _load_report():
    if not E05_DIR.exists():
        pytest.skip("E05 feature separability ainda nao foi executada")
    try:
        with open(E05_DIR / "feature_separability_report.json") as f:
            return json.load(f)
    except Exception as exc:
        raise AssertionError(f"Falha ao carregar relatorio E05: {exc}") from exc


def test_report_artifacts_exist():
    assert (E05_DIR / "feature_separability_report.json").exists()
    assert (E05_DIR / "feature_separability_report.md").exists()
    assert (E05_DIR / "feature_separability_summary.csv").exists()


def test_f_separable_inside_208_213():
    """F deve ser separavel dentro dos registros 208/213 (MI positiva)."""
    report = _load_report()
    mi_208 = report["regimes"]["record_208"]["mutual_information"]["F_vs_rest"]
    mi_213 = report["regimes"]["record_213"]["mutual_information"]["F_vs_rest"]
    assert max(mi_208.values()) > 0.0
    assert max(mi_213.values()) > 0.0


def test_f_not_generalizing_outside_208_213():
    """F nao generaliza fora dos registros 208/213."""
    report = _load_report()
    lgo = report["leave_group_out"]
    f1_208 = lgo["without_group_27"]["f1_F"]
    f1_213 = lgo["without_group_30"]["f1_F"]
    assert f1_208 < 0.30, f"F1(F) sem 208={f1_208:.4f} nao indica falta de generalizacao"
    assert f1_213 < 0.50, f"F1(F) sem 213={f1_213:.4f} nao indica falta de generalizacao"


def test_top_features_are_rr_dominant():
    """Top features por permutation importance devem incluir dominancia de RR."""
    report = _load_report()
    perm = report["permutation_importance"]
    top = sorted(perm.items(), key=lambda x: x[1], reverse=True)[:5]
    top_names = [n for n, _ in top]
    rr_features = {"rr_prev", "rr_next", "rr_local_mean", "rr_local_std", "rr_ratio"}
    overlap = rr_features.intersection(top_names)
    assert len(overlap) >= 3, f"Top features nao dominadas por RR: {top_names}"


def test_lofo_baseline_is_low():
    """Baseline F1-macro global do RandomForest e baixo, confirmando separabilidade fraca."""
    report = _load_report()
    baseline = report["leave_one_feature_out"]["baseline_f1_macro"]
    assert 0.30 < baseline < 0.80, f"baseline F1-macro={baseline:.4f} fora da faixa esperada"
