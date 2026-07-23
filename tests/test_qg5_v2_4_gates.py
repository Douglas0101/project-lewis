"""Testes dos Quality Gates QG5 v2.4 redesenhados (E04)."""

# pyright: reportArgumentType=false

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.models.qg5_gates import (
    QG5PatientwiseGate,
    QG5PublicationGate,
    QG5SmokeBalancedGate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E04_DIR = PROJECT_ROOT / "experiments" / "stage2_v2.4_research" / "E04_qg5_gates"


def _load_report():
    if not E04_DIR.exists():
        pytest.skip("E04 QG5 gates ainda nao foram executados")
    try:
        with open(E04_DIR / "qg5_v2.4_report.json") as f:
            return json.load(f)
    except Exception as exc:
        raise AssertionError(f"Falha ao carregar relatorio QG5: {exc}") from exc


@pytest.mark.requires_artifacts
def test_qg5_report_exists():
    assert (E04_DIR / "qg5_v2.4_report.json").exists()
    assert (E04_DIR / "qg5_v2.4_report.md").exists()


@pytest.mark.requires_artifacts
def test_smoke_balanced_is_diagnostic_only():
    report = _load_report()
    smoke = next(g for g in report["gates"] if g["name"] == "QG5_SMOKE_BALANCED")
    assert smoke["diagnostic_only"]
    assert smoke["passed"]


@pytest.mark.requires_artifacts
def test_patientwise_fails_honestly():
    """QG5_PATIENTWISE deve falhar porque F1(F) inter-paciente < 0.50."""
    report = _load_report()
    patientwise = next(g for g in report["gates"] if g["name"] == "QG5_PATIENTWISE")
    assert not patientwise["diagnostic_only"]
    assert not patientwise["passed"]
    assert any("F1(F)" in f for f in patientwise["failures"])
    mean_f = patientwise["metrics"]["F1_F_mean"]
    assert mean_f < 0.50, f"mean F1(F)={mean_f:.4f} nao esta abaixo de 0.50"


@pytest.mark.requires_artifacts
def test_publication_status_is_research_candidate():
    report = _load_report()
    assert report["status"] == "RESEARCH_CANDIDATE_NOT_PUBLICATION_READY"
    assert not report["can_publish"]


class _FakeGateResult:
    gate_name = "QG5_PATIENTWISE"
    diagnostic_only = False
    metrics: dict = {}
    failures: list = []
    notes: list = []

    def __init__(self, passed: bool) -> None:
        self.passed = passed
        self.failures = ["F1(F)"] if not passed else []


class _DummyScaler:
    def transform(self, X):
        return X


class _DummyModel:
    """Modelo que retorna argmax alinhado com X (para teste de logica)."""

    def predict(self, X, **kwargs):
        proba = np.zeros((X.shape[0], 3), dtype=np.float32)
        for i in range(X.shape[0]):
            proba[i, i % 3] = 1.0
        return proba


def test_publication_gate_logic():
    """Publicacao so e autorizada quando todos os gates formais passam."""
    X = np.eye(6, dtype=np.float32)
    y = np.array([0, 1, 2] * 2, dtype=np.int64)
    smoke = QG5SmokeBalancedGate(min_f1_v=0.0, min_f1_f=0.0, min_f1_macro=0.0).evaluate(
        _DummyModel(), _DummyScaler(), X, y
    )

    gate = QG5PublicationGate()
    gate.add(smoke)
    gate.add(_FakeGateResult(passed=True))
    assert gate.can_publish()

    gate2 = QG5PublicationGate()
    gate2.add(smoke)
    gate2.add(_FakeGateResult(passed=False))
    assert not gate2.can_publish()


def test_smoke_balanced_on_dummy():
    """Smoke balanced executa e e marcado como diagnostico."""
    X = np.eye(6, dtype=np.float32)
    y = np.array([0, 1, 2] * 2, dtype=np.int64)
    gate = QG5SmokeBalancedGate(min_f1_v=0.0, min_f1_f=0.0, min_f1_macro=0.0)
    result = gate.evaluate(_DummyModel(), _DummyScaler(), X, y)
    assert result.diagnostic_only


@pytest.mark.requires_artifacts
def test_patientwise_gate_returns_aggregated_metrics():
    npz = np.load(PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features.npz")
    X, y, groups = npz["X"], npz["y"], npz["groups"]
    gate = QG5PatientwiseGate()
    result = gate.evaluate(
        PROJECT_ROOT / "experiments" / "stage2_mlp_features_v2.3_focal_smote_v14",
        X,
        y,
        groups,
    )
    assert "F1_F_mean" in result.metrics
    assert "F1_F_min" in result.metrics
    assert "per_fold" in result.metrics
    assert len(result.metrics["per_fold"]) == 5
