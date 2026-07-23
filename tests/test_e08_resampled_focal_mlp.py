"""Testes do treinamento E08 com dataset resampled e focal loss."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E08_DIR = PROJECT_ROOT / "experiments" / "stage2_v2.4_research" / "E08_resampled_focal_mlp"


def _load_summary():
    if not (E08_DIR / "summary.json").exists():
        pytest.skip("E08 treinamento ainda nao foi executado")
    try:
        with open(E08_DIR / "summary.json") as f:
            return json.load(f)
    except Exception as exc:
        raise AssertionError(f"Falha ao carregar summary E08: {exc}") from exc


def test_e08_summary_exists():
    assert (E08_DIR / "summary.json").exists()


def test_e08_mean_f1_macro_improved():
    summary = _load_summary()
    f1_macro = summary["mean"]["F1_macro"]
    assert f1_macro >= 0.55, f"F1-macro={f1_macro:.4f} nao atingiu 0.55"


def test_e08_mean_f1_f_towards_target():
    summary = _load_summary()
    f1_f = summary["mean"]["F1_F"]
    assert f1_f >= 0.40, f"F1(F)={f1_f:.4f} nao atingiu 0.40"


def test_e08_config_uses_resampled_and_focal():
    summary = _load_summary()
    assert summary["class_weight"] == [1.0, 1.0, 8.0]
    assert summary["focal_alpha"] == [0.20, 0.15, 3.00]
    assert summary["optimize_thresholds"]
    assert summary["hidden_units"] == 256


def test_e08_thresholds_saved():
    assert (E08_DIR / "stage2_thresholds.json").exists()
