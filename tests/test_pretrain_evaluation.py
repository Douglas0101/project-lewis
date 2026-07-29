"""Tests for advanced evaluation (FASE 8)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.pretrain_evaluation import (
    apply_temperature,
    brier_score,
    calibration_summary,
    confusion_per_class,
    ece_mce,
    fit_temperature,
    nll_multilabel,
    sigmoid_to_logits,
)


def test_ece_zero_when_perfectly_calibrated():
    y_true = np.array([0, 0, 1, 1], dtype=float)
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    ece, mce = ece_mce(y_true, y_prob)
    assert ece == pytest.approx(0.0)
    assert mce == pytest.approx(0.0)
    assert brier_score(y_true, y_prob) == pytest.approx(0.0)


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(7)
    n = 4000
    y_true = rng.binomial(1, 0.3, size=(n, 5)).astype(float)
    # overconfident: push true labels to extreme probabilities, com ruído
    y_prob = np.clip(
        y_true * 0.98 + rng.normal(0, 0.02, (n, 5)) + (1 - y_true) * 0.02,
        1e-4,
        1 - 1e-4,
    )
    # 20% dos rótulos errados mantém o problema honesto
    flip = rng.random((n, 5)) < 0.2
    y_prob = np.where(flip, 1.0 - y_prob, y_prob)

    nll_before = nll_multilabel(y_true, sigmoid_to_logits(y_prob))
    t = fit_temperature(y_true, y_prob)
    y_cal = apply_temperature(y_prob, t)
    nll_after = nll_multilabel(y_true, sigmoid_to_logits(y_cal))

    assert t > 1.0, "overconfidence deve exigir T>1"
    assert nll_after < nll_before


def test_temperature_near_one_when_calibrated():
    rng = np.random.default_rng(3)
    base = rng.uniform(0.05, 0.95, size=(2000, 5))
    y_true = (rng.uniform(size=base.shape) < base).astype(float)
    t = fit_temperature(y_true, base)
    assert 0.5 < t < 2.0


def test_confusion_per_class_counts():
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=int)
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.9], [0.1, 0.2]])
    conf = confusion_per_class(y_true, y_prob, 0.5)
    assert conf["NORM"] == {"tp": 1, "fp": 0, "tn": 2, "fn": 1}
    assert conf["CD"] == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}


def test_calibration_summary_schema():
    rng = np.random.default_rng(11)
    y_prob = rng.uniform(size=(500, 5))
    y_true = (rng.uniform(size=(500, 5)) < 0.3).astype(int)
    summary = calibration_summary(y_true, y_prob)
    assert set(summary["per_class"]) == {"NORM", "CD", "MI", "HYP", "STTC"}
    assert 0.0 <= summary["macro"]["ece"] <= 1.0
    assert 0.0 <= summary["macro"]["brier"] <= 1.0


def test_reliability_bins_cover_all_samples():
    rng = np.random.default_rng(5)
    y_prob = rng.uniform(size=(200, 5))
    y_true = (rng.uniform(size=(200, 5)) < 0.5).astype(int)
    rel = calibration_summary(y_true, y_prob)["reliability"]
    for cls, rows in rel["per_class"].items():
        assert sum(r["count"] for r in rows) == 200, cls
        for row in rows:
            if row["count"]:
                assert 0.0 <= row["mean_obs"] <= 1.0
                assert 0.0 <= row["mean_pred"] <= 1.0
