"""Tests for advanced evaluation (FASE 8)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.models.pretrain_evaluation import (
    apply_temperature,
    brier_score,
    calibration_summary,
    confusion_per_class,
    ece_mce,
    evaluate_predictions,
    fit_temperature,
    nll_multilabel,
    sigmoid_to_logits,
    write_evaluation_reports,
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


def test_evaluate_predictions_propagates_n_bins():
    rng = np.random.default_rng(21)
    y_prob = rng.uniform(size=(400, 5))
    y_true = (rng.uniform(size=(400, 5)) < 0.3).astype(int)
    report = evaluate_predictions(y_true, y_prob, n_bins=15)
    assert report["calibration_before"]["n_bins"] == 15
    assert report["temperature_scaling"]["calibration_after"]["n_bins"] == 15
    for rows in report["calibration_before"]["reliability"]["per_class"].values():
        assert len(rows) == 15


def test_temperature_scaling_preserves_macro_auc_roc():
    rng = np.random.default_rng(33)
    n = 3000
    logits = rng.normal(0, 1.5, size=(n, 5))
    y_prob = 1.0 / (1.0 + np.exp(-logits))
    y_true = (rng.uniform(size=(n, 5)) < y_prob).astype(int)
    report = evaluate_predictions(y_true, y_prob)
    ts = report["temperature_scaling"]
    assert ts["auc_roc_macro_before"] is not None
    assert ts["auc_roc_macro_after"] == pytest.approx(ts["auc_roc_macro_before"], abs=1e-6)


def test_write_evaluation_reports_merges_contract_metadata(tmp_path):
    rng = np.random.default_rng(8)
    y_prob = rng.uniform(size=(300, 5))
    y_true = (rng.uniform(size=(300, 5)) < 0.3).astype(int)
    report = evaluate_predictions(y_true, y_prob, n_bins=15)
    contract = {
        "run_id": "test_run",
        "model_id": "a2_focal",
        "n_params": 32005,
        "val_samples": 300,
        "split_version": "chapman-record-disjoint-val0.1-seed13",
        "seed": 13,
        "created_at": "2026-07-31T00:00:00+00:00",
        "sha256_model": "ab" * 32,
    }
    write_evaluation_reports(tmp_path, report, contract=contract)
    cal = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    for key, value in contract.items():
        assert cal[key] == value, key
    assert cal["n_bins"] == 15
    assert cal["temperature"] == pytest.approx(report["temperature_scaling"]["temperature"])
    assert cal["ece_before"] == pytest.approx(report["calibration_before"]["macro"]["ece"])
    assert cal["ece_after"] == pytest.approx(
        report["temperature_scaling"]["calibration_after"]["macro"]["ece"]
    )
    assert "before" in cal and "temperature_scaling" in cal


def test_write_evaluation_reports_without_contract_keeps_legacy_schema(tmp_path):
    rng = np.random.default_rng(9)
    y_prob = rng.uniform(size=(200, 5))
    y_true = (rng.uniform(size=(200, 5)) < 0.3).astype(int)
    report = evaluate_predictions(y_true, y_prob)
    write_evaluation_reports(tmp_path, report)
    cal = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    assert set(cal) == {"before", "temperature_scaling"}
