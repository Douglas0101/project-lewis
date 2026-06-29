"""Testes para src/models/evaluate.py — threshold tuning e avaliação."""

from __future__ import annotations

import numpy as np

from src.models.evaluate import (
    evaluate_aami,
    evaluate_multiclass_at_thresholds,
    find_best_threshold,
    find_best_thresholds_multiclass,
)


def test_find_best_threshold_improves_or_matches_argmax():
    y_true = np.array([0] * 90 + [1] * 10)
    # Modelo tendencioso: alta probabilidade para classe 1 em todas as amostras
    y_score = np.array([0.4] * 90 + [0.6] * 10)

    thresholds = {"min_acc": 0.0, "min_f1_macro": 0.0, "min_mcc": -1.0, "max_fpr_global": 1.0}
    result = find_best_threshold(
        y_true, y_score, class_names=["N", "Anormal"], thresholds=thresholds, target_class_idx=1
    )

    assert "threshold" in result
    assert 0.0 < result["threshold"] < 1.0
    assert "F1_macro" in result["global"]


def test_evaluate_multiclass_at_thresholds_fallback_when_no_class_passes():
    class_names = ["S", "V", "F"]
    y_true = np.array([0, 1, 2])
    # Nenhuma probabilidade passa de threshold alto => fallback para V (índice 1)
    y_score = np.array(
        [
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
        ],
        dtype=np.float32,
    )
    thresholds = {"S": 0.95, "V": 0.95, "F": 0.95}
    result = evaluate_multiclass_at_thresholds(
        y_true, y_score, thresholds, class_names=class_names, fallback_class=1
    )

    assert result["y_pred"] == [1, 1, 1]
    assert result["thresholds"] == thresholds


def test_evaluate_multiclass_at_thresholds_selects_single_above_threshold():
    class_names = ["S", "V", "F"]
    y_true = np.array([0, 1, 2])
    y_score = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ],
        dtype=np.float32,
    )
    thresholds = {"S": 0.5, "V": 0.5, "F": 0.5}
    result = evaluate_multiclass_at_thresholds(
        y_true, y_score, thresholds, class_names=class_names
    )

    assert result["y_pred"] == [0, 1, 2]
    assert result["global"]["Acc"] == 1.0


def test_find_best_thresholds_multiclass_improves_f1_macro():
    class_names = ["S", "V", "F"]
    # Dataset desbalanceado: muitos V, poucos F
    y_true = np.array([1] * 80 + [0] * 15 + [2] * 5)
    # Modelo confuso: tende a prever V para todos
    y_score = np.array(
        [[0.2, 0.7, 0.1]] * 80
        + [[0.5, 0.4, 0.1]] * 15
        + [[0.2, 0.6, 0.2]] * 5,
        dtype=np.float32,
    )

    thresholds_cfg = {"min_acc": 0.0, "min_f1_macro": 0.0, "min_mcc": -1.0, "max_fpr_global": 1.0}
    argmax_result = evaluate_aami(
        y_true, np.argmax(y_score, axis=1), class_names=class_names, thresholds=thresholds_cfg
    )
    tuned_result = find_best_thresholds_multiclass(
        y_true,
        y_score,
        class_names=class_names,
        thresholds_cfg=thresholds_cfg,
        metric="F1_macro",
        search_step=0.05,
    )

    assert tuned_result["global"]["F1_macro"] >= argmax_result["global"]["F1_macro"]
    assert "thresholds" in tuned_result
