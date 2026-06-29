"""Testes para o pipeline integrado de duas etapas (v2.0)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.backbone_1d import build_backbone_1d
from src.models.evaluate import find_best_threshold
from src.models.two_stage_pipeline import build_integrated_predictions, evaluate_two_stage


@pytest.mark.parametrize(
    "stage1_pred, stage2_pred, expected",
    [
        (
            np.array([0, 1, 1, 1]),
            np.array([0, 1, 2]),  # S=0, V=1, F=2 (apenas amostras Anormal)
            np.array([0, 1, 2, 3]),  # N, S, V, F
        ),
        (
            np.array([0, 0, 0]),
            np.array([]),
            np.array([0, 0, 0]),
        ),
    ],
)
def test_build_integrated_predictions(stage1_pred, stage2_pred, expected):
    result = build_integrated_predictions(stage1_pred, stage2_pred)
    np.testing.assert_array_equal(result, expected)


def test_find_best_threshold_passes_qg():
    # Dados quase separáveis: N tem score baixo, Anormal tem score alto
    y_true = np.array([0] * 100 + [1] * 50)
    y_score = np.concatenate(
        [np.random.rand(100) * 0.4, 0.6 + np.random.rand(50) * 0.4]
    )
    thresholds = {
        "min_acc": 0.85,
        "min_f1_macro": 0.80,
        "min_mcc": 0.60,
        "max_fpr_global": 0.20,
        "per_class": {
            "N": {"Se": 0.80},
            "Anormal": {"Se": 0.90, "PPV": 0.70},
        },
    }
    result = find_best_threshold(
        y_true,
        y_score,
        class_names=["N", "Anormal"],
        thresholds=thresholds,
    )
    assert result["passes_qg5"]
    assert "threshold" in result
    assert 0.0 < result["threshold"] < 1.0


def test_evaluate_two_stage_no_index_error():
    """Regressão: avaliação integrada não deve falhar por shape mismatch."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 500, 1)).astype(np.float32)
    y_aami = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 1, 2, 3, 4, 0, 1, 2])

    stage1_model = build_backbone_1d(input_len=500, num_classes=2)
    stage2_model = build_backbone_1d(input_len=500, num_classes=3)

    # Scalers dummy: mean=0, std=1 (StandardScaler-like)
    from sklearn.preprocessing import StandardScaler

    stage1_scaler = StandardScaler()
    stage1_scaler.fit(X.reshape(-1, 1))
    stage2_scaler = StandardScaler()
    stage2_scaler.fit(X.reshape(-1, 1))

    results = evaluate_two_stage(
        stage1_model=stage1_model,
        stage1_scaler=stage1_scaler,
        stage1_threshold=0.5,
        stage2_model=stage2_model,
        stage2_scaler=stage2_scaler,
        X=X,
        y_aami=y_aami,
    )
    assert "stage1" in results
    assert "stage2" in results
    assert "integrated" in results
    assert len(results["integrated"]["per_class"]) == 4


def test_find_best_threshold_no_pass():
    y_true = np.array([0] * 100 + [1] * 50)
    # Scores aleatórios não separáveis
    rng = np.random.default_rng(42)
    y_score = rng.random(150)
    thresholds = {
        "min_acc": 0.99,
        "min_f1_macro": 0.99,
        "min_mcc": 0.99,
        "max_fpr_global": 0.01,
        "per_class": {
            "N": {"Se": 0.99},
            "Anormal": {"Se": 0.99, "PPV": 0.99},
        },
    }
    result = find_best_threshold(
        y_true,
        y_score,
        class_names=["N", "Anormal"],
        thresholds=thresholds,
    )
    assert not result["passes_qg5"]
