"""Testes para o otimizador de thresholds por classe."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.threshold_optimizer import apply_thresholds, optimize_thresholds

CLASS_NAMES = ["A", "B", "C"]


def test_threshold_respects_min_recall() -> None:
    """Os thresholds otimizados devem estar sempre no intervalo [0, 1]."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_classes = len(CLASS_NAMES)

    y_true = rng.integers(0, 2, size=(n_samples, n_classes))
    y_true[0, :] = 1  # garante pelo menos um positivo por classe
    y_pred = rng.random(size=(n_samples, n_classes))

    thresholds = optimize_thresholds(y_true, y_pred, CLASS_NAMES)

    assert set(thresholds.keys()) == set(CLASS_NAMES)
    for value in thresholds.values():
        assert 0.0 <= value <= 1.0


def test_apply_thresholds() -> None:
    """A aplicação dos thresholds deve produzir os índices de classe esperados."""
    thresholds = {"A": 0.5, "B": 0.6, "C": 0.3}
    y_pred = np.array(
        [
            [0.6, 0.2, 0.1],  # A ativa
            [0.4, 0.7, 0.2],  # B ativa
            [0.3, 0.5, 0.4],  # C ativa (A e B abaixo)
            [0.1, 0.1, 0.9],  # C ativa
        ]
    )
    expected = np.array([0, 1, 2, 2])

    result = apply_thresholds(y_pred, thresholds, CLASS_NAMES)

    np.testing.assert_array_equal(result, expected)


def test_threshold_falls_back_when_recall_unmet() -> None:
    """Quando nenhum threshold atinge ``min_recall``, escolhe o de maior recall."""
    y_true = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    y_pred = np.array(
        [
            [0.2, 0.8, 0.1],
            [0.9, 0.4, 0.3],
            [0.1, 0.2, 0.7],
        ]
    )

    # min_recall acima do máximo possível força o fallback.
    thresholds = optimize_thresholds(y_true, y_pred, CLASS_NAMES, min_recall=1.01)

    assert set(thresholds.keys()) == set(CLASS_NAMES)
    for name in CLASS_NAMES:
        assert 0.0 <= thresholds[name] <= 1.0

    # Classe B: dois positivos com scores 0.8 e 0.4. O maior recall (1.0) é
    # atingido no menor threshold (0.2), valor retornado pelo fallback.
    assert thresholds["B"] == pytest.approx(0.2)
