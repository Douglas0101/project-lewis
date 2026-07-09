"""Tests for src.inference.threshold_decision."""

from __future__ import annotations

import numpy as np

from src.inference.threshold_decision import predict_with_thresholds


CLASS_NAMES = ["S", "V", "F"]


def test_single_class_above_threshold() -> None:
    scores = np.array([[0.6, 0.3, 0.1]], dtype=np.float32)
    thresholds = {"S": 0.5, "V": 0.5, "F": 0.5}
    result = predict_with_thresholds(scores, thresholds, CLASS_NAMES, fallback_class=1)
    assert result.tolist() == [0]


def test_multiple_classes_above_threshold_tie_break() -> None:
    scores = np.array([[0.7, 0.6, 0.2]], dtype=np.float32)
    thresholds = {"S": 0.5, "V": 0.5, "F": 0.5}
    result = predict_with_thresholds(scores, thresholds, CLASS_NAMES, fallback_class=1)
    assert result.tolist() == [0]


def test_no_class_above_threshold_uses_fallback() -> None:
    scores = np.array([[0.9, 0.05, 0.05]], dtype=np.float32)  # argmax S, but below threshold
    thresholds = {"S": 0.95, "V": 0.5, "F": 0.5}
    result = predict_with_thresholds(scores, thresholds, CLASS_NAMES, fallback_class=1)
    assert result.tolist() == [1]


def test_batch_with_mixed_cases() -> None:
    scores = np.array(
        [
            [0.6, 0.3, 0.1],  # S above
            [0.3, 0.6, 0.1],  # V above
            [0.4, 0.4, 0.2],  # none above -> fallback V
            [0.5, 0.5, 0.0],  # S and V above -> tie break S
        ],
        dtype=np.float32,
    )
    thresholds = {"S": 0.5, "V": 0.5, "F": 0.5}
    result = predict_with_thresholds(scores, thresholds, CLASS_NAMES, fallback_class=1)
    assert result.tolist() == [0, 1, 1, 0]


def test_missing_class_name_uses_default_threshold() -> None:
    scores = np.array([[0.6, 0.3, 0.1]], dtype=np.float32)
    thresholds = {"S": 0.5, "V": 0.5}  # F missing
    result = predict_with_thresholds(scores, thresholds, CLASS_NAMES, fallback_class=1)
    assert result.tolist() == [0]
