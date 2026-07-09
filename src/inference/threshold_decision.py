"""Shared one-vs-rest threshold decision logic for multiclass classifiers."""

from __future__ import annotations

import numpy as np


def predict_with_thresholds(
    scores: np.ndarray,
    thresholds: dict[str, float],
    class_names: list[str],
    fallback_class: int = 1,
) -> np.ndarray:
    """Apply per-class thresholds and resolve ties.

    Parameters
    ----------
    scores : np.ndarray
        Softmax probabilities, shape (n_samples, n_classes).
    thresholds : dict[str, float]
        Threshold per class name.
    class_names : list[str]
        Ordered class names matching columns of ``scores``.
    fallback_class : int
        Class index used when no class exceeds its threshold.

    Returns
    -------
    np.ndarray
        Predicted class indices, shape (n_samples,).
    """
    n_samples = scores.shape[0]
    thr_array = np.array(
        [thresholds.get(name, 0.5) for name in class_names],
        dtype=np.float32,
    )
    above = scores >= thr_array
    y_pred = np.full(n_samples, fallback_class, dtype=np.int64)

    single = above.sum(axis=1) == 1
    y_pred[single] = np.argmax(above[single], axis=1)

    multi = above.sum(axis=1) > 1
    if multi.any():
        masked = scores.copy()
        masked[~above] = -1.0
        y_pred[multi] = np.argmax(masked[multi], axis=1)

    none_above = above.sum(axis=1) == 0
    if none_above.any():
        y_pred[none_above] = np.argmax(scores[none_above], axis=1)

    return y_pred
