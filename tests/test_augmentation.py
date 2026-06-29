"""Testes para src/data/augmentation.py."""

from __future__ import annotations

import numpy as np

from src.data.augmentation import ecg_time_augment, oversample_class, oversample_per_class


def test_oversample_class_increases_minority_count():
    X = np.random.randn(10, 500, 1).astype(np.float32)
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    X_out, y_out = oversample_class(X, y, class_idx=1, factor=3)
    assert len(X_out) == 10 + 2 * 3  # 10 originais + 6 novas da classe 1
    assert (y_out == 1).sum() == 9
    assert (y_out == 0).sum() == 7
    assert X_out.shape[1:] == (500, 1)


def test_oversample_class_factor_one_returns_same():
    X = np.random.randn(5, 500, 1).astype(np.float32)
    y = np.array([0, 0, 1, 1, 1])
    X_out, y_out = oversample_class(X, y, class_idx=1, factor=1)
    assert len(X_out) == len(X)
    assert np.array_equal(y_out, y)


def test_oversample_class_no_samples_returns_same():
    X = np.random.randn(5, 500, 1).astype(np.float32)
    y = np.array([0, 0, 0, 0, 0])
    X_out, y_out = oversample_class(X, y, class_idx=1, factor=5)
    assert len(X_out) == len(X)
    assert np.array_equal(y_out, y)


def test_ecg_time_augment_preserves_shape():
    X = np.random.randn(8, 500, 1).astype(np.float32)
    rng = np.random.default_rng(42)
    X_aug = ecg_time_augment(X, rng=rng)
    assert X_aug.shape == X.shape
    # Augmentation deve alterar os valores (com alta probabilidade)
    assert not np.allclose(X_aug, X)


def test_oversample_per_class_increases_configured_classes():
    X = np.random.randn(12, 500, 1).astype(np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2])
    config = {
        "1": {"factor": 3, "methods": ["jitter"], "intensity": "low"},
        "2": {"factor": 2, "methods": ["time_warp"], "intensity": "low"},
    }
    X_out, y_out = oversample_per_class(X, y, config=config, seed=42)

    # classe 1: 2 originais + 2*(3-1) = 6
    # classe 2: 6 originais + 6*(2-1) = 12
    # classe 0: 4
    assert (y_out == 0).sum() == 4
    assert (y_out == 1).sum() == 6
    assert (y_out == 2).sum() == 12
    assert len(X_out) == len(y_out) == 22
    assert X_out.shape[1:] == (500, 1)


def test_oversample_per_class_factor_one_unchanged():
    X = np.random.randn(10, 500, 1).astype(np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2])
    config = {
        "1": {"factor": 1, "methods": ["jitter"], "intensity": "low"},
    }
    X_out, y_out = oversample_per_class(X, y, config=config, seed=42)
    assert len(X_out) == len(X)
    assert np.array_equal(y_out, y)


def test_oversample_per_class_missing_class_is_noop():
    X = np.random.randn(10, 500, 1).astype(np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2])
    config = {
        "3": {"factor": 5, "methods": ["jitter"], "intensity": "low"},
    }
    X_out, y_out = oversample_per_class(X, y, config=config, seed=42)
    assert len(X_out) == len(X)
    assert np.array_equal(y_out, y)
