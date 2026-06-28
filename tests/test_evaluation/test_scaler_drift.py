import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.evaluation.scaler_drift import check_drift, compute_drift_metrics


@pytest.fixture
def fitted_scaler():
    rng = np.random.RandomState(42)
    base = rng.normal(loc=0.0, scale=1.0, size=(1000, 2))
    scaler = StandardScaler()
    scaler.fit(base)
    return scaler


def test_drift_detects_large_mean_shift(fitted_scaler):
    rng = np.random.RandomState(1)
    batch = rng.normal(loc=5.0, scale=1.0, size=(100, 2))
    assert check_drift(batch, fitted_scaler) is False


def test_drift_safe_batch_returns_true(fitted_scaler):
    rng = np.random.RandomState(2)
    batch = rng.normal(loc=0.0, scale=1.0, size=(1000, 2))
    batch = (batch - batch.mean(axis=0)) / batch.std(axis=0)
    assert check_drift(batch, fitted_scaler) is True


def test_compute_drift_metrics_keys(fitted_scaler):
    rng = np.random.RandomState(3)
    batch = rng.normal(loc=0.0, scale=1.0, size=(100, 2))
    metrics = compute_drift_metrics(batch, fitted_scaler)
    assert set(metrics.keys()) == {"mean_drift", "scale_drift"}


def test_compute_drift_metrics_invalid_input(fitted_scaler):
    with pytest.raises(ValueError, match="batch deve ser ndarray 2D não vazio"):
        compute_drift_metrics(np.array([]), fitted_scaler)

    valid_batch = np.random.RandomState(4).normal(loc=0.0, scale=1.0, size=(10, 2))
    invalid_scaler = object()
    with pytest.raises(ValueError, match="scaler deve ter mean_ e scale_"):
        compute_drift_metrics(valid_batch, invalid_scaler)
