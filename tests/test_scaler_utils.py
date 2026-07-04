import numpy as np
from src.features.scaler_utils import fit_feature_scaler_on_train, scale_features


def test_scaler_fitted_on_train_only():
    np.random.seed(42)
    X = np.random.randn(100, 12).astype(np.float32)
    groups = np.repeat(np.arange(20), 5)
    scaler, train_idx, val_idx = fit_feature_scaler_on_train(X, groups, n_splits=5)

    assert len(set(train_idx) & set(val_idx)) == 0

    X_train_scaled = scale_features(X[train_idx], scaler)
    X_val_scaled = scale_features(X[val_idx], scaler)

    assert not np.any(np.isnan(X_train_scaled))
    assert not np.any(np.isnan(X_val_scaled))
    assert np.allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(X_train_scaled.std(axis=0), 1.0, atol=1e-6)


def test_scaler_no_group_overlap():
    np.random.seed(42)
    X = np.random.randn(60, 5).astype(np.float32)
    groups = np.repeat(np.arange(10), 6)
    _, train_idx, val_idx = fit_feature_scaler_on_train(X, groups, n_splits=5)
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    assert len(train_groups & val_groups) == 0
