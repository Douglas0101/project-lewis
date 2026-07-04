import numpy as np
import pandas as pd
from src.features.scaler_utils import fit_feature_scaler_on_train, scale_features


def test_prepare_stage1_features_scaler_logic():
    np.random.seed(42)
    n = 50
    records = [f"r{i % 10}" for i in range(n)]
    feature_cols = [
        "rr_prev",
        "rr_next",
        "rr_ratio",
        "rr_local_mean",
        "rr_local_std",
        "rmssd",
        "heart_rate",
        "r_amplitude",
        "q_depth",
        "t_amplitude",
        "qrs_width_ms",
        "qrs_area",
        "st_slope_mV_s",
    ]
    df = pd.DataFrame(
        np.random.rand(n, len(feature_cols)).astype(np.float32),
        columns=feature_cols,
    )
    df["record_id"] = records
    groups = np.array([int(r[1:]) for r in records])
    X = df[feature_cols].values.astype(np.float32)

    scaler, train_idx, val_idx = fit_feature_scaler_on_train(X, groups, n_splits=5)
    X_scaled = scale_features(X, scaler)

    assert X_scaled.shape == X.shape
    assert not np.any(np.isnan(X_scaled))
    assert not np.any(np.isinf(X_scaled))
    assert np.allclose(X_scaled[train_idx].mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(X_scaled[train_idx].std(axis=0), 1.0, atol=1e-6)
