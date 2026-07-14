import numpy as np
import pandas as pd

from src.features.scaler_utils import fit_feature_scaler_on_train, scale_features


def _make_synthetic_feature_data(
    n: int = 50,
    feature_cols: list[str] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create synthetic feature DataFrame, feature matrix, and groups."""
    rng = np.random.default_rng(seed)
    if feature_cols is None:
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
    records = [f"r{i % 10}" for i in range(n)]
    df = pd.DataFrame(rng.random((n, len(feature_cols))).astype(np.float32), columns=feature_cols)
    df["record_id"] = records
    groups = np.array([int(r[1:]) for r in records])
    x = df[feature_cols].values.astype(np.float32)
    return df, x, groups


def test_prepare_stage1_features_scaler_logic():
    _, x, groups = _make_synthetic_feature_data()

    scaler, train_idx, _ = fit_feature_scaler_on_train(x, groups, n_splits=5)
    x_scaled = scale_features(x, scaler)

    assert x_scaled.shape == x.shape
    assert not np.any(np.isnan(x_scaled))
    assert not np.any(np.isinf(x_scaled))
    assert np.allclose(x_scaled[train_idx].mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(x_scaled[train_idx].std(axis=0), 1.0, atol=1e-6)


def test_scaler_roundtrip(tmp_path):
    import joblib

    _, x, groups = _make_synthetic_feature_data()

    scaler, _, _ = fit_feature_scaler_on_train(x, groups, n_splits=5)
    x_scaled = scale_features(x, scaler)

    scaler_path = tmp_path / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    loaded = joblib.load(scaler_path)
    np.testing.assert_array_almost_equal(loaded.transform(x), x_scaled, decimal=6)


def test_median_imputation_uses_train_only():
    feature_cols = ["rr_prev", "qrs_width_ms"]
    _, x, groups = _make_synthetic_feature_data(feature_cols=feature_cols)
    x[0, 0] = np.nan  # introduce NaN in a validation sample

    scaler, train_idx, _ = fit_feature_scaler_on_train(x, groups, n_splits=5)

    # Impute using train median only
    from pandas import DataFrame

    train_median = DataFrame(x[train_idx], columns=feature_cols).median()
    x_imputed = x.copy()
    for j, col in enumerate(feature_cols):
        mask = np.isnan(x_imputed[:, j])
        x_imputed[mask, j] = train_median[col]

    x_scaled = scale_features(x_imputed, scaler)
    assert not np.any(np.isnan(x_scaled))
