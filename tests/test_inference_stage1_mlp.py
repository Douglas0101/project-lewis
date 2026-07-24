"""Testes para MLPStage1Runner."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import keras
import numpy as np
import pytest

from src.inference.stage1_mlp_runner import MLPStage1Runner


@pytest.fixture
def dummy_mlp_artifacts(tmp_path: Path):
    """Cria modelo, scaler e config dummy para testes."""
    feature_names = ["feat_a", "feat_b"]
    model = keras.Sequential(
        [
            keras.layers.Dense(4, activation="relu", input_shape=(2,)),
            keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model_path = tmp_path / "model.keras"
    model.save(str(model_path), save_format="keras")

    scaler_path = tmp_path / "scaler.pkl"
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(np.random.randn(10, 2))
    joblib.dump(scaler, scaler_path)

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"feature_names": feature_names}))

    return model_path, scaler_path, config_path, feature_names


def test_predict_returns_expected_shapes(dummy_mlp_artifacts):
    """predict deve retornar arrays com shapes corretos."""
    model_path, scaler_path, config_path, feature_names = dummy_mlp_artifacts
    runner = MLPStage1Runner(model_path, scaler_path, config_path)

    n = 7
    features = {name: np.random.randn(n) for name in feature_names}
    result = runner.predict(features)

    assert result["y_pred"].shape == (n,)
    assert result["y_proba"].shape == (n, 2)
    assert set(np.unique(result["y_pred"])).issubset({0, 1})


def test_predict_with_threshold(dummy_mlp_artifacts):
    """predict com threshold deve respeitar o threshold."""
    model_path, scaler_path, config_path, feature_names = dummy_mlp_artifacts
    runner = MLPStage1Runner(model_path, scaler_path, config_path)

    features = {name: np.zeros(5) for name in feature_names}
    result = runner.predict(features, threshold=0.0)
    assert np.all(result["y_pred"] == 1)

    result = runner.predict(features, threshold=1.0)
    assert np.all(result["y_pred"] == 0)
