"""R04: prove trainable weights are immutable during any forward pass."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest

from src.models.keras_loader import load_keras_model

pytestmark = pytest.mark.requires_artifacts

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "stage1_float32_v2.0.keras"
FIXTURE_PATH = (
    PROJECT_ROOT / "artifacts" / "stage1_recall_investigation" / "R03" / "loader_fixture.npz"
)


def _hash_trainable(model: Any) -> str:
    digest = hashlib.sha256()
    for variable in model.trainable_variables:
        digest.update(variable.numpy().tobytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def fixture_x() -> np.ndarray:
    with np.load(FIXTURE_PATH, allow_pickle=False) as data:
        return data["X_scaled"].astype(np.float32, copy=False)


def test_predict_does_not_change_trainable_weights(fixture_x: np.ndarray) -> None:
    model = load_keras_model(MODEL_PATH, compile=False)
    before = _hash_trainable(model)
    _ = np.asarray(model.predict(fixture_x, verbose=0))  # type: ignore[reportArgumentType]
    after = _hash_trainable(model)
    assert before == after


def test_training_false_does_not_change_trainable_weights(fixture_x: np.ndarray) -> None:
    model = load_keras_model(MODEL_PATH, compile=False)
    before = _hash_trainable(model)
    _ = np.asarray(model(fixture_x, training=False))
    after = _hash_trainable(model)
    assert before == after


def test_training_true_does_not_change_trainable_weights(fixture_x: np.ndarray) -> None:
    """Dropout is active, but weights must remain unchanged."""
    model = load_keras_model(MODEL_PATH, compile=False)
    before = _hash_trainable(model)
    _ = np.asarray(model(fixture_x, training=True))
    after = _hash_trainable(model)
    assert before == after
