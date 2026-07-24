"""R04: prove Dropout is in inference mode and does not affect predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.models.keras_loader import load_keras_model

pytestmark = pytest.mark.requires_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "stage1_float32_v2.0.keras"
FIXTURE_PATH = (
    PROJECT_ROOT / "artifacts" / "stage1_recall_investigation" / "R03" / "loader_fixture.npz"
)


def _load_inventory() -> dict[str, Any]:
    output = (
        PROJECT_ROOT
        / "artifacts"
        / "stage1_recall_investigation"
        / "R04"
        / "dropout_inventory.json"
    )
    if not output.exists():
        pytest.skip("R04 dropout inventory not generated yet")
    try:
        return json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid R04 dropout inventory: {output}") from error


def test_dropout_exists_and_has_rate_between_zero_and_one() -> None:
    """Model must contain exactly one Dropout layer with a valid rate."""
    inventory = _load_inventory()
    assert inventory["class_name"] == "Dropout"
    assert 0.0 < inventory["rate"] < 1.0


def test_dropout_has_seed_generator() -> None:
    """Keras 3 Dropout uses a SeedGenerator for RNG reproducibility."""
    inventory = _load_inventory()
    assert bool(inventory["has_seed_generator"])
    assert inventory["seed_generator_type"] == "keras.src.random.seed_generator.SeedGenerator"


def test_dropout_layer_is_trainable_but_non_trainable_count_is_one() -> None:
    """Dropout layer is trainable but contributes only one RNG state variable."""
    inventory = _load_inventory()
    assert bool(inventory["trainable"])
    assert inventory["non_trainable_variable_count"] == 1


def test_training_false_does_not_activate_dropout() -> None:
    """Three training=False calls are identical, proving Dropout is inactive."""
    import numpy as np

    model = load_keras_model(MODEL_PATH, compile=False)
    with np.load(FIXTURE_PATH, allow_pickle=False) as data:
        x_values = data["X_scaled"].astype(np.float32, copy=False)

    outputs = [np.asarray(model(x_values, training=False)) for _ in range(3)]
    for i in range(len(outputs) - 1):
        left, right = outputs[i], outputs[i + 1]
        assert np.array_equal(left, right)
        assert np.sum(np.argmax(left, axis=1) != np.argmax(right, axis=1)) == 0


def test_predict_does_not_activate_dropout() -> None:
    """Three predict() calls are identical, proving Dropout is inactive."""
    import numpy as np

    model = load_keras_model(MODEL_PATH, compile=False)
    with np.load(FIXTURE_PATH, allow_pickle=False) as data:
        x_values = data["X_scaled"].astype(np.float32, copy=False)

    outputs = [
        np.asarray(model.predict(x_values, verbose=0))  # type: ignore[reportArgumentType]
        for _ in range(3)
    ]
    for i in range(len(outputs) - 1):
        left, right = outputs[i], outputs[i + 1]
        assert np.array_equal(left, right)


def test_training_true_changes_output() -> None:
    """Dropout must be active when training=True; outputs may diverge."""
    import numpy as np

    model = load_keras_model(MODEL_PATH, compile=False)
    with np.load(FIXTURE_PATH, allow_pickle=False) as data:
        x_values = data["X_scaled"].astype(np.float32, copy=False)

    inference = np.asarray(model.predict(x_values, verbose=0))  # type: ignore[reportArgumentType]
    training = np.asarray(model(x_values, training=True))
    # Dropout is active, so outputs must differ from inference on at least one sample.
    assert not np.array_equal(inference, training)
