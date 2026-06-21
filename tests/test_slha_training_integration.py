"""Integration tests for SLHA opt-in in pretrain_chapman and finetune_mitbih."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.models.finetune_mitbih import finetune_mitbih
from src.models.pretrain_chapman import pretrain_chapman


class _NoOpCallback(tf.keras.callbacks.Callback):
    """Stand-in callback that ignores any constructor arguments."""

    def __init__(self, *args, **kwargs):
        super().__init__()


class _NoOpAUC(tf.keras.metrics.Metric):
    """Stand-in AUC metric that always reports zero."""

    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", "auc")
        super().__init__(name=name)

    def update_state(self, *args, **kwargs):
        pass

    def result(self):
        return tf.constant(0.0, dtype=tf.float32)

    def reset_state(self):
        pass

    def reset_states(self):
        pass


def _dummy_generator(batch_size: int = 4, steps: int = 2):
    """Yield ``steps`` batches of random multi-label data."""

    def generator():
        for _ in range(steps):
            X = np.random.randn(batch_size, 500, 1).astype(np.float32)
            y = np.zeros((batch_size, 5), dtype=np.float32)
            yield X, y

    return generator


@pytest.mark.qg4
def test_pretrain_chapman_accepts_use_slha(tmp_path, monkeypatch):
    monkeypatch.setattr(tf.keras.callbacks, "TensorBoard", _NoOpCallback)
    monkeypatch.setattr(tf.keras.metrics, "AUC", _NoOpAUC)

    model, history = pretrain_chapman(
        data_generator=_dummy_generator(batch_size=4, steps=2),
        val_generator=_dummy_generator(batch_size=4, steps=2),
        steps_per_epoch=2,
        validation_steps=2,
        epochs=1,
        batch_size=4,
        experiment_dir=tmp_path / "exp_pretrain",
        use_slha=True,
    )
    assert isinstance(model, tf.keras.Model)
    assert "loss" in history


@pytest.mark.qg5
def test_finetune_mitbih_accepts_use_slha(tmp_path, monkeypatch):
    monkeypatch.setattr(tf.keras.callbacks, "TensorBoard", _NoOpCallback)
    monkeypatch.setattr(tf.keras.metrics, "AUC", _NoOpAUC)

    inputs = tf.keras.Input(shape=(500, 1))
    x = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    X = np.random.randn(8, 500, 1).astype(np.float32)
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int32)

    model, history = finetune_mitbih(
        model,
        X_train=X,
        y_train=y,
        X_val=X,
        y_val=y,
        epochs=1,
        batch_size=4,
        experiment_dir=tmp_path / "exp_finetune",
        use_slha=True,
    )
    assert isinstance(model, tf.keras.Model)
    assert "loss" in history
