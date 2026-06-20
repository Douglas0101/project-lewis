"""Testes do módulo de warmup do SLHA."""

import numpy as np
import tensorflow as tf

from src.models.slha.warmup import warmup_model


def _dummy_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(500, 1)),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def test_warmup_returns_memory_estimate():
    model = _dummy_model()
    X = np.random.randn(4, 500, 1).astype("float32")
    y = np.array([0, 1, 0, 1], dtype="int32")
    result = warmup_model(model, X, y, batch_size=2)
    assert result.batch_time_ms > 0
    assert result.samples_per_second > 0
    assert result.peak_ram_mb >= 0
