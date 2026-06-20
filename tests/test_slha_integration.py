import numpy as np
import tensorflow as tf

from src.models.slha import auto_configure_training


def test_auto_configure_returns_config():
    X = np.random.randn(8, 500, 1).astype("float32")
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype="int32")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(500, 1)),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    config = auto_configure_training(X, y, model, reference_batch_size=32)
    assert config.batch_size >= 1
    assert config.accelerator in {"cpu", "gpu"}
    assert config.precision in {"float32", "mixed_float16"}
