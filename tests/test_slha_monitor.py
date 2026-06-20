import json
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.models.slha.monitor import ResourceMonitor


def _tiny_dataset(n=8):
    X = np.random.randn(n, 10, 1).astype("float32")
    y = np.zeros(n, dtype="int32")
    return X, y


def _tiny_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(10, 1)),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def test_monitor_writes_resource_logs():
    X, y = _tiny_dataset()
    model = _tiny_model()
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "resources.jsonl"
        monitor = ResourceMonitor(log_path=log_path)
        model.fit(X, y, epochs=2, batch_size=4, callbacks=[monitor], verbose=0)
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            entry = json.loads(line)
            assert "epoch" in entry
            assert "cpu_percent" in entry
            assert "ram_used_gb" in entry
