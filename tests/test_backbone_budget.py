"""Backbone budget + TFLite-export smoke tests (FASE 6).

Pins the frozen A0 (params == 19.933) and enforces the embedded budget on
variants: params <= 2×A0, estimated FlatBuffer <= 64 KB, TFLite-convertible.
Also covers the A2 training-layer loss variants (pos_weight / focal).
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.models.backbones import (
    A0_BASELINE_PARAMS,
    MAX_FLATBUFFER_KB,
    MAX_PARAMS,
    BackboneSpec,
    build_backbone,
)
from src.models.pretrain_losses import build_loss, estimate_pos_weights


def _spec(arch: str) -> BackboneSpec:
    return BackboneSpec(arch=arch, input_len=500, num_classes=5, dropout_rate=0.3)


def test_a0_params_pinned():
    model = build_backbone(_spec("a0"))
    assert model.count_params() == A0_BASELINE_PARAMS


@pytest.mark.parametrize("arch", ["a0", "a1", "a2"])
def test_variant_budget_and_io(arch):
    model = build_backbone(_spec(arch))
    assert model.count_params() <= MAX_PARAMS
    est_kb = model.count_params() * 1.3 / 1024
    assert est_kb <= MAX_FLATBUFFER_KB
    assert model.input_shape == (None, 500, 1)
    assert model.output_shape == (None, 5)
    out = model(tf.zeros((2, 500, 1)), training=False)
    assert out.shape == (2, 5)
    assert float(tf.reduce_max(out)) <= 1.0 and float(tf.reduce_min(out)) >= 0.0


@pytest.mark.parametrize("arch", ["a0", "a1"])
def test_tflite_export_smoke(arch, tmp_path):
    """Variant must convert to TFLite (float32) without custom ops."""
    model = build_backbone(_spec(arch))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_bytes = converter.convert()
    assert len(tflite_bytes) <= MAX_FLATBUFFER_KB * 1024 * 4  # float32 ≈ 4B/param
    out = tmp_path / f"{arch}.tflite"
    out.write_bytes(tflite_bytes)
    assert out.stat().st_size == len(tflite_bytes)


def test_factory_rejects_unknown_arch():
    with pytest.raises(ValueError, match="unknown architecture"):
        build_backbone(_spec("zzz"))


# ---------------------------------------------------------------------------
# A2 — loss variants (training layer)
# ---------------------------------------------------------------------------


def test_bce_weighted_penalizes_positive_class_more():
    y_true = tf.constant([[1.0, 0.0]])
    y_pred = tf.constant([[0.1, 0.1]])  # erra mais na classe 0 (positiva)
    pos_weight = np.array([5.0, 1.0], dtype=np.float32)
    loss_fn = build_loss("bce_weighted", pos_weight=pos_weight)
    value = float(loss_fn(y_true, y_pred))
    plain = float(tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred)))
    assert value > plain, "pos_weight deve elevar a loss quando erra a minoritária"


def test_focal_loss_builds_and_is_finite():
    loss_fn = build_loss("focal")
    y_true = tf.constant([[1.0, 0.0, 1.0]])
    y_pred = tf.constant([[0.9, 0.2, 0.4]])
    value = float(loss_fn(y_true, y_pred))
    assert np.isfinite(value) and value >= 0.0


def test_build_loss_default_is_bce():
    assert build_loss("bce") == "binary_crossentropy"
    with pytest.raises(ValueError):
        build_loss("nope")


def test_estimate_pos_weights_uses_train_records_only():
    records = {"r1", "r2", "r3"}
    diagnosis = {
        "r1": [1, 0, 0, 0, 0],
        "r2": [1, 1, 0, 0, 0],
        "r3": [1, 0, 0, 0, 0],
    }
    weights = estimate_pos_weights(records, diagnosis)
    # classe 0: 3 pos / 0 neg -> piso 1.0; classe 1: 1 pos / 2 neg -> 2.0;
    # classes 2-4: 0 pos -> teto proporcional (3/1 = 3.0)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(2.0)
    assert weights[2] == pytest.approx(3.0)
    assert weights[2] >= weights[1] >= weights[0]
