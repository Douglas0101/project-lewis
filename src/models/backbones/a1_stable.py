"""A1_stable — residual backbone for the Chapman pretrain (FASE 6).

Stem conv + 3 residual blocks + GAP + dropout + sigmoid head. Residual paths
improve gradient flow vs the frozen A0. **BatchNorm is deliberately absent**:
the project TFLM constraints (see ``src/models/backbone_1d.py`` docstring)
forbid BatchNorm/LSTM/SeparableConv/attention, so stability comes from the
residual connections — every op is TFLite/INT8-friendly (Conv1D, Add, ReLU,
MaxPool1D, GlobalAveragePooling1D, Dense, Dropout).

Budget: ~32k params (<= 2× A0), est. FlatBuffer ≈ 41 KB (<= 64 KB).
"""

from __future__ import annotations

import logging

import tensorflow as tf

from src.models.backbone_1d import TFLMConstraints

from .spec import BackboneSpec

LOGGER = logging.getLogger("lewis.camada04.backbones.a1")


def _residual_block(x: tf.Tensor, filters: int, kernel_size: int, name: str) -> tf.Tensor:
    """Conv-relu → Conv → Add(skip) → relu (1x1 projection when channels differ)."""
    y = tf.keras.layers.Conv1D(
        filters, kernel_size, padding="same", activation="relu", name=f"{name}_conv1"
    )(x)
    y = tf.keras.layers.Conv1D(filters, kernel_size, padding="same", name=f"{name}_conv2")(y)
    if x.shape[-1] != filters:
        x = tf.keras.layers.Conv1D(filters, 1, padding="same", name=f"{name}_proj")(x)
    out = tf.keras.layers.Add(name=f"{name}_add")([x, y])
    return tf.keras.layers.Activation("relu", name=f"{name}_relu")(out)


def build_a1_stable(spec: BackboneSpec) -> tf.keras.Model:
    """Build A1_stable."""
    inputs = tf.keras.Input(shape=(spec.input_len, 1), name="input")
    x = tf.keras.layers.Conv1D(
        16, 7, padding="same", activation="relu", name="stem_conv"
    )(inputs)
    x = tf.keras.layers.MaxPooling1D(2, name="stem_pool")(x)  # 250

    x = _residual_block(x, 16, 5, "res1")
    x = tf.keras.layers.MaxPooling1D(2, name="pool1")(x)  # 125
    x = _residual_block(x, 32, 5, "res2")
    x = tf.keras.layers.MaxPooling1D(2, name="pool2")(x)  # 62
    x = _residual_block(x, 64, 3, "res3")

    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dropout(spec.dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(spec.num_classes, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="a1_stable")
    info = TFLMConstraints.validate_model(model)
    LOGGER.info(
        "Backbone a1_stable | params=%d | est_flatbuffer=%dKB",
        info["total_params"],
        info["flatbuffer_kb_est"],
    )
    return model
