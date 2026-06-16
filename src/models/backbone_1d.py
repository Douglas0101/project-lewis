"""Backbone 1D-CNN leve para ECG arrhythmia classification (TFLM/STM32F4).

Arquitetura (~13K params, <25KB FlatBuffer INT8):
    Input(500, 1)
    → Conv1D(16, 7, relu, padding="same") → MaxPool1D(2)   # 250
    → Conv1D(32, 5, relu, padding="same") → MaxPool1D(2)   # 125
    → Conv1D(64, 3, relu, padding="same") → MaxPool1D(2)   # 62
    → GlobalAveragePooling1D()                              # 64
    → Dense(64, relu) → Dropout(0.3)
    → Dense(num_classes, activation="softmax")

Restrições TFLM:
- Sem LSTM/GRU/RNN, BatchNorm, SeparableConv1D, attention, GroupNorm/LayerNorm
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import tensorflow as tf

LOGGER = logging.getLogger("lewis.camada04.backbone")


class TFLMConstraints:
    """Validações de compatibilidade com TensorFlow Lite Micro."""

    MAX_PARAMS = 20_000
    MAX_FLATBUFFER_KB = 64
    MAX_ARENA_KB = 64

    @classmethod
    def validate_model(cls, model: tf.keras.Model) -> dict:
        """Checa se modelo respeita limites TFLM.

        Returns
        -------
        dict with keys: total_params, flatbuffer_kb (estimated), passes.
        """
        total_params = model.count_params()
        # Estimativa: float32 ~4 bytes/param; INT8 ~1 byte/param + overhead ~30%
        flatbuffer_kb_est = int(total_params * 1.3 / 1024)
        passes = (
            total_params <= cls.MAX_PARAMS and flatbuffer_kb_est <= cls.MAX_FLATBUFFER_KB * 1024
        )
        return {
            "total_params": total_params,
            "flatbuffer_kb_est": flatbuffer_kb_est,
            "passes": passes,
        }


def build_backbone_1d(
    input_len: int = 500,
    num_classes: int = 5,
    embedding_dim: int = 64,
    dropout_rate: float = 0.3,
    name: str = "lewis_backbone",
) -> tf.keras.Model:
    """Constrói backbone 1D-CNN enxuto.

    Parameters
    ----------
    input_len : int
        Comprimento do segmento (amostras @ 500Hz). Default 500 (1000ms).
    num_classes : int
        Número de classes de saída. Default 5 (AAMI: N, S, V, F, Q).
    embedding_dim : int
        Dimensão do embedding antes do classificador. Default 64.
    dropout_rate : float
        Taxa de dropout. Default 0.3.
    name : str
        Nome do modelo.

    Returns
    -------
    tf.keras.Model
        Modelo compilável (ainda não compilado).
    """
    inputs = tf.keras.Input(shape=(input_len, 1), name="input")

    # Block 1: 500 → 250
    x = tf.keras.layers.Conv1D(
        filters=16,
        kernel_size=7,
        padding="same",
        activation="relu",
        name="conv1d_1",
    )(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="maxpool_1")(x)

    # Block 2: 250 → 125
    x = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="conv1d_2",
    )(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="maxpool_2")(x)

    # Block 3: 125 → 62
    x = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1d_3",
    )(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="maxpool_3")(x)

    # Global Average Pooling: 62 → 64 features
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)

    # Embedding
    x = tf.keras.layers.Dense(embedding_dim, activation="relu", name="embedding")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)

    # Classifier — softmax para classificação single-label (fine-tuning)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    info = TFLMConstraints.validate_model(model)
    LOGGER.info(
        "Backbone %s | params=%d | est_flatbuffer=%dKB | passes=%s",
        name,
        info["total_params"],
        info["flatbuffer_kb_est"],
        info["passes"],
    )
    return model


def build_backbone_1d_multilabel(
    input_len: int = 500,
    num_classes: int = 5,
    embedding_dim: int = 64,
    dropout_rate: float = 0.3,
    name: str = "lewis_backbone_pretrain",
) -> tf.keras.Model:
    """Backbone com saída sigmoid para pré-treino multi-label (Chapman).

    Parameters
    ----------
    input_len : int
        Comprimento do segmento. Default 500.
    num_classes : int
        Número de superclasses SCP-ECG. Default 5 (NORM, CD, MI, HYP, STTC).
    embedding_dim : int
        Dimensão do embedding. Default 64.
    dropout_rate : float
        Taxa de dropout. Default 0.3.
    name : str
        Nome do modelo.

    Returns
    -------
    tf.keras.Model
        Modelo com saída sigmoid (multi-label).
    """
    model = build_backbone_1d(
        input_len=input_len,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        name=name,
    )
    # Substituir softmax por sigmoid na última camada
    model.layers[-1].activation = tf.keras.activations.sigmoid
    return model


def freeze_conv_layers(model: tf.keras.Model) -> tf.keras.Model:
    """Congela todas as camadas convolucionais + GAP para transfer learning.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo Keras.

    Returns
    -------
    tf.keras.Model
        Modelo com camadas convolucionais congeladas.
    """
    frozen_types = ("Conv1D", "MaxPooling1D", "GlobalAveragePooling1D")
    for layer in model.layers:
        if layer.__class__.__name__ in frozen_types:
            layer.trainable = False
            LOGGER.debug("Frozen layer: %s", layer.name)
    return model


def unfreeze_all(model: tf.keras.Model) -> tf.keras.Model:
    """Descongela todas as camadas."""
    for layer in model.layers:
        layer.trainable = True
    return model


def save_model_config(
    model: tf.keras.Model,
    output_path: Path,
    extra: Optional[dict] = None,
) -> None:
    """Salva config JSON com arquitetura e parâmetros."""
    import json

    config = {
        "name": model.name,
        "input_shape": [None if s is None else int(s) for s in model.input_shape],
        "output_shape": [None if s is None else int(s) for s in model.output_shape],
        "total_params": int(model.count_params()),
        "trainable_params": int(
            sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        ),
        "non_trainable_params": int(
            sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
        ),
        "layers": [
            {
                "name": layer.name,
                "class": layer.__class__.__name__,
                "trainable": bool(layer.trainable),
            }
            for layer in model.layers
        ],
    }
    if extra:
        config.update(extra)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
    LOGGER.info("Config salva em %s", output_path)
