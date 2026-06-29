"""Backbone 1D-CNN leve para ECG arrhythmia classification (TFLM/STM32F4).

Arquitetura (~20K params, <26KB FlatBuffer INT8):
    Input(500, 1)
    → Conv1D(16, 7, relu, padding="same") → MaxPool1D(2)   # 250
    → Conv1D(40, 5, relu, padding="same") → MaxPool1D(2)   # 125
    → Conv1D(80, 3, relu, padding="same") → MaxPool1D(2)   # 62
    → GlobalAveragePooling1D()                              # 80
    → Dense(80, relu) → Dropout(0.3)
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
    embedding_dim: int = 80,
    dropout_rate: float = 0.3,
    kernel_regularizer=None,
    name: str = "lewis_backbone",
    conv_filters: tuple[int, int, int] = (16, 40, 80),
    conv_kernels: tuple[int, int, int] = (7, 5, 3),
    dense_units: int = 80,
    channels: int = 1,
) -> tf.keras.Model:
    """Constrói backbone 1D-CNN enxuto.

    Parameters
    ----------
    input_len : int
        Comprimento do segmento (amostras @ 500Hz). Default 500 (1000ms).
    num_classes : int
        Número de classes de saída. Default 5 (AAMI: N, S, V, F, Q).
    embedding_dim : int
        Dimensão do embedding antes do classificador. Default 80.
    dropout_rate : float
        Taxa de dropout. Default 0.3.
    kernel_regularizer : tf.keras.regularizers.Regularizer, optional
        Regularizador L1/L2 para kernels de Conv1D/Dense.
    name : str
        Nome do modelo.
    conv_filters : tuple[int, int, int]
        Número de filtros dos 3 blocos Conv1D. Default (16, 40, 80).
    conv_kernels : tuple[int, int, int]
        Tamanhos de kernel dos 3 blocos Conv1D. Default (7, 5, 3).
    dense_units : int
        Unidades da camada densa intermediária (embedding). Default 80.
    channels : int
        Número de canais de entrada. Default 1 (sinal raw). Use >1 para
        incluir features como canais adicionais.

    Returns
    -------
    tf.keras.Model
        Modelo compilável (ainda não compilado).
    """
    inputs = tf.keras.Input(shape=(input_len, channels), name="input")

    x = inputs
    for idx, (filters, kernel_size) in enumerate(zip(conv_filters, conv_kernels), start=1):
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            kernel_regularizer=kernel_regularizer,
            name=f"conv1d_{idx}",
        )(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, name=f"maxpool_{idx}")(x)

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)

    # Embedding
    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=kernel_regularizer,
        name="embedding",
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)

    # Classifier — softmax para classificação single-label (fine-tuning)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=kernel_regularizer,
        name="output",
    )(x)

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


def build_backbone_1d_with_features(
    input_len: int = 500,
    num_classes: int = 5,
    num_features: int = 2,
    embedding_dim: int = 80,
    dropout_rate: float = 0.3,
    kernel_regularizer=None,
    name: str = "lewis_backbone_features",
    conv_filters: tuple[int, int, int] = (16, 40, 80),
    conv_kernels: tuple[int, int, int] = (7, 5, 3),
    dense_units: int = 80,
) -> tf.keras.Model:
    """Constrói backbone 1D-CNN com entrada auxiliar de features.

    As features (ex.: RR_prev, QRS_width) são concatenadas ao embedding
    obtido pelo GlobalAveragePooling1D, fornecendo informação morfológica e
    de contexto sem exigir que o CNN a inferencie implicitamente.

    Parameters
    ----------
    input_len : int
        Comprimento do segmento (amostras @ 500Hz).
    num_classes : int
        Número de classes de saída.
    num_features : int
        Número de features auxiliares (ex.: 2 para rr_prev + qrs_width_ms).
    embedding_dim : int
        Dimensão do embedding CNN (mantida para compatibilidade).
    dropout_rate : float
        Taxa de dropout.
    kernel_regularizer : tf.keras.regularizers.Regularizer, optional
        Regularizador para kernels.
    name : str
        Nome do modelo.
    conv_filters, conv_kernels, dense_units
        Hiperparâmetros da torre convolucional.

    Returns
    -------
    tf.keras.Model
        Modelo com duas entradas: ``input`` (sinal) e ``features``.
    """
    signal_input = tf.keras.Input(shape=(input_len, 1), name="input")
    feature_input = tf.keras.Input(shape=(num_features,), name="features")

    x = signal_input
    for idx, (filters, kernel_size) in enumerate(zip(conv_filters, conv_kernels), start=1):
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            kernel_regularizer=kernel_regularizer,
            name=f"conv1d_{idx}",
        )(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, name=f"maxpool_{idx}")(x)

    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=kernel_regularizer,
        name="embedding",
    )(x)

    # Combinar embedding CNN com features auxiliares
    x = tf.keras.layers.Concatenate(name="concat_features")([x, feature_input])
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=kernel_regularizer,
        name="output",
    )(x)

    model = tf.keras.Model(
        inputs=[signal_input, feature_input], outputs=outputs, name=name
    )

    info = TFLMConstraints.validate_model(model)
    LOGGER.info(
        "BackboneWithFeatures %s | params=%d | est_flatbuffer=%dKB | passes=%s",
        name,
        info["total_params"],
        info["flatbuffer_kb_est"],
        info["passes"],
    )
    return model


def build_backbone_1d_multilabel(
    input_len: int = 500,
    num_classes: int = 5,
    embedding_dim: int = 80,
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
        Dimensão do embedding. Default 80.
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


def build_stage2_backbone(
    input_len: int = 500,
    num_classes: int = 3,
    dropout_rate: float = 0.3,
    name: str = "lewis_stage2",
) -> tf.keras.Model:
    """Backbone leve para o Estágio 2 (S vs V vs F).

    Arquitetura (~4.200 params, < 7 KB FlatBuffer INT8):
        Input(500, 1)
        → Conv1D(16, 5, stride=2, relu) → MaxPool1D(2)   # 125
        → Conv1D(24, 3, stride=2, relu) → MaxPool1D(2)   # 31
        → Conv1D(32, 3, relu) → GlobalAveragePooling1D()  # 32
        → Dense(16, relu) → Dropout(0.3)
        → Dense(num_classes, activation="softmax")

    Parameters
    ----------
    input_len : int
        Comprimento do segmento. Default 500.
    num_classes : int
        Número de classes de saída. Default 3 (S, V, F).
    dropout_rate : float
        Taxa de dropout. Default 0.3.
    name : str
        Nome do modelo.

    Returns
    -------
    tf.keras.Model
        Modelo leve para Estágio 2.
    """
    inputs = tf.keras.Input(shape=(input_len, 1), name="input")

    x = tf.keras.layers.Conv1D(
        filters=16,
        kernel_size=5,
        strides=2,
        padding="same",
        activation="relu",
        name="conv1d_1",
    )(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="maxpool_1")(x)

    x = tf.keras.layers.Conv1D(
        filters=24,
        kernel_size=3,
        strides=2,
        padding="same",
        activation="relu",
        name="conv1d_2",
    )(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="maxpool_2")(x)

    x = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1d_3",
    )(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)

    x = tf.keras.layers.Dense(16, activation="relu", name="dense_embedding")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
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


def load_backbone_weights_from_pretrained(
    source_path: Path | str,
    target_model: tf.keras.Model,
) -> tf.keras.Model:
    """Carrega pesos compatíveis de um modelo pré-treinado no modelo alvo.

    Copia pesos por nome de camada, ignorando a camada de saída quando os
    shapes não coincidem (ex.: 5 classes -> 2 classes).

    Parameters
    ----------
    source_path : Path | str
        Caminho para o modelo Keras pré-treinado.
    target_model : tf.keras.Model
        Modelo destino (mesma arquitetura de backbone, saída pode diferir).

    Returns
    -------
    tf.keras.Model
        Modelo destino com pesos carregados.
    """
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Modelo pré-treinado não encontrado: {source_path}")

    source_model = tf.keras.models.load_model(str(source_path), compile=False)
    source_layers = {layer.name: layer for layer in source_model.layers}

    for layer in target_model.layers:
        if layer.name not in source_layers:
            LOGGER.warning("Camada %s não existe no modelo fonte", layer.name)
            continue
        source_layer = source_layers[layer.name]
        try:
            layer.set_weights(source_layer.get_weights())
            LOGGER.debug("Pesos copiados: %s", layer.name)
        except ValueError as exc:
            LOGGER.warning("Ignorando pesos de %s (shape incompatível): %s", layer.name, exc)

    LOGGER.info("Pesos pré-treinados carregados de %s", source_path)
    return target_model


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
