"""Pipeline de Quantization-Aware Training (QAT) INT8 para TFLM.

Aplica QAT leve sobre um modelo Keras FP32 e converte para TensorFlow Lite
full-integer (INT8), adequado para execução em microcontroladores via
TensorFlow Lite Micro.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import (
    default_8bit_quantize_registry as _registry,
)

_QuantizeInfo = _registry._QuantizeInfo
_no_quantize = _registry._no_quantize


LOGGER = logging.getLogger("lewis.camada05.qat")


def _register_qat_layers() -> None:
    """Registra camadas 1D usadas pelo Project-Lewis no registry QAT padrão.

    ``tfmot`` 0.8.x não inclui ``Conv1D`` nem ``MaxPooling1D`` no
    ``Default8BitQuantizeRegistry``. Sem esse registro, ``quantize_model``
    levanta ``RuntimeError`` para modelos com convoluções 1D.
    """
    registry_cls = _registry.Default8BitQuantizeRegistry
    info_list = registry_cls._LAYER_QUANTIZE_INFO
    present = {info.layer_type for info in info_list}

    if tf_keras.layers.Conv1D not in present:
        info_list.append(_QuantizeInfo(tf_keras.layers.Conv1D, ["kernel"], ["activation"]))
    if tf_keras.layers.MaxPooling1D not in present:
        info_list.append(_no_quantize(tf_keras.layers.MaxPooling1D))


_register_qat_layers()


def build_model(stage: int = 1, config: dict | None = None) -> tf.keras.Model:
    """Constrói um modelo pequeno compatível com QAT INT8.

    ``tensorflow_model_optimization`` opera sobre o Keras "legacy"
    (``tf_keras``); por isso este builder usa explicitamente ``tf_keras``
    para garantir que ``quantize_model`` reconheça as camadas.

    Parameters
    ----------
    stage : int
        Estágio do pipeline: ``1`` para binário (N vs Anormal) ou ``2`` para
        multiclasse (S vs V vs F).
    config : dict | None
        Hiperparâmetros opcionais (``input_len``, ``num_classes``,
        ``conv_filters``, ``dense_units``, ``dropout_rate``).

    Returns
    -------
    tf.keras.Model
        Modelo Keras não compilado.
    """
    config = config or {}
    input_len = int(config.get("input_len", 500))
    num_classes = int(config.get("num_classes", 2 if stage == 1 else 3))
    conv_filters = tuple(config.get("conv_filters", (8, 16, 16)))
    conv_kernels = tuple(config.get("conv_kernels", (7, 5, 3)))
    dense_units = int(config.get("dense_units", 16))
    dropout_rate = float(config.get("dropout_rate", 0.3))
    name = config.get("name", f"lewis_qat_stage{stage}")

    inputs = tf_keras.Input(shape=(input_len, 1), name="input")
    x = inputs
    for idx, (filters, kernel_size) in enumerate(zip(conv_filters, conv_kernels), start=1):
        x = tf_keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            name=f"conv1d_{idx}",
        )(x)
        x = tf_keras.layers.MaxPooling1D(pool_size=2, name=f"maxpool_{idx}")(x)

    x = tf_keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf_keras.layers.Dense(dense_units, activation="relu", name="embedding")(x)
    x = tf_keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf_keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf_keras.Model(inputs=inputs, outputs=outputs, name=name)
    LOGGER.info(
        "QAT model %s | params=%d | stage=%d",
        name,
        model.count_params(),
        stage,
    )
    return model


def quantize_model(
    model: tf.keras.Model,
    representative_data: Callable,
    output_path: Path,
    fallback: str = "mixed_precision",
) -> Path:
    """Aplica QAT e converte para TFLite INT8.

    Parameters
    ----------
    model : keras.Model
        Modelo FP32 treinado.
    representative_data : callable
        Generator que yielda listas de tensores float32.
    output_path : Path
        Caminho para salvar o .tflite.
    fallback : str
        Estratégia de fallback se ΔF1 for alto (reservado para uso futuro).
    """
    q_aware_model = tfmot.quantization.keras.quantize_model(model)
    q_aware_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # fine-tuning leve é feito externamente; esta função foca na conversão
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_data
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    LOGGER.info(
        "Modelo QAT INT8 salvo em %s (%.2f KB, fallback=%s)",
        output_path,
        len(tflite_model) / 1024,
        fallback,
    )
    return output_path
