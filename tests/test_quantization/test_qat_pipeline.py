"""Testes do pipeline QAT INT8."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.quantization.qat_pipeline import build_model, quantize_model


@pytest.fixture
def representative_dataset():
    """Generator representativo com 10 amostras de zeros."""

    def _generator():
        for _ in range(10):
            yield [np.zeros((1, 500, 1), dtype=np.float32)]

    return _generator


def test_qat_outputs_int8_model(tmp_path, representative_dataset):
    """Deve converter um modelo pequeno e salvar o arquivo .tflite."""
    model = build_model(stage=1, config={"conv_filters": (8, 16, 16), "dense_units": 16})
    output_path = tmp_path / "model_qat_int8.tflite"
    result = quantize_model(model, representative_dataset, output_path)
    assert result.exists()
    assert result.stat().st_size > 0


def test_qat_model_is_int8(tmp_path, representative_dataset):
    """Deve produzir um modelo TFLite com entrada e saída int8."""
    model = build_model(stage=1, config={"conv_filters": (8, 16, 16), "dense_units": 16})
    output_path = tmp_path / "model_qat_int8.tflite"
    quantize_model(model, representative_dataset, output_path)

    interpreter = tf.lite.Interpreter(model_path=str(output_path))
    interpreter.allocate_tensors()

    input_dtype = interpreter.get_input_details()[0]["dtype"]
    output_dtype = interpreter.get_output_details()[0]["dtype"]

    assert input_dtype == np.int8
    assert output_dtype == np.int8
