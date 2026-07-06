"""Testes para o executor de modelos TFLite INT8.

Valida:
* Carregamento e alocação de modelo .tflite
* Aplicação de input_scale e input_zero_point
* Retorno de logits/probabilidades float32
* Suporte a classificador binário e multiclasse
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.inference.quantized_runner import QuantizedModelRunner
from src.models.backbone_1d import build_backbone_1d
from src.quantization.export_tflite import export_tflite
from src.quantization.ptq import representative_dataset_random


def _make_quantized_model(tmp_path: Path, num_classes: int) -> Path:
    """Cria um modelo TFLite INT8 dummy para os testes."""
    model = build_backbone_1d(input_len=500, num_classes=num_classes)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
    )

    rng = np.random.default_rng(42)
    n_samples = 30
    X = rng.standard_normal((n_samples, 500, 1)).astype(np.float32)
    y = np.tile(np.arange(num_classes, dtype=int), n_samples // num_classes + 1)[:n_samples]
    model.fit(X, y, epochs=1, batch_size=8, verbose=0)

    rep_data = representative_dataset_random(X, n_samples=10, seed=42)
    return export_tflite(
        model=model,
        representative_data=rep_data,
        output_dir=tmp_path,
        model_name="test_model",
        version="2.0.0",
        allow_float=False,
    )


class TestQuantizedModelRunner:
    """Testes unitários do executor TFLite INT8."""

    def test_load_missing_model_raises(self):
        """Deve levantar FileNotFoundError quando o modelo não existe."""
        with pytest.raises(FileNotFoundError):
            QuantizedModelRunner("nao_existe.tflite")

    def test_allocate_extracts_quantization_params(self, tmp_path):
        """Deve extrair scales e zero_points corretamente."""
        tflite_path = _make_quantized_model(tmp_path, 2)
        runner = QuantizedModelRunner(tflite_path).allocate()

        assert runner.input_scale > 0.0
        assert runner.output_scale > 0.0
        assert -128 <= runner.input_zero_point <= 127
        assert runner.input_shape == (1, 500, 1)

    def test_binary_model_returns_float32_logits(self, tmp_path):
        """Classificador binário deve retornar array (1, 2) float32."""
        tflite_path = _make_quantized_model(tmp_path, 2)
        runner = QuantizedModelRunner(tflite_path).allocate()

        x = np.zeros((1, 500, 1), dtype=np.float32)
        out = runner.run(x)

        assert out.shape == (1, 2)
        assert out.dtype == np.float32
        assert np.isfinite(out).all()

    def test_multiclass_model_returns_float32_logits(self, tmp_path):
        """Classificador multiclasse deve retornar array (1, 3) float32."""
        tflite_path = _make_quantized_model(tmp_path, 3)
        runner = QuantizedModelRunner(tflite_path).allocate()

        x = np.zeros((1, 500, 1), dtype=np.float32)
        out = runner.run(x)

        assert out.shape == (1, 3)
        assert out.dtype == np.float32
        assert np.isfinite(out).all()

    def test_single_sample_without_batch_dimension(self, tmp_path):
        """Deve aceitar amostra com shape (500, 1) e expandir para (1, 500, 1)."""
        tflite_path = _make_quantized_model(tmp_path, 3)
        runner = QuantizedModelRunner(tflite_path).allocate()

        x = np.zeros((500, 1), dtype=np.float32)
        out = runner.run(x)

        assert out.shape == (1, 3)

    def test_runner_predict_alias(self, tmp_path):
        """predict deve ser alias para run."""
        tflite_path = _make_quantized_model(tmp_path, 2)
        runner = QuantizedModelRunner(tflite_path).allocate()
        x = np.zeros((1, 500, 1), dtype=np.float32)

        np.testing.assert_array_equal(runner.predict(x), runner.run(x))
