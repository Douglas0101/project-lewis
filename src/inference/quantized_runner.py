"""Executor de modelos TensorFlow Lite INT8 full-integer.

Realiza quantização da entrada, invocação do interpretador e dequantização
da saída, retornando logits ou probabilidades em float32.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

LOGGER = logging.getLogger("lewis.inference.quantized_runner")


class QuantizedModelRunner:
    """Carrega e executa um modelo TFLite INT8.

    Parameters
    ----------
    model_path : Path | str
        Caminho para o arquivo ``.tflite`` quantizado.

    Attributes
    ----------
    input_scale : float
        Escala de quantização do tensor de entrada.
    input_zero_point : int
        Zero-point do tensor de entrada.
    output_scale : float
        Escala de quantização do tensor de saída.
    output_zero_point : int
        Zero-point do tensor de saída.
    """

    def __init__(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo TFLite não encontrado: {self.model_path}")

        self._interpreter: tf.lite.Interpreter | None = None
        self._input_details: list[dict[str, Any]] = []
        self._output_details: list[dict[str, Any]] = []
        self.input_scale = 1.0
        self.input_zero_point = 0
        self.output_scale = 1.0
        self.output_zero_point = 0

    def allocate(self) -> QuantizedModelRunner:
        """Aloca tensores e extrai parâmetros de quantização.

        Returns
        -------
        QuantizedModelRunner
            A própria instância, permitindo encadeamento.
        """
        LOGGER.debug("Alocando interpretador TFLite: %s", self.model_path)
        interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        interpreter.allocate_tensors()

        self._interpreter = interpreter
        self._input_details = interpreter.get_input_details()
        self._output_details = interpreter.get_output_details()

        input_params = self._input_details[0]["quantization_parameters"]
        output_params = self._output_details[0]["quantization_parameters"]

        self.input_scale = float(input_params["scales"][0])
        self.input_zero_point = int(input_params["zero_points"][0])
        self.output_scale = float(output_params["scales"][0])
        self.output_zero_point = int(output_params["zero_points"][0])

        return self

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Retorna o shape esperado pelo tensor de entrada."""
        if not self._input_details:
            self.allocate()
        return tuple(self._input_details[0]["shape"])

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        """Converte entrada float32 para int8 usando scale/zero_point."""
        x = np.asarray(x, dtype=np.float32)
        quantized = np.round(x / self.input_scale) + self.input_zero_point
        return np.clip(quantized, -128, 127).astype(np.int8)

    def _dequantize(self, q: np.ndarray) -> np.ndarray:
        """Converte saída int8 para float32 usando scale/zero_point."""
        return (q.astype(np.float32) - self.output_zero_point) * self.output_scale

    def _ensure_input_shape(self, x: np.ndarray) -> np.ndarray:
        """Garante que a entrada possui o shape exigido pelo modelo."""
        expected = self.input_shape
        if x.shape == expected:
            return x

        if x.ndim == len(expected) - 1 and x.shape == expected[1:]:
            return np.expand_dims(x, axis=0)

        raise ValueError(f"Shape de entrada {x.shape} incompatível com o esperado {expected}")

    def run(self, x: np.ndarray) -> np.ndarray:
        """Executa uma inferência e retorna logits/probabilidades float32.

        Parameters
        ----------
        x : np.ndarray
            Entrada float32 com shape compatível com ``input_shape``.
            Uma amostra sem dimensão de batch também é aceita.

        Returns
        -------
        np.ndarray
            Saída dequantizada com shape ``(batch, num_classes)``.
        """
        if self._interpreter is None:
            self.allocate()

        interpreter = self._interpreter
        assert interpreter is not None

        input_details = self._input_details[0]
        output_details = self._output_details[0]

        x = self._ensure_input_shape(x)
        x_int8 = self._quantize(x)

        interpreter.set_tensor(input_details["index"], x_int8)
        interpreter.invoke()
        output_int8 = interpreter.get_tensor(output_details["index"])

        return self._dequantize(output_int8)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Alias para ``run``."""
        return self.run(x)
