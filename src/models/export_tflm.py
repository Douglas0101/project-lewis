"""Exportação para TensorFlow Lite Micro com PTQ INT8.

Módulo de compatibilidade que delega para src.quantization.*.
Gera:
- model_int8.tflite (FlatBuffer)
- model_int8.h (array C via gerador Python puro)
- quantization_params.json (scales, zero_points)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf

from src.quantization.export_tflite import (
    export_tflite,
    extract_quantization_params,
    generate_c_header,
    validate_tflm_size,
)
from src.quantization.ptq import (
    calibrate,
    quantize_model,
    representative_dataset_factory,
    representative_dataset_random,
    representative_dataset_stratified,
    validate_int8_io,
)

__all__ = [
    "calibrate",
    "quantize_model",
    "export_tflm",
    "export_tflite",
    "validate_tflm_size",
    "validate_int8_io",
    "representative_dataset_factory",
    "representative_dataset_random",
    "representative_dataset_stratified",
    "aami_stratified_dataset_factory",
    "extract_quantization_params",
    "generate_c_header",
]


def aami_stratified_dataset_factory(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int = 200,
    seed: int = 42,
) -> Callable:
    """Dataset representativo estratificado pelas 5 classes AAMI.

    Parameters
    ----------
    X : np.ndarray
        Segmentos ECG, shape (n, 500, 1).
    y : np.ndarray
        Labels AAMI (strings 'N','S','V','F','Q' ou inteiros 0..4).
    n_samples : int
        Total de amostras representativas.
    seed : int
        Seed para reprodutibilidade.

    Returns
    -------
    Callable
        Generator compatível com TFLiteConverter.
    """
    return representative_dataset_stratified(X, y, n_samples=n_samples, seed=seed)


def export_tflm(
    model: tf.keras.Model,
    representative_data: Callable,
    output_dir: Path,
    model_name: str = "model_int8",
) -> Path:
    """Exporta modelo Keras para TFLM INT8 (wrapper legado).

    Parameters
    ----------
    model : tf.keras.Model
        Modelo treinado (float32).
    representative_data : Callable
        Generator de dados representativos.
    output_dir : Path
        Diretório de saída.
    model_name : str
        Nome base dos arquivos.

    Returns
    -------
    Path
        Caminho do .tflite gerado.
    """
    return export_tflite(
        model,
        representative_data,
        output_dir,
        model_name=model_name,
    )
