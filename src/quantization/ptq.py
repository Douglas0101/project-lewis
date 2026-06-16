"""Post-Training Quantization (PTQ) INT8 full-integer para TFLM.

Fornece calibração e conversão de modelos Keras para TensorFlow Lite
com quantização INT8 full-integer, adequada para execução em
microcontroladores via TensorFlow Lite Micro.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tensorflow as tf

LOGGER = logging.getLogger("lewis.camada05.ptq")


def calibrate(
    model: tf.keras.Model,
    representative_data: Callable,
    allow_float: bool = False,
) -> tf.lite.Interpreter:
    """Calibra e converte modelo Keras para TFLite INT8 full-integer.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo treinado em float32.
    representative_data : Callable
        Generator compatível com TFLiteConverter (yield [np.ndarray]).
    allow_float : bool
        Se True, permite fallback para operações float (delega quantização
        parcial). Default False para full-integer obrigatório.

    Returns
    -------
    bytes
        Modelo TFLite quantizado em INT8.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data

    if allow_float:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
    else:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    LOGGER.info("Convertendo modelo para TFLite INT8 full-integer...")
    tflite_model = converter.convert()
    LOGGER.info("Conversão concluída: %d bytes", len(tflite_model))
    return tflite_model


def quantize_model(
    model: tf.keras.Model,
    representative_data: Callable,
    output_path: Path | str,
    allow_float: bool = False,
) -> Path:
    """Quantiza modelo e persiste o FlatBuffer .tflite.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo Keras treinado.
    representative_data : Callable
        Dataset representativo (generator).
    output_path : Path | str
        Caminho para salvar o arquivo .tflite.
    allow_float : bool
        Permite fallback float se necessário.

    Returns
    -------
    Path
        Caminho do .tflite salvo.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tflite_model = calibrate(model, representative_data, allow_float=allow_float)
    output_path.write_bytes(tflite_model)

    size_kb = len(tflite_model) / 1024
    LOGGER.info("Modelo quantizado salvo: %s (%.2f KB)", output_path, size_kb)
    return output_path


def representative_dataset_random(
    X: np.ndarray,
    n_samples: int = 200,
    seed: int = 42,
) -> Callable:
    """Factory de dataset representativo por amostragem aleatória.

    Parameters
    ----------
    X : np.ndarray
        Dados de calibração, shape (n, 500, 1).
    n_samples : int
        Número máximo de amostras representativas.
    seed : int
        Seed para reprodutibilidade.

    Returns
    -------
    Callable
        Generator compatível com TFLiteConverter.
    """
    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, len(X))
    indices = rng.choice(len(X), size=n_samples, replace=False)
    samples = X[indices].astype(np.float32)

    def _generator():
        for sample in samples:
            yield [np.expand_dims(sample, axis=0)]

    return _generator


def representative_dataset_factory(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    n_samples: int = 200,
    seed: int = 42,
) -> Callable:
    """Factory de dataset representativo, estratificada quando y é fornecido.

    Se ``y`` for None, realiza amostragem aleatória. Caso contrário,
    garante representação proporcional das classes AAMI (ou quaisquer
    classes em ``y``).

    Parameters
    ----------
    X : np.ndarray
        Dados de calibração, shape (n, length, 1).
    y : np.ndarray | None
        Labels correspondentes (strings ou inteiros).
    n_samples : int
        Número total de amostras representativas.
    seed : int
        Seed para reprodutibilidade.

    Returns
    -------
    Callable
        Generator compatível com TFLiteConverter.
    """
    if y is None:
        return representative_dataset_random(X, n_samples=n_samples, seed=seed)
    return representative_dataset_stratified(X, y, n_samples=n_samples, seed=seed)


def representative_dataset_stratified(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int = 200,
    seed: int = 42,
) -> Callable:
    """Factory de dataset representativo estratificado por classe.

    Garante que cada classe presente em ``y`` tenha pelo menos uma
    amostra, distribuindo o restante proporcionalmente.

    Parameters
    ----------
    X : np.ndarray
        Dados de calibração, shape (n, length, 1).
    y : np.ndarray
        Labels (strings ou inteiros).
    n_samples : int
        Número total de amostras.
    seed : int
        Seed para reprodutibilidade.

    Returns
    -------
    Callable
        Generator compatível com TFLiteConverter.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    n_samples = max(n_classes, min(n_samples, len(X)))

    # Pelo menos 1 amostra por classe; restante proporcional
    base = np.ones(n_classes, dtype=int)
    remaining = n_samples - n_classes
    if remaining > 0:
        proportions = counts / counts.sum()
        extra = np.floor(proportions * remaining).astype(int)
        # Distribuir possíveis sobras
        for _ in range(remaining - int(extra.sum())):
            fractional = proportions - extra / max(remaining, 1)
            idx = int(np.argmax(fractional))
            extra[idx] += 1
        per_class = base + extra
    else:
        per_class = base

    selected_indices: list[int] = []
    for cls, n in zip(classes, per_class):
        indices = np.where(y == cls)[0].astype(np.int64)
        n_selected = min(int(n), int(len(indices)))
        chosen = rng.choice(indices, size=n_selected, replace=False)
        selected_indices.extend(chosen.tolist())

    rng.shuffle(selected_indices)
    samples = X[selected_indices].astype(np.float32)

    def _generator():
        for sample in samples:
            yield [np.expand_dims(sample, axis=0)]

    return _generator


def validate_int8_io(tflite_model: bytes) -> bool:
    """Verifica se modelo quantizado possui entrada/saída int8.

    Parameters
    ----------
    tflite_model : bytes
        Conteúdo do modelo TFLite.

    Returns
    -------
    bool
        True se entrada e saída são int8.
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_dtype = interpreter.get_input_details()[0]["dtype"]
    output_dtype = interpreter.get_output_details()[0]["dtype"]
    return input_dtype == np.int8 and output_dtype == np.int8


def main(
    model_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Entry-point CLI para quantização PTQ.

    Parameters
    ----------
    model_path : Path | None
        Caminho do modelo Keras salvo. Se None, usa modelo dummy.
    output_path : Path | None
        Caminho de saída do .tflite.

    Returns
    -------
    Path
        Caminho do .tflite gerado.
    """
    logging.basicConfig(level=logging.INFO)

    if output_path is None:
        output_path = Path("models/quantized/model_int8.tflite")

    if model_path is not None:
        model = tf.keras.models.load_model(model_path)
    else:
        from src.models.backbone_1d import build_backbone_1d

        model = build_backbone_1d(input_len=500, num_classes=5)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    X = np.random.randn(100, 500, 1).astype(np.float32)
    rep_data = representative_dataset_random(X, n_samples=50, seed=42)

    return quantize_model(model, rep_data, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PTQ INT8 para Project-Lewis")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Caminho do modelo Keras (.keras ou .h5).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("models/quantized/model_int8.tflite"),
        help="Caminho de saída do .tflite.",
    )
    args = parser.parse_args()
    main(args.model_path, args.output_path)
