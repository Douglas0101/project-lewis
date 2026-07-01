"""Testes de degradação por quantização INT8 full-integer (QG6).

Compara modelos Keras float32 e modelos TFLite INT8 nos mesmos subconjuntos de
validação, verificando que a queda de F1-macro atende o threshold do quality
gate QG6 (ΔF1-macro < 2 pontos percentuais).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from src.inference.quantized_runner import QuantizedModelRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

STAGE1_FLOAT = MODELS_DIR / "stage1_float32_v2.0.keras"
STAGE1_INT8 = MODELS_DIR / "quantized" / "stage1_int8_v2.0.tflite"
STAGE1_SCALER = MODELS_DIR / "input_scaler_stage1_v2.0.pkl"
STAGE1_THRESHOLD = MODELS_DIR / "stage1_threshold_v2.0.json"
STAGE1_DATA = FEATURES_DIR / "stage1_binary.npz"

STAGE2_FLOAT = MODELS_DIR / "stage2_float32_v2.0.keras"
STAGE2_INT8 = MODELS_DIR / "quantized" / "stage2_int8_v2.0.tflite"
STAGE2_SCALER = MODELS_DIR / "input_scaler_stage2_v2.0.pkl"
STAGE2_DATA = FEATURES_DIR / "stage2_multiclass.npz"

MAX_SAMPLES = 1024
DELTA_F1_THRESHOLD = 0.02


pytest.importorskip("tensorflow")


def _stage1_threshold() -> float:
    """Carrega o threshold do Estágio 1 ou retorna 0.5 como padrão."""
    if STAGE1_THRESHOLD.exists():
        data = json.loads(STAGE1_THRESHOLD.read_text(encoding="utf-8"))
        return float(data.get("threshold", 0.5))
    return 0.5


def _load_stage_data(stage: int, max_samples: int = MAX_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    """Carrega um subconjunto estratificado dos dados de validação do estágio."""
    if stage == 1:
        data_path = STAGE1_DATA
    elif stage == 2:
        data_path = STAGE2_DATA
    else:
        raise ValueError(f"Estágio deve ser 1 ou 2; recebido {stage}")

    if not data_path.exists():
        pytest.skip(f"Dados de features não encontrados: {data_path}")

    archive = np.load(data_path, allow_pickle=True)
    X = archive["X"]
    y = archive["y"]

    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X), size=max_samples, replace=False)
        # Preserva estratificação aproximada ordenando por label antes de amostrar.
        sorted_idx = np.argsort(y[indices])
        indices = indices[sorted_idx]
        X = X[indices]
        y = y[indices]

    return np.asarray(X, dtype=np.float32), np.asarray(y)


def _normalize(X: np.ndarray, scaler_path: Path) -> np.ndarray:
    """Aplica o scaler de entrada com reshape canônico (n, seq_len, 1)."""
    if not scaler_path.exists():
        pytest.skip(f"Scaler não encontrado: {scaler_path}")

    scaler = joblib.load(scaler_path)
    n, seq_len, channels = X.shape
    return scaler.transform(X.reshape(-1, channels)).reshape(n, seq_len, channels)


def _predict_float(model_path: Path, X: np.ndarray, stage: int) -> np.ndarray:
    """Executa inferência com modelo Keras float32 e retorna classes preditas."""
    import tensorflow as tf

    model = tf.keras.models.load_model(str(model_path), compile=False)
    logits = model.predict(X, verbose=0)

    if stage == 1:
        threshold = _stage1_threshold()
        return (logits[:, 1] >= threshold).astype(np.int64)
    return np.argmax(logits, axis=1).astype(np.int64)


def _predict_quantized(model_path: Path, X: np.ndarray, stage: int) -> np.ndarray:
    """Executa inferência com modelo TFLite INT8 e retorna classes preditas.

    O modelo INT8 exportado possui batch fixo em 1; portanto a execução é
    amostra a amostra, acumulando os logits para cálculo das métricas.
    """
    runner = QuantizedModelRunner(model_path).allocate()
    outputs: list[np.ndarray] = []
    for sample in X:
        outputs.append(runner.run(sample))
    logits = np.concatenate(outputs, axis=0)

    if stage == 1:
        threshold = _stage1_threshold()
        return (logits[:, 1] >= threshold).astype(np.int64)
    return np.argmax(logits, axis=1).astype(np.int64)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula F1-macro, acurácia e MCC."""
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def _evaluate_stage(
    stage: int,
    max_samples: int = MAX_SAMPLES,
) -> dict[str, Any]:
    """Avalia float32 e INT8 no mesmo subset e retorna métricas comparativas."""
    if stage == 1:
        float_path = STAGE1_FLOAT
        int8_path = STAGE1_INT8
        scaler_path = STAGE1_SCALER
    else:
        float_path = STAGE2_FLOAT
        int8_path = STAGE2_INT8
        scaler_path = STAGE2_SCALER

    if not float_path.exists():
        pytest.skip(f"Modelo float32 não encontrado: {float_path}")
    if not int8_path.exists():
        pytest.skip(f"Modelo INT8 não encontrado: {int8_path}")

    X_raw, y_true = _load_stage_data(stage, max_samples)
    X = _normalize(X_raw, scaler_path)

    y_pred_float = _predict_float(float_path, X, stage)
    y_pred_int8 = _predict_quantized(int8_path, X, stage)

    metrics_float = _compute_metrics(y_true, y_pred_float)
    metrics_int8 = _compute_metrics(y_true, y_pred_int8)

    return {
        "stage": stage,
        "n_samples": len(y_true),
        "float": metrics_float,
        "int8": metrics_int8,
        "delta_f1_macro": abs(metrics_float["f1_macro"] - metrics_int8["f1_macro"]),
    }


@pytest.mark.qg6
def test_quantization_degradation_stage1() -> None:
    """QG6: degradação do Estágio 1 entre float32 e INT8 deve ser < 2%."""
    result = _evaluate_stage(1)

    print(
        f"Estágio 1 (n={result['n_samples']}): "
        f"float F1={result['float']['f1_macro']:.4f}, "
        f"INT8 F1={result['int8']['f1_macro']:.4f}, "
        f"ΔF1={result['delta_f1_macro']:.4f}"
    )

    assert result["delta_f1_macro"] < DELTA_F1_THRESHOLD, (
        f"Degradação de F1-macro do Estágio 1 excedeu {DELTA_F1_THRESHOLD}: "
        f"Δ={result['delta_f1_macro']:.4f}"
    )


@pytest.mark.qg6
def test_quantization_degradation_stage2() -> None:
    """QG6: degradação do Estágio 2 entre float32 e INT8 deve ser < 2%."""
    result = _evaluate_stage(2)

    print(
        f"Estágio 2 (n={result['n_samples']}): "
        f"float F1={result['float']['f1_macro']:.4f}, "
        f"INT8 F1={result['int8']['f1_macro']:.4f}, "
        f"ΔF1={result['delta_f1_macro']:.4f}"
    )

    assert result["delta_f1_macro"] < DELTA_F1_THRESHOLD, (
        f"Degradação de F1-macro do Estágio 2 excedeu {DELTA_F1_THRESHOLD}: "
        f"Δ={result['delta_f1_macro']:.4f}"
    )
