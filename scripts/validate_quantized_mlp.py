"""Valida equivalência entre MLP float32 e INT8 quantizado (QG6/QG10)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.quantized_runner import QuantizedModelRunner
from src.inference.two_stage_mlp_pipeline import TwoStageMLPPipeline
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("validate_quantized_mlp")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _evaluate_stage1(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
    is_quantized: bool = False,
) -> Dict[str, float]:
    if is_quantized:
        # QuantizedModelRunner expects dequantized float output.
        proba = np.array([model.run(x[np.newaxis, ...])[0] for x in X])
    else:
        proba = model.predict(X, batch_size=4096, verbose=0)
    score_anormal = proba[:, 1]
    y_pred = (score_anormal >= threshold).astype(np.int64)
    y_true_bin = (y != 0).astype(np.int64)
    return {
        "f1_macro": float(
            f1_score(y_true_bin, y_pred, labels=[0, 1], average="macro", zero_division=0.0)
        ),
        "accuracy": float(accuracy_score(y_true_bin, y_pred)),
    }


def _evaluate_stage2(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    is_quantized: bool = False,
) -> Dict[str, float]:
    if is_quantized:
        proba = np.array([model.run(x[np.newaxis, ...])[0] for x in X])
    else:
        proba = model.predict(X, batch_size=4096, verbose=0)
    y_pred = np.argmax(proba, axis=1).astype(np.int64)
    return {
        "f1_macro": float(
            f1_score(y, y_pred, labels=[0, 1, 2], average="macro", zero_division=0.0)
        ),
        "accuracy": float(accuracy_score(y, y_pred)),
        "per_class_f1": {
            cls: float(score)
            for cls, score in zip(
                ["S", "V", "F"],
                f1_score(y, y_pred, labels=[0, 1, 2], average=None, zero_division=0.0),
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida MLP float32 vs INT8")
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()

    # Stage1
    s1_data = np.load(PROJECT_ROOT / "data/features/stage1_binary_features.npz")
    X1, y1 = s1_data["X"].astype(np.float32), s1_data["y"].astype(np.int64)
    if args.max_samples < len(X1):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X1), size=args.max_samples, replace=False)
        X1, y1 = X1[idx], y1[idx]

    threshold = json.loads(
        (PROJECT_ROOT / "models/stage1_threshold_v2.3.json").read_text(encoding="utf-8")
    )["threshold"]

    s1_float = tf.keras.models.load_model(
        PROJECT_ROOT / "models/stage1_float32_v2.3.keras", compile=False
    )
    s1_int8 = QuantizedModelRunner(
        PROJECT_ROOT / "models/quantized/stage1_int8_v2.3.tflite"
    ).allocate()

    m1_float = _evaluate_stage1(s1_float, X1, y1, threshold)
    m1_int8 = _evaluate_stage1(s1_int8, X1, y1, threshold, is_quantized=True)

    LOGGER.info("Stage1 float32: %s", m1_float)
    LOGGER.info("Stage1 int8:    %s", m1_int8)

    # Stage2
    s2_data = np.load(PROJECT_ROOT / "data/features/stage2_multiclass_features.npz")
    X2, y2 = s2_data["X"].astype(np.float32), s2_data["y"].astype(np.int64)
    scaler2 = joblib.load(PROJECT_ROOT / "models/input_scaler_stage2_v2.3.pkl")
    X2 = scaler2.transform(X2)

    # Stratified subset
    selected: list[int] = []
    rng = np.random.default_rng(42)
    for cls in range(3):
        idx = np.where(y2 == cls)[0]
        n = min(len(idx), args.max_samples // 3)
        selected.extend(rng.choice(idx, size=n, replace=False).tolist())
    selected = np.array(selected)
    rng.shuffle(selected)
    X2, y2 = X2[selected], y2[selected]

    s2_float = tf.keras.models.load_model(
        PROJECT_ROOT / "models/stage2_float32_v2.3.keras", compile=False
    )
    s2_int8 = QuantizedModelRunner(
        PROJECT_ROOT / "models/quantized/stage2_int8_v2.3.tflite"
    ).allocate()

    m2_float = _evaluate_stage2(s2_float, X2, y2)
    m2_int8 = _evaluate_stage2(s2_int8, X2, y2, is_quantized=True)

    LOGGER.info("Stage2 float32: %s", m2_float)
    LOGGER.info("Stage2 int8:    %s", m2_int8)

    delta1 = abs(m1_float["f1_macro"] - m1_int8["f1_macro"])
    delta2 = abs(m2_float["f1_macro"] - m2_int8["f1_macro"])
    LOGGER.info("Delta F1-macro Stage1: %.4f", delta1)
    LOGGER.info("Delta F1-macro Stage2: %.4f", delta2)

    ok = delta1 < 0.02 and delta2 < 0.02
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
