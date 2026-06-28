"""Quantiza o modelo finetuned_float32_v1.1.keras para INT8/TFLM."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupKFold

from src.models.backbone_1d import build_backbone_1d
from src.quantization.export_tflite import export_tflite, validate_tflm_size

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("lewis.camada05.quantize")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "finetuned_float32_v1.1.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "input_scaler_v1.1.pkl"
FEATURE_NPZ = PROJECT_ROOT / "data" / "features" / "finetuning_mitbih_family.npz"
FEATURE_PARQUET = PROJECT_ROOT / "data" / "features" / "finetuning_mitbih_family.parquet"
OUTPUT_DIR = PROJECT_ROOT / "models" / "quantized"
VERSION = "1.1.0"


def _representative_data_gen(
    X: np.ndarray,
    batch_size: int = 1,
    n_batches: int = 200,
):
    """Yields representative batches for full-integer quantization."""
    n = len(X)
    indices = np.random.RandomState(42).choice(n, size=n_batches * batch_size, replace=False)
    for i in range(n_batches):
        batch = X[indices[i * batch_size : (i + 1) * batch_size]]
        yield [batch.astype(np.float32)]


def main() -> int:
    LOGGER.info("Loading model from %s", MODEL_PATH)
    model = tf.keras.models.load_model(str(MODEL_PATH))

    LOGGER.info("Loading features")
    npz = np.load(FEATURE_NPZ)
    X_full = npz["X"].astype(np.float32)
    if X_full.ndim == 2:
        X_full = X_full[..., np.newaxis]
    y_full = npz["y"].astype(np.int64)
    df = pd.read_parquet(FEATURE_PARQUET)
    groups = df["record_id"].astype(str).values

    LOGGER.info("Loading scaler from %s", SCALER_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Use fold-0 training data as representative (same fold as the saved scaler)
    gkf = GroupKFold(n_splits=5)
    for fold_idx, (train_idx, _) in enumerate(gkf.split(X_full, y_full, groups)):
        if fold_idx == 0:
            X_train = X_full[train_idx]
            break
    X_train_norm = (
        scaler.transform(X_train.reshape(-1, 1))
        .reshape(-1, 500, 1)
        .astype(np.float32)
    )

    LOGGER.info("Exporting INT8 TFLite to %s", OUTPUT_DIR)
    tflite_path = export_tflite(
        model=model,
        representative_data=lambda: _representative_data_gen(X_train_norm),
        output_dir=OUTPUT_DIR,
        model_name="finetuned_int8_v1.1",
        version=VERSION,
        allow_float=False,
    )

    if validate_tflm_size(tflite_path, max_kb=64):
        LOGGER.info("TFLite model is within 64 KB limit.")
        return 0
    LOGGER.error("TFLite model exceeds 64 KB limit.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
