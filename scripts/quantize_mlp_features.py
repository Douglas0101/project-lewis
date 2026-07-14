"""Quantização INT8 full-integer dos MLPs sobre features v2.3.

Diferente do quantize_two_stage_v2.0.py, aqui os modelos esperam vetores de
features 2D (n_samples, n_features), não sinais 1D com canal.
"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.quantization.export_tflite import export_tflite, validate_tflm_size
from src.quantization.ptq import representative_dataset_factory

LOGGER = logging.getLogger("lewis.camada05.quantize_mlp_features")


def _balanced_calibration_dataset(
    X: np.ndarray,
    y: np.ndarray,
    n_cal: int = 500,
    seed: int = 42,
):
    """Gera dataset de calibração INT8 estratificado 50/50 (Stage 1 binário).

    Garante que a calibração não seja enviesada para a classe majoritária,
    preservando a sensibilidade da classe minoritária no modelo quantizado.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"_balanced_calibration_dataset espera 2 classes, tem {len(classes)}")

    n_per_class = n_cal // 2
    selected: list[int] = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        n = min(n_per_class, len(idx))
        selected.extend(rng.choice(idx, size=n, replace=False).tolist())

    rng.shuffle(selected)
    samples = X[selected].astype(np.float32)

    def _generator():
        for sample in samples:
            yield [np.expand_dims(sample, axis=0)]

    return _generator


def _quantize_model(
    keras_path: Path,
    scaler_path: Path,
    feature_npz: Path,
    output_name: str,
    output_dir: Path,
    n_cal: int = 500,
) -> Dict[str, Any]:
    """Quantiza um modelo Keras MLP de features e salva artefatos TFLM."""
    LOGGER.info("Quantizando %s -> %s", keras_path, output_name)

    model = tf.keras.models.load_model(str(keras_path), compile=False)
    scaler = joblib.load(scaler_path)

    data = np.load(feature_npz)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    # Preserva shape 2D (n_samples, n_features) — sem adicionar canal.
    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {X.shape}")
    if scaler.n_features_in_ != X.shape[1]:
        raise ValueError(
            f"Scaler espera {scaler.n_features_in_} features, mas os dados têm {X.shape[1]}."
        )

    # O modelo MLP foi treinado sobre features escalonadas; a calibração
    # precisa receber dados na mesma escala.
    X_scaled = scaler.transform(X)

    # Para Stage 1 (binário), força calibração balanceada 50/50 para evitar
    # viés contra a classe minoritária. Stage 2 usa estratificação proporcional.
    if len(np.unique(y)) == 2 and "stage1" in output_name:
        representative_data = _balanced_calibration_dataset(X_scaled, y, n_cal=n_cal)
    else:
        representative_data = representative_dataset_factory(X_scaled, y=y, n_samples=n_cal)

    tflite_path = export_tflite(
        model=model,
        representative_data=representative_data,
        output_dir=output_dir,
        model_name=output_name,
        version="2.3.0",
        allow_float=False,
    )

    size_kb = tflite_path.stat().st_size / 1024
    passes_size = validate_tflm_size(tflite_path, max_kb=64)

    result = {
        "keras": str(keras_path),
        "tflite": str(tflite_path),
        "size_kb": round(size_kb, 2),
        "passes_qg6_size": bool(passes_size),
    }
    LOGGER.info(
        "%s | size=%.2f KB | passes QG6 size=%s",
        output_name,
        size_kb,
        passes_size,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quantização INT8 dos MLPs sobre features v2.3"
    )
    parser.add_argument(
        "--stage1-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "stage1_float32_v2.3.keras",
    )
    parser.add_argument(
        "--stage1-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "stage1_binary_features.npz",
    )
    parser.add_argument(
        "--stage1-scaler",
        type=Path,
        default=PROJECT_ROOT / "models" / "input_scaler_stage1_v2.3.pkl",
    )
    parser.add_argument(
        "--stage2-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "stage2_float32_v2.3.keras",
    )
    parser.add_argument(
        "--stage2-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features.npz",
    )
    parser.add_argument(
        "--stage2-scaler",
        type=Path,
        default=PROJECT_ROOT / "models" / "input_scaler_stage2_v2.3.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "quantized",
    )
    parser.add_argument(
        "--n-cal",
        type=int,
        default=500,
        help="Número de amostras para calibração INT8",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "stage1": {},
        "stage2": {},
    }

    if args.stage1_model.exists():
        summary["stage1"] = _quantize_model(
            args.stage1_model,
            args.stage1_scaler,
            args.stage1_features,
            "stage1_int8_v2.3",
            args.output_dir,
            n_cal=args.n_cal,
        )
    else:
        LOGGER.warning("Modelo Stage1 não encontrado: %s", args.stage1_model)

    if args.stage2_model.exists():
        summary["stage2"] = _quantize_model(
            args.stage2_model,
            args.stage2_scaler,
            args.stage2_features,
            "stage2_int8_v2.3",
            args.output_dir,
            n_cal=args.n_cal,
        )
    else:
        LOGGER.warning("Modelo Stage2 não encontrado: %s", args.stage2_model)

    summary_path = args.output_dir / "quantization_summary_v2.3.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    LOGGER.info("Resumo de quantização salvo em %s", summary_path)

    passes = all(r.get("passes_qg6_size", False) for r in summary.values() if r)
    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
