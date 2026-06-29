"""Quantização INT8 full-integer dos modelos v2.0 (Estágio 1 e Estágio 2)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.quantization.export_tflite import export_tflite, validate_tflm_size
from src.quantization.ptq import representative_dataset_factory

LOGGER = logging.getLogger("lewis.camada05.quantize_two_stage")


def _quantize_model(
    keras_path: Path,
    feature_npz: Path,
    output_name: str,
    output_dir: Path,
    n_cal: int = 500,
) -> Dict[str, Any]:
    """Quantiza um modelo Keras e salva artefatos TFLM."""
    LOGGER.info("Quantizando %s -> %s", keras_path, output_name)

    model = tf.keras.models.load_model(str(keras_path), compile=False)

    data = np.load(feature_npz)
    X = data["X"].astype(np.float32)
    if X.ndim == 2:
        X = X[..., np.newaxis]

    representative_data = representative_dataset_factory(X, y=None, n_samples=n_cal)

    tflite_path = export_tflite(
        model=model,
        representative_data=representative_data,
        output_dir=output_dir,
        model_name=output_name,
        version="2.0.0",
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
        description="Quantização INT8 dos modelos v2.0"
    )
    parser.add_argument(
        "--stage1-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "stage1_float32_v2.0.keras",
    )
    parser.add_argument(
        "--stage1-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "stage1_binary.npz",
    )
    parser.add_argument(
        "--stage2-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "stage2_float32_v2.0.keras",
    )
    parser.add_argument(
        "--stage2-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "stage2_multiclass.npz",
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
            args.stage1_features,
            "stage1_int8_v2.0",
            args.output_dir,
            n_cal=args.n_cal,
        )
    else:
        LOGGER.warning("Modelo Stage1 não encontrado: %s", args.stage1_model)

    if args.stage2_model.exists():
        summary["stage2"] = _quantize_model(
            args.stage2_model,
            args.stage2_features,
            "stage2_int8_v2.0",
            args.output_dir,
            n_cal=args.n_cal,
        )
    else:
        LOGGER.warning("Modelo Stage2 não encontrado: %s", args.stage2_model)

    summary_path = args.output_dir / "quantization_summary_v2.0.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    LOGGER.info("Resumo de quantização salvo em %s", summary_path)

    passes = all(r.get("passes_qg6_size", False) for r in summary.values() if r)
    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
