"""Aplica pruning estruturado + QAT + conversão INT8 em um modelo Keras.

Este script carrega um modelo ``.keras``, executa o pipeline completo de
otimização (pruning estruturado de canais, fine-tuning, QAT e conversão TFLite
INT8) e exporta o FlatBuffer junto com os parâmetros de quantização em JSON.

Uso:
    python scripts/apply_pruning_qat.py \
        --model models/stage1_float32_v2.0.keras \
        --output-dir models/quantized \
        --data data/features/pruning_qat_data.npz \
        --target-sparsity 0.30 \
        --fine-tune-epochs 5 \
        --qat-epochs 3 \
        --batch-size 64 \
        --learning-rate 1e-4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pruning_qat import prune_qat_pipeline  # noqa: E402

LOGGER = logging.getLogger("lewis.scripts.apply_pruning_qat")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aplica pruning estruturado + QAT e exporta TFLite INT8."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Caminho para o modelo Keras (.keras).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Diretório de saída para .tflite e .json.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Arquivo .npz contendo X_train, y_train, X_val, y_val.",
    )
    parser.add_argument(
        "--target-sparsity",
        type=float,
        default=0.30,
        help="Fração de filtros Conv1D a remover (default: 0.30).",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=5,
        help="Épocas de fine-tuning após pruning (default: 5).",
    )
    parser.add_argument(
        "--qat-epochs",
        type=int,
        default=3,
        help="Épocas de fine-tuning durante QAT (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Tamanho do batch (default: 64).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Taxa de aprendizado (default: 1e-4).",
    )
    return parser.parse_args(argv)


def _load_data(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carrega arrays de treino e validação de um arquivo NPZ."""
    data = np.load(npz_path)
    required = {"X_train", "y_train", "X_val", "y_val"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Arquivo NPZ incompleto. Faltando: {missing}")
    return (
        data["X_train"].astype(np.float32),
        data["y_train"].astype(np.int64),
        data["X_val"].astype(np.float32),
        data["y_val"].astype(np.int64),
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point do script."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    if not args.model.exists():
        LOGGER.error("Modelo não encontrado: %s", args.model)
        return 1

    X_train, y_train, X_val, y_val = _load_data(args.data)
    LOGGER.info(
        "Dados carregados | train=%d | val=%d | shape=%s",
        len(X_train),
        len(X_val),
        X_train.shape,
    )

    result: Dict[str, Any] = prune_qat_pipeline(
        model_path=args.model,
        output_dir=args.output_dir,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        target_sparsity=args.target_sparsity,
        fine_tune_epochs=args.fine_tune_epochs,
        qat_epochs=args.qat_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    LOGGER.info(
        "Pipeline concluído | params originais=%d | params podados=%d | "
        "redução=%.1f%% | tflite=%s | params=%s",
        result["original_params"],
        result["pruned_params"],
        result["reduction_pct"],
        result["tflite_path"],
        result["params_path"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
