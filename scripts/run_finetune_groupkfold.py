"""Orchestrate GroupKFold fine-tuning for MIT-BIH+ (Camada 04).

Usage:
    python scripts/run_finetune_groupkfold.py \
        --config config/finetune_v1.0.yaml \
        --backbone models/backbone_pretrained_v1.0.keras
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import train_group_kfold
from src.tracking.integrations import (
    finish_tracking_experiment,
    record_summary_metrics,
    start_tracking_experiment,
)

LOGGER = logging.getLogger("lewis.camada04.run_finetune")


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_features(feature_npz: Path, feature_parquet: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load segmented beats and metadata."""
    LOGGER.info("Loading features from %s", feature_npz)
    data = np.load(feature_npz)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    # Adicionar dimensão de canal: (n, 500) -> (n, 500, 1)
    if X.ndim == 2:
        X = X[..., np.newaxis]

    LOGGER.info("Loading metadata from %s", feature_parquet)
    df = pd.read_parquet(feature_parquet)

    if len(X) != len(df) or len(y) != len(df):
        raise ValueError(
            f"Mismatch: X={len(X)}, y={len(y)}, df={len(df)}"
        )

    LOGGER.info("Loaded %d beats | X shape=%s | classes=%d", len(X), X.shape, len(np.unique(y)))
    return X, y, df


def _build_groups(df: pd.DataFrame) -> np.ndarray:
    """Build group array from record_id for GroupKFold.

    Each unique record_id is a distinct patient/recording. Never mix beats
    from the same record between train and test.
    """
    unique_records = df["record_id"].unique()
    record_to_group = {rec: idx for idx, rec in enumerate(unique_records)}
    groups = df["record_id"].map(record_to_group).to_numpy(dtype=np.int64)
    LOGGER.info("Built groups | n_patients=%d", len(unique_records))
    return groups


def _copy_best_fold(summary: Dict[str, Any], experiment_dir: Path, output_dir: Path) -> None:
    """Copy best fold model and scaler to canonical paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    best_fold = summary["best_fold"]
    best_fold_dir = experiment_dir / f"fold_{best_fold}"

    src_model = best_fold_dir / "finetuned_float32.keras"
    src_scaler = best_fold_dir / "input_scaler.pkl"

    dst_model = output_dir / "finetuned_float32_v1.0.keras"
    dst_scaler = output_dir / "input_scaler_v1.0.pkl"

    if src_model.exists():
        shutil.copy(str(src_model), str(dst_model))
        LOGGER.info("Best model copied to %s", dst_model)
    else:
        LOGGER.error("Best model not found at %s", src_model)

    if src_scaler.exists():
        shutil.copy(str(src_scaler), str(dst_scaler))
        LOGGER.info("Best scaler copied to %s", dst_scaler)
    else:
        LOGGER.error("Best scaler not found at %s", src_scaler)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tuning Project-Lewis em MIT-BIH+ com GroupKFold por paciente"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "finetune_v1.0.yaml",
        help="Caminho para config/finetune_v*.yaml",
    )
    parser.add_argument(
        "--backbone",
        type=Path,
        default=None,
        help="Caminho para backbone pré-treinado (.keras). Se omitido, treina do zero.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Diretório para salvar modelo final",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Sobrescrever número de épocas do config",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        help="Sobrescrever número de folds do config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Sobrescrever batch size do config",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Sobrescrever learning rate do config",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    cfg = _load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    ds_cfg = cfg["dataset"]

    freeze_backbone = args.backbone is not None
    if freeze_backbone and not args.backbone.exists():
        LOGGER.error("Backbone not found: %s", args.backbone)
        return 1

    X, y, df = _load_features(
        feature_npz=PROJECT_ROOT / ds_cfg["feature_npz"],
        feature_parquet=PROJECT_ROOT / ds_cfg["feature_parquet"],
    )
    groups = _build_groups(df)

    tracking_experiment_id = start_tracking_experiment(
        name=f"finetune_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        stage="finetune",
        config_path=args.config,
        description="Fine-tuning MIT-BIH+ com GroupKFold",
    )

    experiment_dir = PROJECT_ROOT / "experiments" / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_groupkfold"
    )

    summary = train_group_kfold(
        X=X,
        y=y,
        groups=groups,
        backbone_weights=args.backbone,
        freeze_backbone=freeze_backbone,
        n_splits=args.n_splits if args.n_splits is not None else cfg["group_kfold"]["n_splits"],
        epochs=args.epochs if args.epochs is not None else train_cfg["epochs"],
        batch_size=args.batch_size if args.batch_size is not None else train_cfg["batch_size"],
        learning_rate=args.learning_rate if args.learning_rate is not None else train_cfg["learning_rate"],
        seed=cfg["group_kfold"]["seed"],
        experiment_dir=experiment_dir,
        monitor=train_cfg["monitor"],
        tracking_experiment_id=tracking_experiment_id,
        tracking_stage_label="finetune",
    )

    LOGGER.info(
        "GroupKFold complete | mean F1-macro=%.4f ± %.4f | passes QG5=%s",
        summary["mean_metrics"]["F1_macro"],
        summary["std_metrics"]["F1_macro"],
        summary["passes_qg5"],
    )

    record_summary_metrics(
        experiment_id=tracking_experiment_id,
        summary=summary,
        stage_label="finetune",
    )
    finish_tracking_experiment(
        experiment_id=tracking_experiment_id,
        status="completed" if summary["passes_qg5"] else "failed",
    )

    # Persist lineage for the final model selection
    lineage = {
        "backbone_weights": str(args.backbone),
        "experiment_dir": str(experiment_dir),
        "best_fold": summary["best_fold"],
        "mean_metrics": summary["mean_metrics"],
        "std_metrics": summary["std_metrics"],
        "passes_qg5": summary["passes_qg5"],
        "config": str(args.config),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    lineage_dir = PROJECT_ROOT / "data" / "lineage" / "models"
    lineage_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(cfg["output"]["model_filename"]).stem
    lineage_path = lineage_dir / f"{model_name}.json"
    with lineage_path.open("w", encoding="utf-8") as fh:
        json.dump(lineage, fh, indent=2, ensure_ascii=False)
    LOGGER.info("Lineage saved to %s", lineage_path)

    _copy_best_fold(summary, experiment_dir, args.output_dir)

    return 0 if summary["passes_qg5"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
