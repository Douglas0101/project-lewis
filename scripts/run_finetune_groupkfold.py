"""Orchestrate GroupKFold fine-tuning for MIT-BIH+ (Camada 04).

Usage:
    python scripts/run_finetune_groupkfold.py \
        --config config/finetune_v1.0.yaml \
        --backbone models/backbone_pretrained_v1.0.keras
"""

from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import train_group_kfold
from src.models.training_common import (
    build_base_arg_parser,
    build_groups as _build_groups,
    load_config as _load_config,
    load_features as _load_features,
    write_lineage,
)
from src.tracking.integrations import (
    finish_tracking_experiment,
    record_summary_metrics,
    start_tracking_experiment,
)

LOGGER = logging.getLogger("lewis.camada04.run_finetune")


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
    parser = build_base_arg_parser(
        description="Fine-tuning Project-Lewis em MIT-BIH+ com GroupKFold por paciente",
        default_config=PROJECT_ROOT / "config" / "finetune_v1.0.yaml",
        include_pretrained=False,
        include_batch_size_lr=True,
    )
    parser.add_argument(
        "--backbone",
        type=Path,
        default=None,
        help="Caminho para backbone pré-treinado (.keras). Se omitido, treina do zero.",
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
    }
    write_lineage(lineage, cfg["output"]["model_filename"])

    _copy_best_fold(summary, experiment_dir, args.output_dir)

    return 0 if summary["passes_qg5"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
