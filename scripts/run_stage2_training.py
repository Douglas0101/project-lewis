"""Treinamento do Estágio 2: subtipificador S vs V vs F (Camada 04 v2.0)."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone_1d import build_backbone_1d, load_backbone_weights_from_pretrained
from src.models.train import train_group_kfold
from src.models.training_common import (
    build_base_arg_parser,
    build_groups as _build_groups,
    copy_best_fold_artifacts,
    load_config as _load_config,
    load_features as _load_features,
    resolve_loss,
    write_lineage,
)
from src.tracking.integrations import (
    finish_tracking_experiment,
    record_summary_metrics,
    start_tracking_experiment,
)

LOGGER = logging.getLogger("lewis.camada04.run_stage2")


def _thresholds_from_config(qg_cfg: dict) -> Dict[str, Any]:
    """Converte thresholds do config para o formato de ``evaluate_aami``."""
    f1_cfg = qg_cfg.get("f1", {})
    return {
        "min_acc": qg_cfg.get("min_acc", 0.70),
        "min_f1_macro": qg_cfg.get("min_f1_macro", 0.50),
        "min_mcc": qg_cfg.get("min_mcc", 0.40),
        "max_fpr_global": qg_cfg.get("max_fpr_global", 0.10),
        "per_class": {
            "S": {"F1": f1_cfg.get("S", 0.45)},
            "V": {"F1": f1_cfg.get("V", 0.70)},
            "F": {"F1": f1_cfg.get("F", 0.30)},
        },
    }


def _copy_best_fold(summary: Dict[str, Any], experiment_dir: Path, output_dir: Path, cfg: dict) -> None:
    artifact_map = {
        "model.keras": output_dir / cfg["output"]["model_filename"],
        "input_scaler.pkl": output_dir / cfg["output"]["scaler_filename"],
    }
    copy_best_fold_artifacts(summary, experiment_dir, output_dir, artifact_map)


def main() -> int:
    parser = build_base_arg_parser(
        description="Treinamento Estágio 2 — S vs V vs F",
        default_config=PROJECT_ROOT / "config" / "stage2_multiclass.yaml",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    cfg = _load_config(args.config)
    train_cfg = cfg["training"]
    ds_cfg = cfg["dataset"]
    qg_cfg = cfg["quality_gate"]["qg5_stage2"]

    X, y, df = _load_features(
        feature_npz=PROJECT_ROOT / ds_cfg["feature_npz"],
        feature_parquet=PROJECT_ROOT / ds_cfg["feature_parquet"],
    )
    groups = _build_groups(df)

    tracking_experiment_id = start_tracking_experiment(
        name=f"stage2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        stage="stage2",
        config_path=args.config,
        description="Treinamento Estágio 2 (S vs V vs F) v2.0",
    )

    experiment_dir = PROJECT_ROOT / "experiments" / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_stage2_v2.0"
    )

    thresholds = _thresholds_from_config(qg_cfg)
    class_names = ["S", "V", "F"]

    # Pesos balanceados para o desbalanceamento S/V/F (com teto para evitar viés excessivo)
    classes = np.unique(y)
    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )
    max_weight = float(ds_cfg.get("max_class_weight", 10.0))
    raw_weights = np.minimum(raw_weights, max_weight)
    class_weight = {int(cls): float(w) for cls, w in zip(classes, raw_weights)}
    LOGGER.info("Stage 2 class weights (max=%.1f): %s", max_weight, class_weight)

    def model_builder(input_len: int, num_classes: int) -> tf.keras.Model:
        model_cfg = cfg["model"]
        model = build_backbone_1d(
            input_len=input_len,
            num_classes=num_classes,
            embedding_dim=model_cfg.get("embedding_dim", 80),
            conv_filters=model_cfg.get("conv_filters", [16, 40, 80]),
            conv_kernels=model_cfg.get("conv_kernels", [7, 5, 3]),
            dense_units=model_cfg.get("dense_units", 80),
        )
        if (
            args.pretrained is not None
            and str(args.pretrained).strip() not in ("", ".")
            and args.pretrained.exists()
        ):
            LOGGER.info("Carregando pesos pré-treinados de %s", args.pretrained)
            model = load_backbone_weights_from_pretrained(args.pretrained, model)
        return model

    selection_metric = train_cfg.get("selection_metric", "F1_macro")
    loss = resolve_loss(train_cfg)

    # Compatibilidade: augmentation legado (class/factor) ou class-specific dict
    augment_cfg = cfg.get("augmentation", {})
    if isinstance(augment_cfg, dict) and "per_class" in augment_cfg:
        augment_config = augment_cfg["per_class"]
        augment_class = None
        augment_factor = 1
    else:
        augment_config = None
        augment_class = augment_cfg.get("class")
        augment_factor = augment_cfg.get("factor", 1)

    optimize_thresholds = bool(cfg.get("threshold_tuning", {}).get("enabled", False))

    summary = train_group_kfold(
        X=X,
        y=y,
        groups=groups,
        n_splits=args.n_splits if args.n_splits is not None else cfg["group_kfold"]["n_splits"],
        epochs=args.epochs if args.epochs is not None else train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
        seed=cfg["group_kfold"]["seed"],
        experiment_dir=experiment_dir,
        monitor=train_cfg["monitor"],
        class_names=class_names,
        thresholds=thresholds,
        model_builder=model_builder,
        class_weight=class_weight,
        selection_metric=selection_metric,
        augment_class=augment_class,
        augment_factor=augment_factor,
        augment_config=augment_config,
        loss=loss,
        optimize_thresholds=optimize_thresholds,
        freeze_backbone=args.freeze_backbone,
        tracking_experiment_id=tracking_experiment_id,
        tracking_stage_label="stage2",
        instrumentation_config=cfg.get("instrumentation"),
    )

    LOGGER.info(
        "Stage 2 complete | mean F1-macro=%.4f ± %.4f | passes QG=%s",
        summary["mean_metrics"]["F1_macro"],
        summary["std_metrics"]["F1_macro"],
        summary["passes_qg5"],
    )

    record_summary_metrics(
        experiment_id=tracking_experiment_id,
        summary=summary,
        stage_label="stage2",
    )
    finish_tracking_experiment(
        experiment_id=tracking_experiment_id,
        status="completed" if summary["passes_qg5"] else "failed",
    )

    lineage = {
        "experiment_dir": str(experiment_dir),
        "best_fold": summary["best_fold"],
        "mean_metrics": summary["mean_metrics"],
        "std_metrics": summary["std_metrics"],
        "passes_qg5": summary["passes_qg5"],
        "class_names": class_names,
        "thresholds": thresholds,
        "config": str(args.config),
    }
    write_lineage(lineage, cfg["output"]["model_filename"])

    _copy_best_fold(summary, experiment_dir, args.output_dir, cfg)

    return 0 if summary["passes_qg5"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
