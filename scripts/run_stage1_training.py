"""Treinamento do Estágio 1: detector binário N vs Anormal (Camada 04 v2.0)."""

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
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone_1d import build_backbone_1d, load_backbone_weights_from_pretrained
from src.models.finetune_mitbih import SparseCategoricalFocalLoss
from src.models.train import train_group_kfold
from src.tracking.integrations import (
    finish_tracking_experiment,
    record_summary_metrics,
    start_tracking_experiment,
)

LOGGER = logging.getLogger("lewis.camada04.run_stage1")


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_features(feature_npz: Path, feature_parquet: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    LOGGER.info("Loading features from %s", feature_npz)
    data = np.load(feature_npz)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    if X.ndim == 2:
        X = X[..., np.newaxis]

    LOGGER.info("Loading metadata from %s", feature_parquet)
    df = pd.read_parquet(feature_parquet)

    if len(X) != len(df) or len(y) != len(df):
        raise ValueError(f"Mismatch: X={len(X)}, y={len(y)}, df={len(df)}")

    LOGGER.info("Loaded %d beats | X shape=%s | classes=%d", len(X), X.shape, len(np.unique(y)))
    return X, y, df


def _build_groups(df: pd.DataFrame) -> np.ndarray:
    unique_records = df["record_id"].unique()
    record_to_group = {rec: idx for idx, rec in enumerate(unique_records)}
    groups = df["record_id"].map(record_to_group).to_numpy(dtype=np.int64)
    LOGGER.info("Built groups | n_patients=%d", len(unique_records))
    return groups


def _thresholds_from_config(qg_cfg: dict) -> Dict[str, Any]:
    """Converte thresholds do config para o formato de ``evaluate_aami``."""
    return {
        "min_acc": qg_cfg.get("min_acc", 0.92),
        "min_f1_macro": qg_cfg.get("min_f1_macro", 0.90),
        "min_mcc": qg_cfg.get("min_mcc", 0.70),
        "max_fpr_global": qg_cfg.get("max_fpr_global", 0.05),
        "per_class": {
            "N": {"Se": qg_cfg.get("sensitivity_n", 0.90)},
            "Anormal": {
                "Se": qg_cfg.get("recall_anormal", 0.95),
                "PPV": qg_cfg.get("precision_anormal", 0.70),
            },
        },
    }


def _copy_best_fold(summary: Dict[str, Any], experiment_dir: Path, output_dir: Path, cfg: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_fold = summary["best_fold"]
    best_fold_dir = experiment_dir / f"fold_{best_fold}"

    src_model = best_fold_dir / "model.keras"
    src_scaler = best_fold_dir / "input_scaler.pkl"
    src_threshold = best_fold_dir / "best_weights.weights.threshold.json"

    dst_model = output_dir / cfg["output"]["model_filename"]
    dst_scaler = output_dir / cfg["output"]["scaler_filename"]
    dst_threshold = output_dir / "stage1_threshold.json"

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

    if src_threshold.exists():
        shutil.copy(str(src_threshold), str(dst_threshold))
        LOGGER.info("Best threshold copied to %s", dst_threshold)
    else:
        LOGGER.warning("Threshold file not found at %s", src_threshold)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Treinamento Estágio 1 — N vs Anormal"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "stage1_binary.yaml",
        help="Caminho para config/stage1_binary.yaml",
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
        help="Sobrescrever número de épocas",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        help="Sobrescrever número de folds",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=PROJECT_ROOT / "models" / "finetuned_float32_v1.1.keras",
        help="Modelo pré-treinado para inicializar o backbone (ex.: v1.1)",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Congelar camadas convolucionais do backbone pré-treinado",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    cfg = _load_config(args.config)
    train_cfg = cfg["training"]
    ds_cfg = cfg["dataset"]
    qg_cfg = cfg["quality_gate"]["qg5_stage1"]

    X, y, df = _load_features(
        feature_npz=PROJECT_ROOT / ds_cfg["feature_npz"],
        feature_parquet=PROJECT_ROOT / ds_cfg["feature_parquet"],
    )
    groups = _build_groups(df)

    tracking_experiment_id = start_tracking_experiment(
        name=f"stage1_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        stage="stage1",
        config_path=args.config,
        description="Treinamento Estágio 1 (N vs Anormal) v2.0",
    )

    experiment_dir = PROJECT_ROOT / "experiments" / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_stage1_v2.0"
    )

    thresholds = _thresholds_from_config(qg_cfg)
    class_names = ["N", "Anormal"]

    # Pesos balanceados (sem suavização sqrt) para dar ênfase à classe Anormal
    classes = np.unique(y)
    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )
    class_weight = {int(cls): float(w) for cls, w in zip(classes, raw_weights)}
    LOGGER.info("Stage 1 class weights: %s", class_weight)

    def model_builder(input_len: int, num_classes: int) -> tf.keras.Model:
        model_cfg = cfg["model"]
        model = build_backbone_1d(
            input_len=input_len,
            num_classes=num_classes,
            embedding_dim=model_cfg.get("embedding_dim", 80),
            conv_filters=model_cfg.get("conv_filters", [16, 40, 80]),
            conv_kernels=model_cfg.get("conv_kernels", [7, 5, 3]),
            dense_units=model_cfg.get("dense_units", 80),
            channels=X.shape[2],
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

    # Configuração de loss (crossentropy ou focal loss)
    loss_cfg = train_cfg.get("loss", "sparse_categorical_crossentropy")
    if loss_cfg == "sparse_categorical_crossentropy":
        loss = "sparse_categorical_crossentropy"
    elif loss_cfg == "focal_loss":
        gamma = float(train_cfg.get("focal_gamma", 2.0))
        alpha = train_cfg.get("focal_alpha")
        if alpha is not None:
            alpha = np.array(alpha, dtype=np.float32)
        loss = SparseCategoricalFocalLoss(gamma=gamma, alpha=alpha)
        LOGGER.info("Using focal loss | gamma=%.2f | alpha=%s", gamma, alpha)
    else:
        raise ValueError(f"Unsupported loss: {loss_cfg}")

    augment_config = cfg.get("augmentation")
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
        augment_config=augment_config,
        loss=loss,
        optimize_thresholds=optimize_thresholds,
        freeze_backbone=args.freeze_backbone,
        tracking_experiment_id=tracking_experiment_id,
        tracking_stage_label="stage1",
    )

    LOGGER.info(
        "Stage 1 complete | mean F1-macro=%.4f ± %.4f | passes QG=%s",
        summary["mean_metrics"]["F1_macro"],
        summary["std_metrics"]["F1_macro"],
        summary["passes_qg5"],
    )

    record_summary_metrics(
        experiment_id=tracking_experiment_id,
        summary=summary,
        stage_label="stage1",
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    lineage_dir = PROJECT_ROOT / "data" / "lineage" / "models"
    lineage_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(cfg["output"]["model_filename"]).stem
    lineage_path = lineage_dir / f"{model_name}.json"
    with lineage_path.open("w", encoding="utf-8") as fh:
        json.dump(lineage, fh, indent=2, ensure_ascii=False)
    LOGGER.info("Lineage saved to %s", lineage_path)

    _copy_best_fold(summary, experiment_dir, args.output_dir, cfg)

    return 0 if summary["passes_qg5"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
