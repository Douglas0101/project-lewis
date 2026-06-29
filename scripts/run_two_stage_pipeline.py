"""Avalia o pipeline integrado de duas etapas (Project-Lewis v2.0)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.two_stage_pipeline import (
    evaluate_two_stage,
    load_stage_model,
    save_report_and_json,
)
from src.tracking.integrations import (
    finish_tracking_experiment,
    record_two_stage_results,
    start_tracking_experiment,
)

LOGGER = logging.getLogger("lewis.camada04.run_two_stage_pipeline")


def _load_threshold(threshold_path: Path) -> float:
    """Carrega threshold do Estágio 1 a partir de JSON."""
    with threshold_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return float(data["threshold"])


def _thresholds_stage1() -> Dict[str, Any]:
    # v2.2: metas realistas para modelo leve inter-paciente.
    return {
        "min_acc": 0.75,
        "min_f1_macro": 0.55,
        "min_mcc": 0.10,
        "max_fpr_global": 0.25,
        "per_class": {
            "N": {"Se": 0.85},
            "Anormal": {"Se": 0.30, "PPV": 0.25},
        },
    }


def _thresholds_stage2() -> Dict[str, Any]:
    # v2.2: metas realistas após augmentation da classe F.
    return {
        "min_acc": 0.60,
        "min_f1_macro": 0.45,
        "min_mcc": 0.25,
        "max_fpr_global": 0.25,
        "per_class": {
            "S": {"F1": 0.55, "Se": 0.50},
            "V": {"F1": 0.70, "Se": 0.60},
            "F": {"F1": 0.15, "Se": 0.10},
        },
    }


def _thresholds_integrated() -> Dict[str, Any]:
    # v2.2: metas realistas para o pipeline completo com hardware alvo.
    return {
        "min_acc": 0.78,
        "min_f1_macro": 0.30,
        "min_mcc": 0.05,
        "max_fpr_global": 0.20,
        "per_class": {
            "N": {"F1": 0.85},
            "S": {"F1": 0.15},
            "V": {"F1": 0.15},
            "F": {"F1": 0.04},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Avaliação do pipeline integrado Estágio 1 + Estágio 2"
    )
    parser.add_argument(
        "--stage1-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "stage1_float32_v2.0.keras",
    )
    parser.add_argument(
        "--stage1-scaler",
        type=Path,
        default=PROJECT_ROOT / "models" / "input_scaler_stage1_v2.0.pkl",
    )
    parser.add_argument(
        "--stage1-threshold",
        type=Path,
        default=PROJECT_ROOT / "models" / "stage1_threshold.json",
    )
    parser.add_argument(
        "--stage2-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "stage2_float32_v2.0.keras",
    )
    parser.add_argument(
        "--stage2-scaler",
        type=Path,
        default=PROJECT_ROOT / "models" / "input_scaler_stage2_v2.0.pkl",
    )
    parser.add_argument(
        "--source-npz",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "finetuning_mitbih_family.npz",
    )
    parser.add_argument(
        "--source-parquet",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "finetuning_mitbih_family.parquet",
    )
    parser.add_argument(
        "--stage1-feature-columns",
        type=str,
        nargs="+",
        default=["rr_prev", "qrs_width_ms"],
        help="Colunas do parquet a empilhar como canais adicionais no Estágio 1",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "two_stage_evaluation_v2.0.md",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "two_stage_evaluation_v2.0.json",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    tracking_experiment_id = start_tracking_experiment(
        name=f"two_stage_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        stage="two_stage",
        description="Avaliação do pipeline integrado Estágio 1 + Estágio 2",
    )

    LOGGER.info("Carregando Estágio 1: %s", args.stage1_model)
    stage1_model, stage1_scaler = load_stage_model(
        args.stage1_model, args.stage1_scaler
    )

    stage1_threshold = 0.5
    if args.stage1_threshold.exists():
        stage1_threshold = _load_threshold(args.stage1_threshold)
    LOGGER.info("Threshold Estágio 1: %.4f", stage1_threshold)

    LOGGER.info("Carregando Estágio 2: %s", args.stage2_model)
    stage2_model, stage2_scaler = load_stage_model(
        args.stage2_model, args.stage2_scaler
    )

    LOGGER.info("Carregando dataset fonte: %s", args.source_npz)
    data = np.load(args.source_npz)
    X = data["X"].astype(np.float32)
    y_aami = data["y"].astype(np.int64)
    if X.ndim == 2:
        X = X[..., np.newaxis]
    LOGGER.info("Fonte | n=%d | classes=%s", len(y_aami), np.unique(y_aami))

    # Constrói entrada do Estágio 1 com canais de features se o modelo esperar
    stage1_input_shape = stage1_model.input_shape
    n_channels_expected = stage1_input_shape[-1]
    X_stage1 = X
    if n_channels_expected != X.shape[-1]:
        LOGGER.info(
            "Estágio 1 espera %d canais; empilhando features %s",
            n_channels_expected,
            args.stage1_feature_columns,
        )
        df_source = pd.read_parquet(args.source_parquet)
        missing = set(args.stage1_feature_columns) - set(df_source.columns)
        if missing:
            raise ValueError(f"Colunas de feature ausentes: {missing}")
        features = df_source[args.stage1_feature_columns].to_numpy(dtype=np.float32)
        features_time = np.repeat(features[:, np.newaxis, :], X.shape[1], axis=1)
        X_stage1 = np.concatenate([X, features_time], axis=2).astype(np.float32)
        if X_stage1.shape[-1] != n_channels_expected:
            raise ValueError(
                f"Shape mismatch: modelo espera {n_channels_expected} canais, "
                f"mas X_stage1 tem {X_stage1.shape[-1]}"
            )

    results = evaluate_two_stage(
        stage1_model=stage1_model,
        stage1_scaler=stage1_scaler,
        stage1_threshold=stage1_threshold,
        stage2_model=stage2_model,
        stage2_scaler=stage2_scaler,
        X=X,
        y_aami=y_aami,
        stage1_thresholds=_thresholds_stage1(),
        stage2_thresholds=_thresholds_stage2(),
        integrated_thresholds=_thresholds_integrated(),
        X_stage1=X_stage1,
    )

    save_report_and_json(results, args.output_report, args.output_json)

    record_two_stage_results(
        experiment_id=tracking_experiment_id,
        results=results,
    )
    finish_tracking_experiment(
        experiment_id=tracking_experiment_id,
        status="completed" if results["integrated"]["passes_qg5"] else "failed",
    )

    LOGGER.info(
        "Pipeline integrado | Acc=%.4f | F1-macro=%.4f | Passa QG=%s",
        results["integrated"]["global"]["Acc"],
        results["integrated"]["global"]["F1_macro"],
        results["integrated"]["passes_qg5"],
    )
    return 0 if results["integrated"]["passes_qg5"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
