"""Seleciona o melhor fold dos experimentos MLP v2.3 e publica artefatos.

Critério padrão:
- Estágio 1: maximiza F1-macro no fold de validação (QG5' recall/precision também reportados).
- Estágio 2: maximiza F1-macro no fold de validação, com restrições mínimas
  F1(S) >= 0.55, F1(V) >= 0.70, F1(F) >= 0.15.

Publica em ``models/``:
- ``stage1_float32_v2.3.keras`` + ``input_scaler_stage1_v2.3.pkl``
- ``stage2_float32_v2.3.keras`` + ``input_scaler_stage2_v2.3.pkl``
- ``stage1_threshold_v2.3.json``
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("select_best_mlp_fold")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def _load_summary(experiment_dir: Path) -> Dict[str, Any]:
    path = experiment_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"summary.json não encontrado em {experiment_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _best_stage1_fold(summary: Dict[str, Any]) -> Dict[str, Any]:
    folds = summary.get("folds", [])
    if not folds:
        raise ValueError("Nenhum fold encontrado no summary")
    best = max(folds, key=lambda f: f["eval_result"]["global"]["F1_macro"])
    return best


def _best_stage2_fold(summary: Dict[str, Any]) -> Dict[str, Any]:
    folds = summary.get("folds", [])
    if not folds:
        raise ValueError("Nenhum fold encontrado no summary")

    def _passes(f: Dict[str, Any]) -> bool:
        pc = f["eval_result"]["per_class"]
        f1_s = pc.get("S", {}).get("F1", 0.0)
        f1_v = pc.get("V", {}).get("F1", 0.0)
        f1_f = pc.get("F", {}).get("F1", 0.0)
        macro = f["eval_result"]["global"]["F1_macro"]
        return (
            f1_s >= 0.55
            and f1_v >= 0.70
            and f1_f >= 0.15
            and macro >= 0.45
        )

    passing = [f for f in folds if _passes(f)]
    if passing:
        return max(passing, key=lambda f: f["eval_result"]["global"]["F1_macro"])

    # Fallback: score com penalidade se nenhum fold atender todos thresholds.
    def _score(f: Dict[str, Any]) -> float:
        pc = f["eval_result"]["per_class"]
        f1_s = pc.get("S", {}).get("F1", 0.0)
        f1_v = pc.get("V", {}).get("F1", 0.0)
        f1_f = pc.get("F", {}).get("F1", 0.0)
        penalty = 0.0
        if f1_s < 0.55:
            penalty += (0.55 - f1_s)
        if f1_v < 0.70:
            penalty += (0.70 - f1_v)
        if f1_f < 0.15:
            penalty += (0.15 - f1_f)
        macro = f["eval_result"]["global"]["F1_macro"]
        return macro - 2.0 * penalty

    return max(folds, key=_score)


def _copy_artifacts(
    src_fold_dir: Path,
    dst_model_path: Path,
    dst_scaler_path: Path,
) -> None:
    src_model = src_fold_dir / "model.keras"
    src_scaler = src_fold_dir / "input_scaler.pkl"
    if not src_model.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {src_model}")
    if not src_scaler.exists():
        raise FileNotFoundError(f"Scaler não encontrado: {src_scaler}")

    shutil.copy2(src_model, dst_model_path)
    shutil.copy2(src_scaler, dst_scaler_path)
    LOGGER.info("Publicado %s e %s", dst_model_path, dst_scaler_path)


def _compute_stage1_threshold(model_path: Path, scaler_path: Path, feature_npz: Path) -> float:
    """Computa threshold de operação do Estágio 1 via Youden sobre dados de validação.

    Como proxy, usa todo o dataset de features stage1 para encontrar um threshold
    que maximize F1-macro binário. Em produção, o threshold deve ser derivado do
    fold de validação selecionado.
    """
    model = tf.keras.models.load_model(str(model_path), compile=False)
    scaler = joblib.load(scaler_path)
    data = np.load(feature_npz)
    X = scaler.transform(data["X"].astype(np.float32))
    y = data["y"].astype(np.int64)

    proba = model.predict(X, batch_size=1024, verbose=0)[:, 1]

    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.linspace(0.1, 0.9, 81):
        pred = (proba >= thresh).astype(np.int64)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    LOGGER.info("Threshold Estágio 1 otimizado: %.4f (F1=%.4f)", best_thresh, best_f1)
    return best_thresh


def main() -> int:
    parser = argparse.ArgumentParser(description="Seleciona e publica melhor fold MLP v2.3")
    parser.add_argument(
        "--stage1-exp",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "stage1_mlp_features_v2.3",
    )
    parser.add_argument(
        "--stage2-exp",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "stage2_mlp_features_v2.3",
    )
    parser.add_argument(
        "--stage1-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "stage1_binary_features.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage1_summary = _load_summary(args.stage1_exp)
    stage2_summary = _load_summary(args.stage2_exp)

    stage1_best = _best_stage1_fold(stage1_summary)
    stage2_best = _best_stage2_fold(stage2_summary)

    LOGGER.info(
        "Melhor fold Stage1: fold=%d | F1_macro=%.4f | AUC=%.4f",
        stage1_best["fold"],
        stage1_best["eval_result"]["global"]["F1_macro"],
        stage1_best.get("auc", 0.0),
    )
    LOGGER.info(
        "Melhor fold Stage2: fold=%d | F1_macro=%.4f | F1(S)=%.4f | F1(V)=%.4f | F1(F)=%.4f",
        stage2_best["fold"],
        stage2_best["eval_result"]["global"]["F1_macro"],
        stage2_best["eval_result"]["per_class"]["S"]["F1"],
        stage2_best["eval_result"]["per_class"]["V"]["F1"],
        stage2_best["eval_result"]["per_class"]["F"]["F1"],
    )

    _copy_artifacts(
        args.stage1_exp / f"fold_{stage1_best['fold']}",
        args.output_dir / "stage1_float32_v2.3.keras",
        args.output_dir / "input_scaler_stage1_v2.3.pkl",
    )
    _copy_artifacts(
        args.stage2_exp / f"fold_{stage2_best['fold']}",
        args.output_dir / "stage2_float32_v2.3.keras",
        args.output_dir / "input_scaler_stage2_v2.3.pkl",
    )

    stage2_threshold_src = args.stage2_exp / f"fold_{stage2_best['fold']}" / "stage2_threshold.json"
    stage2_threshold_dst = args.output_dir / "stage2_threshold_v2.3.json"
    if stage2_threshold_src.exists():
        shutil.copy2(stage2_threshold_src, stage2_threshold_dst)
        LOGGER.info("Publicado %s", stage2_threshold_dst)
    else:
        LOGGER.warning("Threshold Stage 2 não encontrado em %s", stage2_threshold_src)

    threshold = _compute_stage1_threshold(
        args.output_dir / "stage1_float32_v2.3.keras",
        args.output_dir / "input_scaler_stage1_v2.3.pkl",
        args.stage1_features,
    )
    threshold_path = args.output_dir / "stage1_threshold_v2.3.json"
    threshold_path.write_text(
        json.dumps({"threshold": threshold, "source": "youden_f1_macro_stage1"}, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Threshold salvo em %s", threshold_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
