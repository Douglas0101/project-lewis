"""Seleciona o melhor fold dos experimentos MLP e publica artefatos.

Critério padrão:
- Estágio 1: maximiza F1-macro no fold de validação (QG5' recall/precision também reportados).
- Estágio 2: maximiza F1-macro no fold de validação, com restrições mínimas
  F1(S) >= 0.55, F1(V) >= 0.70, F1(F) >= 0.50.

Publica em ``models/`` (conforme --target-version):
- ``stage1_float32_v2.4.keras`` + ``input_scaler_stage1_v2.4.pkl``
- ``stage2_float32_v2.4.keras`` + ``input_scaler_stage2_v2.4.pkl``
- ``stage1_threshold_v2.4.json``
- ``stage2_thresholds_v2.4.json``

Para publicar em v2.3 (baseline legado), usar explicitamente:
  --target-version v2.3 --allow-legacy
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("select_best_mlp_fold")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def _load_summary(experiment_dir: Path) -> dict[str, Any]:
    path = experiment_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"summary.json não encontrado em {experiment_dir}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Falha ao ler {path}: {exc}") from exc


def _best_stage1_fold(summary: dict[str, Any]) -> dict[str, Any]:
    folds = summary.get("folds", [])
    if not folds:
        raise ValueError("Nenhum fold encontrado no summary")
    best = max(folds, key=lambda f: f["eval_result"]["global"]["F1_macro"])
    return best


def _best_stage2_fold(summary: dict[str, Any]) -> dict[str, Any]:
    folds = summary.get("folds", [])
    if not folds:
        raise ValueError("Nenhum fold encontrado no summary")

    def _passes(f: dict[str, Any]) -> bool:
        pc = f["eval_result"]["per_class"]
        f1_s = pc.get("S", {}).get("F1", 0.0)
        f1_v = pc.get("V", {}).get("F1", 0.0)
        f1_f = pc.get("F", {}).get("F1", 0.0)
        macro = f["eval_result"]["global"]["F1_macro"]
        return f1_s >= 0.55 and f1_v >= 0.70 and f1_f >= 0.50 and macro >= 0.45

    passing = [f for f in folds if _passes(f)]
    if passing:
        return max(passing, key=lambda f: f["eval_result"]["global"]["F1_macro"])

    # Fallback: score com penalidade se nenhum fold atender todos thresholds.
    def _score(f: dict[str, Any]) -> float:
        pc = f["eval_result"]["per_class"]
        f1_s = pc.get("S", {}).get("F1", 0.0)
        f1_v = pc.get("V", {}).get("F1", 0.0)
        f1_f = pc.get("F", {}).get("F1", 0.0)
        penalty = 0.0
        if f1_s < 0.55:
            penalty += 0.55 - f1_s
        if f1_v < 0.70:
            penalty += 0.70 - f1_v
        if f1_f < 0.50:
            penalty += 0.50 - f1_f
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

    try:
        shutil.copy2(src_model, dst_model_path)
        shutil.copy2(src_scaler, dst_scaler_path)
        LOGGER.info("Publicado %s e %s", dst_model_path, dst_scaler_path)
    except Exception as exc:
        raise RuntimeError(f"Falha ao copiar artefatos: {exc}") from exc


def _compute_stage1_threshold(model_path: Path, scaler_path: Path, feature_npz: Path) -> float:
    """Computa threshold de operação do Estágio 1 via Youden sobre dados de validação.

    Como proxy, usa todo o dataset de features stage1 para encontrar um threshold
    que maximize F1-macro binário. Em produção, o threshold deve ser derivado do
    fold de validação selecionado.
    """
    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
        scaler = joblib.load(scaler_path)
        data = np.load(feature_npz)
        X = scaler.transform(data["X"].astype(np.float32))
        y = data["y"].astype(np.int64)
    except Exception as exc:
        raise RuntimeError(f"Falha ao carregar modelo/dados do Estágio 1: {exc}") from exc

    proba = model.predict(X, batch_size=1024, verbose=0)[:, 1]

    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.linspace(0.1, 0.9, 81):
        try:
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
        except Exception as exc:
            LOGGER.warning("Erro ao avaliar threshold %.2f: %s", thresh, exc)
            continue

    LOGGER.info("Threshold Estágio 1 otimizado: %.4f (F1=%.4f)", best_thresh, best_f1)
    return best_thresh


def _load_or_compute_stage2_thresholds(
    stage2_exp: Path,
    stage2_best: dict[str, Any],
    stage2_features: Path,
) -> dict[str, float]:
    """Carrega thresholds Stage 2 do melhor fold ou re-computa via Youden.

    Preferência é pelo arquivo ``stage2_thresholds.json`` do experimento,
    que já contém a mediana dos folds. Se ausente, usa o melhor fold.
    """
    # 1) Tenta arquivo agregado do experimento
    aggregated_path = stage2_exp / "stage2_thresholds.json"
    if aggregated_path.exists():
        try:
            data = json.loads(aggregated_path.read_text(encoding="utf-8"))
            thresholds = data.get("thresholds")
            if thresholds:
                LOGGER.info("Thresholds Stage 2 carregados do experimento: %s", thresholds)
                return thresholds
        except Exception as exc:
            LOGGER.warning("Erro ao ler thresholds agregados: %s", exc)

    # 2) Tenta arquivo do melhor fold
    fold_thresholds_path = stage2_exp / f"fold_{stage2_best['fold']}" / "stage2_thresholds.json"
    if fold_thresholds_path.exists():
        try:
            data = json.loads(fold_thresholds_path.read_text(encoding="utf-8"))
            thresholds = data.get("thresholds")
            if thresholds:
                LOGGER.info("Thresholds Stage 2 carregados do melhor fold: %s", thresholds)
                return thresholds
        except Exception as exc:
            LOGGER.warning("Erro ao ler thresholds do melhor fold: %s", exc)

    # 3) Fallback: computa Youden no dataset stage2 completo
    LOGGER.warning("Thresholds Stage 2 não encontrados; computando Youden no dataset completo")
    from src.models.evaluate import find_best_thresholds_youden

    try:
        model_path = stage2_exp / f"fold_{stage2_best['fold']}" / "model.keras"
        scaler_path = stage2_exp / f"fold_{stage2_best['fold']}" / "input_scaler.pkl"
        model = tf.keras.models.load_model(str(model_path), compile=False)
        scaler = joblib.load(scaler_path)
        data = np.load(stage2_features)
        X = scaler.transform(data["X"].astype(np.float32))
        y = data["y"].astype(np.int64)
        proba = model.predict(X, batch_size=1024, verbose=0)
        result = find_best_thresholds_youden(y, proba, class_names=["S", "V", "F"])
        return result["thresholds"]
    except Exception as exc:
        raise RuntimeError(f"Falha ao computar thresholds Stage 2: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Seleciona e publica melhor fold MLP")
    parser.add_argument(
        "--target-version",
        type=str,
        choices=["v2.3", "v2.4"],
        required=True,
        help="Versão alvo de publicação. Use v2.4 para novos experimentos de research.",
    )
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Permite publicar em paths v2.3 (apenas para manutenção do baseline legado).",
    )
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
        "--stage2-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
    )
    args = parser.parse_args()

    if args.target_version == "v2.3" and not args.allow_legacy:
        raise ValueError(
            "Publicação em paths v2.3 requer --allow-legacy. "
            "Use --target-version v2.4 para novos experimentos."
        )

    version = args.target_version.lstrip("v")
    try:
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
            args.output_dir / f"stage1_float32_{version}.keras",
            args.output_dir / f"input_scaler_stage1_{version}.pkl",
        )
        _copy_artifacts(
            args.stage2_exp / f"fold_{stage2_best['fold']}",
            args.output_dir / f"stage2_float32_{version}.keras",
            args.output_dir / f"input_scaler_stage2_{version}.pkl",
        )

        threshold = _compute_stage1_threshold(
            args.output_dir / f"stage1_float32_{version}.keras",
            args.output_dir / f"input_scaler_stage1_{version}.pkl",
            args.stage1_features,
        )
        threshold_path = args.output_dir / f"stage1_threshold_{version}.json"
        threshold_path.write_text(
            json.dumps({"threshold": threshold, "source": "youden_f1_macro_stage1"}, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Threshold salvo em %s", threshold_path)

        stage2_thresholds = _load_or_compute_stage2_thresholds(
            args.stage2_exp,
            stage2_best,
            args.stage2_features,
        )
        stage2_threshold_path = args.output_dir / f"stage2_thresholds_{version}.json"
        stage2_threshold_path.write_text(
            json.dumps(
                {"thresholds": stage2_thresholds, "source": "youden_median_across_folds"},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        LOGGER.info("Thresholds Stage 2 salvos em %s", stage2_threshold_path)
    except Exception as exc:
        LOGGER.error("Falha na publicação de artefatos: %s", exc)
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
