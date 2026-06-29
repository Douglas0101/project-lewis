"""Pipeline integrado de duas etapas: Estágio 1 (N vs Anormal) + Estágio 2 (S/V/F).

Avaliação conforme QG5' v2.0 — 4 classes finais (N, S, V, F), excluindo Q.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import tensorflow as tf

from .evaluate import evaluate_aami

LOGGER = logging.getLogger("lewis.camada04.two_stage_pipeline")

AAMI_INTEGRATED_CLASSES = ["N", "S", "V", "F"]
STAGE2_TO_AAMI = {0: "S", 1: "V", 2: "F"}
AAMI_TO_IDX = {cls: i for i, cls in enumerate(AAMI_INTEGRATED_CLASSES)}


def load_stage_model(model_path: Path, scaler_path: Path) -> Tuple[tf.keras.Model, Any]:
    """Carrega modelo Keras e scaler de um estágio.

    Parameters
    ----------
    model_path : Path
        Caminho para o modelo ``.keras``.
    scaler_path : Path
        Caminho para o scaler ``.pkl``.

    Returns
    -------
    tuple
        (modelo, scaler)
    """
    model = tf.keras.models.load_model(str(model_path), compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler


def _normalize(X: np.ndarray, scaler: Any) -> np.ndarray:
    """Aplica z-score global usando o scaler fornecido."""
    n, seq_len, channels = X.shape
    return scaler.transform(X.reshape(-1, channels)).reshape(n, seq_len, channels)


def run_stage1(
    model: tf.keras.Model,
    scaler: Any,
    X: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Executa Estágio 1 e retorna predições binárias + probabilidade Anormal.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo binário N vs Anormal.
    scaler : Any
        Scaler do Estágio 1.
    X : np.ndarray
        Dados brutos (n, 500, 1).
    threshold : float
        Limiar para classe Anormal (padrão 0.5).

    Returns
    -------
    tuple
        (y_pred_binary, y_score_anormal)
    """
    X_norm = _normalize(X, scaler)
    proba = model.predict(X_norm, verbose=0)
    score_anormal = proba[:, 1]
    y_pred = (score_anormal >= threshold).astype(np.int64)
    return y_pred, score_anormal


def run_stage2(
    model: tf.keras.Model,
    scaler: Any,
    X: np.ndarray,
) -> np.ndarray:
    """Executa Estágio 2 sobre amostras classificadas como Anormal.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo S vs V vs F.
    scaler : Any
        Scaler do Estágio 2.
    X : np.ndarray
        Dados brutos (n, 500, 1).

    Returns
    -------
    np.ndarray
        Labels preditos 0=S, 1=V, 2=F.
    """
    X_norm = _normalize(X, scaler)
    proba = model.predict(X_norm, verbose=0)
    return np.argmax(proba, axis=1).astype(np.int64)


def build_integrated_predictions(
    stage1_pred: np.ndarray,
    stage2_pred: np.ndarray,
) -> np.ndarray:
    """Combina predições: N quando Estágio 1 = 0, senão classe do Estágio 2.

    Parameters
    ----------
    stage1_pred : np.ndarray
        Predições binárias (0=N, 1=Anormal).
    stage2_pred : np.ndarray
        Predições S/V/F (0=S, 1=V, 2=F).

    Returns
    -------
    np.ndarray
        Labels integrados no espaço AAMI (N=0, S=1, V=2, F=3).
    """
    integrated = np.zeros_like(stage1_pred)
    abnormal_mask = stage1_pred == 1
    n_abnormal = int(abnormal_mask.sum())
    if n_abnormal > 0:
        if len(stage2_pred) != n_abnormal:
            raise ValueError(
                f"stage2_pred deve ter {n_abnormal} amostras (Anormal), "
                f"mas tem {len(stage2_pred)}"
            )
        integrated[abnormal_mask] = stage2_pred + 1  # S=1, V=2, F=3
    return integrated


def evaluate_two_stage(
    stage1_model: tf.keras.Model,
    stage1_scaler: Any,
    stage1_threshold: float,
    stage2_model: tf.keras.Model,
    stage2_scaler: Any,
    X: np.ndarray,
    y_aami: np.ndarray,
    stage1_thresholds: Optional[Dict[str, Any]] = None,
    stage2_thresholds: Optional[Dict[str, Any]] = None,
    integrated_thresholds: Optional[Dict[str, Any]] = None,
    X_stage1: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Avalia o pipeline integrado no dataset completo.

    Parameters
    ----------
    stage1_model : tf.keras.Model
    stage1_scaler : Any
    stage1_threshold : float
    stage2_model : tf.keras.Model
    stage2_scaler : Any
    X : np.ndarray
        Dados brutos (n, 500, 1) usados pelo Estágio 2.
    y_aami : np.ndarray
        Labels AAMI originais (0=N, 1=S, 2=V, 3=F, 4=Q).
    stage1_thresholds, stage2_thresholds, integrated_thresholds : dict, optional
        Thresholds para ``evaluate_aami``.
    X_stage1 : np.ndarray, optional
        Dados do Estágio 1, possivelmente com canais de features adicionais.
        Se None, reutiliza ``X``.

    Returns
    -------
    dict
        Resultados de cada estágio e do pipeline integrado.
    """
    X_stage1 = X_stage1 if X_stage1 is not None else X

    # Estágio 1 (binário) — inclui Q como Anormal (ou S/V/F se exclude_q)
    y_stage1_pred, _ = run_stage1(
        stage1_model, stage1_scaler, X_stage1, threshold=stage1_threshold
    )
    y_stage1_true = np.where(y_aami == 0, 0, 1)
    stage1_result = evaluate_aami(
        y_stage1_true,
        y_stage1_pred,
        class_names=["N", "Anormal"],
        thresholds=stage1_thresholds,
    )

    # Estágio 2 — avaliado apenas no subconjunto verdadeiramente S/V/F
    stage2_mask = np.isin(y_aami, [1, 2, 3])
    y_stage2_true = y_aami[stage2_mask] - 1  # S=0, V=1, F=2
    X_stage2 = X[stage2_mask]
    y_stage2_pred = run_stage2(stage2_model, stage2_scaler, X_stage2)
    stage2_result = evaluate_aami(
        y_stage2_true,
        y_stage2_pred,
        class_names=["S", "V", "F"],
        thresholds=stage2_thresholds,
    )

    # Pipeline integrado — 4 classes (N, S, V, F), excluindo Q
    integrated_mask = y_aami != 4
    y_integrated_true = y_aami[integrated_mask]

    # Estágio 2 deve ser executado apenas sobre as amostras que o Estágio 1
    # classificou como Anormal dentro do conjunto integrado (sem Q).
    abnormal_integrated_mask = integrated_mask & (y_stage1_pred == 1)
    X_stage2_integrated = X[abnormal_integrated_mask]
    y_stage2_pred_integrated = (
        run_stage2(stage2_model, stage2_scaler, X_stage2_integrated)
        if X_stage2_integrated.shape[0] > 0
        else np.array([], dtype=np.int64)
    )
    y_integrated_pred = build_integrated_predictions(
        y_stage1_pred[integrated_mask],
        y_stage2_pred_integrated,
    )
    integrated_result = evaluate_aami(
        y_integrated_true,
        y_integrated_pred,
        class_names=AAMI_INTEGRATED_CLASSES,
        thresholds=integrated_thresholds,
    )

    return {
        "stage1": stage1_result,
        "stage2": stage2_result,
        "integrated": integrated_result,
    }


def generate_report(results: Dict[str, Any]) -> str:
    """Gera relatório Markdown com os resultados do pipeline."""
    lines: List[str] = []
    lines.append("# Pipeline Integrado v2.0 — Avaliação QG5'\n")
    lines.append("## Estágio 1 (N vs Anormal)\n")
    lines.append(f"- Acc: {results['stage1']['global']['Acc']:.4f}\n")
    lines.append(f"- F1-macro: {results['stage1']['global']['F1_macro']:.4f}\n")
    lines.append(f"- MCC: {results['stage1']['global']['MCC']:.4f}\n")
    lines.append(f"- Passa QG: {results['stage1']['passes_qg5']}\n")
    lines.append("\n### Por classe\n")
    lines.append("| Classe | Se | PPV | F1 |\n")
    lines.append("|--------|----|-----|----|\n")
    for cls, m in results["stage1"]["per_class"].items():
        lines.append(f"| {cls} | {m['Se']:.4f} | {m['PPV']:.4f} | {m['F1']:.4f} |\n")

    lines.append("\n## Estágio 2 (S vs V vs F)\n")
    lines.append(f"- Acc: {results['stage2']['global']['Acc']:.4f}\n")
    lines.append(f"- F1-macro: {results['stage2']['global']['F1_macro']:.4f}\n")
    lines.append(f"- MCC: {results['stage2']['global']['MCC']:.4f}\n")
    lines.append(f"- Passa QG: {results['stage2']['passes_qg5']}\n")
    lines.append("\n### Por classe\n")
    lines.append("| Classe | Se | PPV | F1 |\n")
    lines.append("|--------|----|-----|----|\n")
    for cls, m in results["stage2"]["per_class"].items():
        lines.append(f"| {cls} | {m['Se']:.4f} | {m['PPV']:.4f} | {m['F1']:.4f} |\n")

    lines.append("\n## Pipeline Integrado (N, S, V, F)\n")
    lines.append(f"- Acc: {results['integrated']['global']['Acc']:.4f}\n")
    lines.append(f"- F1-macro: {results['integrated']['global']['F1_macro']:.4f}\n")
    lines.append(f"- MCC: {results['integrated']['global']['MCC']:.4f}\n")
    lines.append(f"- FPR global: {results['integrated']['global']['FPR_global']:.4f}\n")
    lines.append(f"- Passa QG: {results['integrated']['passes_qg5']}\n")
    lines.append("\n### Por classe\n")
    lines.append("| Classe | Se | PPV | Spe | F1 |\n")
    lines.append("|--------|----|-----|-----|----|\n")
    for cls, m in results["integrated"]["per_class"].items():
        lines.append(
            f"| {cls} | {m['Se']:.4f} | {m['PPV']:.4f} | {m['Spe']:.4f} | {m['F1']:.4f} |\n"
        )

    return "".join(lines)


def save_report_and_json(
    results: Dict[str, Any],
    report_path: Path,
    json_path: Path,
) -> None:
    """Salva relatório Markdown e JSON."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    report = generate_report(results)
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write(report)
    LOGGER.info("Relatório salvo em %s", report_path)

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    LOGGER.info("Resultados JSON salvos em %s", json_path)
