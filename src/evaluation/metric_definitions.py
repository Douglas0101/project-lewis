"""Definições de métricas do avaliador canônico (ML Protocol v2, evaluator v2.0).

Fonte única de verdade para métricas equalizadas (docs/ml_protocol_v2.md §3):

- macro-médias NÃO ponderadas por suporte (suporte sempre reportado ao lado);
- PR-AUC é a métrica preferencial sob desbalanceio; AUROC para comparabilidade;
- F1 é sempre reportado em threshold fixo 0.5 E em thresholds tunados;
- BCE multi-label ≡ NLL binário médio por elemento (paridade com o legado
  ``nll_multilabel`` de ``src/models/pretrain_evaluation.py``).

Somente numpy/sklearn — nenhum import de TensorFlow neste módulo.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Registro navegável das métricas do protocolo (espelha docs/ml_protocol_v2.md §3).
METRIC_REGISTRY: dict[str, dict[str, str]] = {
    # primárias (pré-treino SCP-ECG multi-label)
    "macro_pr_auc": {"kind": "primary", "direction": "max"},
    "macro_auroc": {"kind": "primary", "direction": "max"},
    "ece_post_calibration": {"kind": "primary", "direction": "min"},
    "brier_mean": {"kind": "primary", "direction": "min"},
    "delta_quantization_macro_pr_auc": {"kind": "primary", "direction": "min"},
    # secundárias
    "per_class_pr_auc": {"kind": "secondary", "direction": "max"},
    "per_class_auroc": {"kind": "secondary", "direction": "max"},
    "per_class_f1": {"kind": "secondary", "direction": "max"},
    "per_class_precision": {"kind": "secondary", "direction": "max"},
    "per_class_recall": {"kind": "secondary", "direction": "max"},
    "macro_f1_at_0.5": {"kind": "secondary", "direction": "max"},
    "macro_f1_tuned": {"kind": "secondary", "direction": "max"},
    "bce": {"kind": "secondary", "direction": "min"},
    "bce_post_temperature": {"kind": "secondary", "direction": "min"},
    "nll": {"kind": "secondary", "direction": "min"},
    "nll_post_temperature": {"kind": "secondary", "direction": "min"},
    "mce": {"kind": "secondary", "direction": "min"},
    "rejection_rate": {"kind": "secondary", "direction": "min"},
    # guarda (edge/firmware)
    "model_size_int8": {"kind": "guard", "direction": "min"},
    "latency_renode": {"kind": "guard", "direction": "min"},
    "sram_total": {"kind": "guard", "direction": "min"},
    "arena_used": {"kind": "guard", "direction": "min"},
    "bitexact_atol_1_lsb": {"kind": "guard", "direction": "max"},
    "cosine_fidelity": {"kind": "guard", "direction": "max"},
    "saturation_int8": {"kind": "guard", "direction": "min"},
    "sha256_provenance": {"kind": "guard", "direction": "na"},
}

_EPS = 1e-12


def per_class_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Sequence[str],
    threshold: float = 0.5,
) -> dict:
    """Métricas por classe: AUROC, PR-AUC, P/R/F1 em ``threshold`` e suporte.

    Numericamente idêntico a ``compute_per_class_metrics`` do legado quando
    ``class_names == SCP_SUPERCLASSES`` e ``threshold == 0.5`` (mesmas funções
    sklearn, mesmos defaults) — a paridade é por construção, não por acordo.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape or y_true.ndim != 2:
        raise ValueError(
            "per_class_metrics espera y_true e y_score 2D com o mesmo shape; "
            f"recebido {y_true.shape} vs {y_score.shape}"
        )
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class: dict[str, dict] = {}
    for idx, name in enumerate(class_names):
        yt, ys = y_true[:, idx], y_score[:, idx]
        n_pos = int(yt.sum())
        per_class[name] = {
            "support": n_pos,
            "auc_roc": float(roc_auc_score(yt, ys)) if 0 < n_pos < len(yt) else None,
            "auc_pr": float(average_precision_score(yt, ys)) if n_pos > 0 else None,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
        }
    return {"threshold": float(threshold), "per_class": per_class}


def _macro(values: list[Optional[float]]) -> Optional[float]:
    """Média não ponderada ignorando None; None se não houver valor algum."""
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def macro_auroc(metrics_per_class: dict) -> Optional[float]:
    """Macro AUROC (média não ponderada das AUROC por classe)."""
    return _macro([m["auc_roc"] for m in metrics_per_class["per_class"].values()])


def macro_pr_auc(metrics_per_class: dict) -> Optional[float]:
    """Macro PR-AUC (média não ponderada das PR-AUC por classe)."""
    return _macro([m["auc_pr"] for m in metrics_per_class["per_class"].values()])


def f1_at_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: dict[str, float],
    class_names: Sequence[str],
) -> dict:
    """P/R/F1 por classe com thresholds individuais + macro-F1 (não ponderado)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    per_class: dict[str, dict] = {}
    f1s: list[float] = []
    for idx, name in enumerate(class_names):
        yt = y_true[:, idx]
        yp = (y_score[:, idx] >= thresholds[name]).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            yt, yp, average="binary", zero_division=0
        )
        per_class[name] = {
            "threshold": float(thresholds[name]),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        f1s.append(float(f1))
    return {
        "per_class": per_class,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
    }


def bce_multilabel(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """BCE médio por elemento (≡ NLL binário multi-label; paridade com o legado)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), _EPS, 1.0 - _EPS)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def nll_softmax(y_true: np.ndarray, proba: np.ndarray) -> float:
    """NLL multiclasse (softmax): média de -log p da classe verdadeira."""
    proba = np.clip(np.asarray(proba, dtype=np.float64), _EPS, 1.0)
    y_true = np.asarray(y_true, dtype=np.int64)
    return float(-np.mean(np.log(proba[np.arange(len(y_true)), y_true])))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]) -> dict:
    """TP/FP/TN/FN por classe (multi-label, decisões já binarizadas)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[str, dict] = {}
    for idx, name in enumerate(class_names):
        yt, yp = y_true[:, idx], y_pred[:, idx]
        out[name] = {
            "tp": int(((yt == 1) & (yp == 1)).sum()),
            "fp": int(((yt == 0) & (yp == 1)).sum()),
            "tn": int(((yt == 0) & (yp == 0)).sum()),
            "fn": int(((yt == 1) & (yp == 0)).sum()),
        }
    return out
