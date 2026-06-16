"""Avaliação AAMI EC57 — Sens, PPV, FPR, F1, MCC.

Conforme ANSI/AAMI EC57:1998 e literatura de arrhythmia classification.
Métricas primárias: F1-macro > 0.85, MCC > 0.80, Acc > 93%.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef

LOGGER = logging.getLogger("lewis.camada04.evaluate")

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]


def evaluate_aami(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Avaliação completa conforme AAMI EC57.

    Parameters
    ----------
    y_true : np.ndarray
        Labels verdadeiros (shape: (n_samples,)).
    y_pred : np.ndarray
        Labels preditos (shape: (n_samples,)).
    class_names : list[str], optional
        Nomes das classes. Default: ["N", "S", "V", "F", "Q"].

    Returns
    -------
    dict
        {
            "per_class": {cls: {"TP", "FN", "FP", "TN", "Se", "PPV", "FPR", "Spe", "F1"}},
            "global": {"Acc", "F1_macro", "MCC", "FPR_global"},
            "confusion_matrix": np.ndarray,
            "passes_qg5": bool,
        }
    """
    if class_names is None:
        class_names = AAMI_CLASSES

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class: Dict[str, Dict[str, float]] = {}
    for i, cls in enumerate(class_names):
        tp = float(cm[i, i])
        fn = float(cm[i, :].sum() - tp)
        fp = float(cm[:, i].sum() - tp)
        tn = float(cm.sum() - tp - fn - fp)

        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * se * ppv / (se + ppv) if (se + ppv) > 0 else 0.0

        per_class[cls] = {
            "TP": int(tp),
            "FN": int(fn),
            "FP": int(fp),
            "TN": int(tn),
            "Se": round(se, 4),
            "PPV": round(ppv, 4),
            "FPR": round(fpr, 4),
            "Spe": round(spe, 4),
            "F1": round(f1, 4),
        }

    # Global metrics
    total_correct = float(np.trace(cm))
    total_samples = float(cm.sum())
    acc = total_correct / total_samples if total_samples > 0 else 0.0

    f1_scores = [per_class[cls]["F1"] for cls in class_names]
    f1_macro = float(np.mean(f1_scores))

    mcc = matthews_corrcoef(y_true, y_pred)

    # FPR global: FP_total / (FP_total + TN_total)
    fp_total = sum(per_class[cls]["FP"] for cls in class_names)
    tn_total = sum(per_class[cls]["TN"] for cls in class_names)
    fpr_global = fp_total / (fp_total + tn_total) if (fp_total + tn_total) > 0 else 0.0

    # QG5 thresholds
    thresholds = {
        "N": {"Se": 0.96},
        "V": {"Se": 0.90},
        "S": {"Se": 0.75},
        "F": {"Se": 0.60},
        "Q": {"Se": 0.70},
    }
    passes_qg5 = (
        acc > 0.93
        and f1_macro > 0.85
        and mcc > 0.80
        and fpr_global < 0.05
        and all(
            per_class[cls]["Se"] >= thresholds[cls]["Se"]
            for cls in class_names
            if cls in thresholds
        )
    )

    result = {
        "per_class": per_class,
        "global": {
            "Acc": round(acc, 4),
            "F1_macro": round(f1_macro, 4),
            "MCC": round(mcc, 4),
            "FPR_global": round(fpr_global, 4),
        },
        "confusion_matrix": cm.tolist(),
        "passes_qg5": passes_qg5,
    }

    LOGGER.info(
        "AAMI Eval | Acc=%.3f | F1-macro=%.3f | MCC=%.3f | FPR=%.3f | QG5=%s",
        acc,
        f1_macro,
        mcc,
        fpr_global,
        passes_qg5,
    )
    for cls in class_names:
        LOGGER.info(
            "  %s | Se=%.3f | PPV=%.3f | FPR=%.3f | F1=%.3f",
            cls,
            per_class[cls]["Se"],
            per_class[cls]["PPV"],
            per_class[cls]["FPR"],
            per_class[cls]["F1"],
        )
    return result


def evaluate_fold(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Avalia um fold de GroupKFold.

    Parameters
    ----------
    model
        Modelo Keras treinado.
    X_test : np.ndarray
        Dados de teste.
    y_test : np.ndarray
        Labels de teste (inteiro).
    class_names : list[str], optional
        Nomes das classes.

    Returns
    -------
    dict
        Resultado de evaluate_aami + "y_pred" array.
    """
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    result = evaluate_aami(y_test, y_pred, class_names=class_names)
    result["y_pred"] = y_pred.tolist()
    return result
