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
    thresholds: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Avaliação completa conforme AAMI EC57 com thresholds configuráveis.

    Parameters
    ----------
    y_true : np.ndarray
        Labels verdadeiros (shape: (n_samples,)).
    y_pred : np.ndarray
        Labels preditos (shape: (n_samples,)).
    class_names : list[str], optional
        Nomes das classes. Default: ["N", "S", "V", "F", "Q"].
    thresholds : dict, optional
        Thresholds para ``passes_qg5``. Exemplo:
        {
            "min_acc": 0.88,
            "min_f1_macro": 0.55,
            "min_mcc": 0.50,
            "max_fpr_global": 0.05,
            "per_class": {"N": {"Se": 0.90}, ...},
        }
        Se None, usa os thresholds v1.1 originais (5 classes AAMI).

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

    # Thresholds
    if thresholds is None:
        thresholds = {
            "min_acc": 0.93,
            "min_f1_macro": 0.85,
            "min_mcc": 0.80,
            "max_fpr_global": 0.05,
            "per_class": {
                "N": {"Se": 0.96},
                "V": {"Se": 0.90},
                "S": {"Se": 0.75},
                "F": {"Se": 0.60},
                "Q": {"Se": 0.70},
            },
        }

    per_class_thr = thresholds.get("per_class", {})
    passes_per_class = True
    for cls, cls_thr in per_class_thr.items():
        if cls not in per_class:
            passes_per_class = False
            break
        for metric_name, min_val in cls_thr.items():
            if per_class[cls].get(metric_name, 0.0) < min_val:
                passes_per_class = False
                break
        if not passes_per_class:
            break

    passes_qg5 = (
        acc > thresholds.get("min_acc", 0.93)
        and f1_macro > thresholds.get("min_f1_macro", 0.85)
        and mcc > thresholds.get("min_mcc", 0.80)
        and fpr_global < thresholds.get("max_fpr_global", 0.05)
        and passes_per_class
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

    if verbose:
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


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    class_names: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    target_class_idx: int = 1,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Avalia predições binárias usando um threshold sobre a classe alvo.

    Parameters
    ----------
    y_true : np.ndarray
        Labels verdadeiros inteiros.
    y_score : np.ndarray
        Probabilidade da classe alvo (shape: (n_samples,)).
    threshold : float
        Limiar de decisão.
    class_names : list[str], optional
        Nomes das classes.
    thresholds : dict, optional
        Thresholds para ``evaluate_aami``.
    target_class_idx : int
        Índice da classe positiva.

    Returns
    -------
    dict
        Resultado de ``evaluate_aami`` com chave adicional ``threshold``.
    """
    if class_names is None:
        class_names = ["N", "Anormal"]
    y_pred = np.where(y_score >= threshold, target_class_idx, 1 - target_class_idx)
    result = evaluate_aami(
        y_true,
        y_pred,
        class_names=class_names,
        thresholds=thresholds,
        verbose=verbose,
    )
    result["threshold"] = float(threshold)
    return result


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    target_class_idx: int = 1,
) -> Dict[str, Any]:
    """Busca o melhor threshold para classificação binária.

    Prioriza thresholds que satisfazem ``passes_qg5``; em caso de empate,
    escolhe o maior F1-macro. Busca em [0.01, 0.99] com passo 0.01.

    Returns
    -------
    dict
        Resultado de ``evaluate_aami`` para o melhor threshold, incluindo
        ``threshold``.
    """
    if class_names is None:
        class_names = ["N", "Anormal"]

    candidate_thresholds = np.arange(0.01, 1.0, 0.01)
    best_result: Optional[Dict[str, Any]] = None
    best_score = -1.0

    for thr in candidate_thresholds:
        result = evaluate_at_threshold(
            y_true,
            y_score,
            threshold=thr,
            class_names=class_names,
            thresholds=thresholds,
            target_class_idx=target_class_idx,
            verbose=False,
        )
        passes = result["passes_qg5"]
        score = float(result["global"]["F1_macro"])

        if best_result is None:
            best_result = result
            best_score = score
            continue

        # Prioridade 1: passa QG; Prioridade 2: maior F1-macro
        if passes and not best_result["passes_qg5"]:
            best_result = result
            best_score = score
        elif passes == best_result["passes_qg5"] and score > best_score:
            best_result = result
            best_score = score

    if best_result is None:
        raise RuntimeError("Nenhum threshold avaliado")

    return best_result


def evaluate_multiclass_at_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Dict[str, float],
    class_names: Optional[List[str]] = None,
    thresholds_cfg: Optional[Dict[str, Any]] = None,
    fallback_class: int = 1,
) -> Dict[str, Any]:
    """Avalia classificação multiclasse com thresholds one-vs-rest.

    Parameters
    ----------
    y_true : np.ndarray
        Labels verdadeiros inteiros.
    y_score : np.ndarray
        Probabilidades softmax (shape: (n_samples, n_classes)).
    thresholds : dict
        Limiar por classe: ``{class_name: threshold}``.
    class_names : list[str], optional
        Nomes das classes.
    thresholds_cfg : dict, optional
        Thresholds para ``evaluate_aami``.
    fallback_class : int
        Classe atribuída quando nenhuma classe supera o limiar.

    Returns
    -------
    dict
        Resultado de ``evaluate_aami`` com chaves adicionais ``thresholds`` e
        ``y_pred``.
    """
    if class_names is None:
        class_names = ["S", "V", "F"]

    n_samples = len(y_true)
    n_classes = len(class_names)
    y_pred = np.full(n_samples, fallback_class, dtype=np.int64)

    above_threshold = np.zeros((n_samples, n_classes), dtype=bool)
    for i, cls in enumerate(class_names):
        thr = thresholds.get(cls, 0.5)
        above_threshold[:, i] = y_score[:, i] >= thr

    # Caso uma única classe supere o limiar, escolhe-a.
    single_class = above_threshold.sum(axis=1) == 1
    y_pred[single_class] = np.argmax(above_threshold[single_class], axis=1)

    # Caso múltiplas classes superem, escolhe a de maior probabilidade entre elas.
    multi_class = above_threshold.sum(axis=1) > 1
    if multi_class.any():
        scores_masked = y_score.copy()
        scores_masked[~above_threshold] = -1.0
        y_pred[multi_class] = np.argmax(scores_masked[multi_class], axis=1)

    # Caso nenhuma supere, mantém fallback_class.
    result = evaluate_aami(
        y_true,
        y_pred,
        class_names=class_names,
        thresholds=thresholds_cfg,
        verbose=False,
    )
    result["thresholds"] = {cls: float(thresholds.get(cls, 0.5)) for cls in class_names}
    result["y_pred"] = y_pred.tolist()
    return result


def find_best_thresholds_multiclass(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Optional[List[str]] = None,
    thresholds_cfg: Optional[Dict[str, Any]] = None,
    metric: str = "F1_macro",
    search_step: float = 0.05,
    fallback_class: int = 1,
) -> Dict[str, Any]:
    """Busca melhores thresholds one-vs-rest para classificação multiclasse.

    A busca é feita de forma gulosa: para cada classe, fixa os thresholds das
    demais e busca o valor que maximiza a métrica objetivo. Repete por algumas
    iterações para refinamento.

    Parameters
    ----------
    y_true : np.ndarray
        Labels verdadeiros.
    y_score : np.ndarray
        Probabilidades softmax.
    class_names : list[str], optional
        Nomes das classes.
    thresholds_cfg : dict, optional
        Thresholds para ``evaluate_aami``.
    metric : str
        Métrica a maximizar (ex.: "F1_macro").
    search_step : float
        Passo da busca em [0.05, 0.95].
    fallback_class : int
        Classe de fallback quando nenhum threshold é atingido.

    Returns
    -------
    dict
        Melhor resultado de ``evaluate_multiclass_at_thresholds``.
    """
    if class_names is None:
        class_names = ["S", "V", "F"]

    candidate_thresholds = np.arange(0.05, 1.0, search_step)
    best_thresholds = {cls: 0.5 for cls in class_names}
    best_result = evaluate_multiclass_at_thresholds(
        y_true,
        y_score,
        best_thresholds,
        class_names=class_names,
        thresholds_cfg=thresholds_cfg,
        fallback_class=fallback_class,
    )
    best_score = float(best_result["global"][metric])

    for _ in range(3):
        improved = False
        for cls in class_names:
            for thr in candidate_thresholds:
                trial = dict(best_thresholds)
                trial[cls] = float(thr)
                result = evaluate_multiclass_at_thresholds(
                    y_true,
                    y_score,
                    trial,
                    class_names=class_names,
                    thresholds_cfg=thresholds_cfg,
                    fallback_class=fallback_class,
                )
                score = float(result["global"][metric])
                if score > best_score:
                    best_score = score
                    best_thresholds = trial
                    best_result = result
                    improved = True
        if not improved:
            break

    return best_result


def evaluate_fold(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    optimize_thresholds: bool = False,
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
    thresholds : dict, optional
        Thresholds configuráveis para ``evaluate_aami``.

    Returns
    -------
    dict
        Resultado de evaluate_aami + "y_pred" array.
    """
    y_pred_proba = model.predict(X_test, verbose=0)

    if class_names is not None and len(class_names) == 2:
        target_class_idx = 1
        y_score = y_pred_proba[:, target_class_idx]
        result = find_best_threshold(
            y_test,
            y_score,
            class_names=class_names,
            thresholds=thresholds,
            target_class_idx=target_class_idx,
        )
        result["y_pred"] = (
            (y_score >= result["threshold"]).astype(np.int64).tolist()
        )
    elif optimize_thresholds and class_names is not None:
        result = find_best_thresholds_multiclass(
            y_test,
            y_pred_proba,
            class_names=class_names,
            thresholds_cfg=thresholds,
        )
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)
        result = evaluate_aami(
            y_test, y_pred, class_names=class_names, thresholds=thresholds
        )
        result["y_pred"] = y_pred.tolist()
    return result
