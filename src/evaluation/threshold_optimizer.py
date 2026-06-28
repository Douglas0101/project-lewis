"""Otimização e aplicação de thresholds dinâmicos por classe."""

from typing import Dict

import numpy as np
from sklearn.metrics import precision_recall_curve


def optimize_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    min_recall: float = 0.30,
) -> Dict[str, float]:
    """Seleciona um threshold por classe maximizando F1 sob restrição de recall mínimo.

    Quando nenhum threshold satisfaz ``min_recall``, escolhe o threshold de maior
    recall disponível (fallback).

    Parameters
    ----------
    y_true : np.ndarray
        Matriz binária de rótulos verdadeiros com shape ``(n_samples, n_classes)``.
    y_pred : np.ndarray
        Matriz de scores preditos com shape ``(n_samples, n_classes)``.
    class_names : list[str]
        Nomes das classes na ordem das colunas.
    min_recall : float, optional
        Recall mínimo exigido para que um threshold seja elegível, por padrão 0.30.

    Returns
    -------
    Dict[str, float]
        Dicionário ``{nome_da_classe: threshold}``.
    """
    thresholds: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        precision, recall, thresh = precision_recall_curve(y_true[:, i], y_pred[:, i])
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        valid = recall[1:] >= min_recall
        if valid.any():
            idx = np.argmax(f1[1:][valid])
            thresholds[name] = float(thresh[valid][idx])
        else:
            idx = np.argmax(recall[1:])
            thresholds[name] = float(thresh[idx])
    return thresholds


def apply_thresholds(
    y_pred: np.ndarray, thresholds: Dict[str, float], class_names: list[str]
) -> np.ndarray:
    """Aplica thresholds por classe e retorna índices das classes ativas.

    Cada amostra recebe o índice da última classe cujo score ultrapassar o
    threshold correspondente. A ordem em ``class_names`` define a precedência
    em caso de múltiplas classes ativas.

    Parameters
    ----------
    y_pred : np.ndarray
        Matriz de scores preditos com shape ``(n_samples, n_classes)``.
    thresholds : Dict[str, float]
        Dicionário ``{nome_da_classe: threshold}``.
    class_names : list[str]
        Nomes das classes na ordem das colunas.

    Returns
    -------
    np.ndarray
        Vetor de índices de classe ativados, shape ``(n_samples,)``.
    """
    result = np.zeros(len(y_pred), dtype=int)
    for i, name in enumerate(class_names):
        mask = y_pred[:, i] >= thresholds[name]
        result[mask] = i
    return result
