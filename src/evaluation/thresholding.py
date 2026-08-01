"""Políticas de threshold do avaliador canônico (ML Protocol v2, seção 6).

Regras duras:

- thresholds são ajustados SOMENTE em validation/calibration, nunca em test —
  ``fit_thresholds`` e ``apply_thresholds`` são funções separadas exatamente
  para tornar esse fluxo explícito;
- todo ``metrics.json`` referencia a política e o ``fit_split`` usados;
- mudança de política quebra comparabilidade (ver ``schema.check_comparable``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import precision_recall_curve

from src.evaluation.threshold_optimizer import optimize_thresholds

POLICIES = (
    "fixed_0.5",
    "max_f1_per_class",
    "min_sensitivity_per_class",
    "cost_sensitive",
)


@dataclass(frozen=True)
class ThresholdPolicy:
    """Política de threshold: nome + parâmetros (imutável)."""

    name: str
    min_recall: float = 0.30
    cost_fn: float = 1.0
    cost_fp: float = 1.0

    def __post_init__(self) -> None:
        if self.name not in POLICIES:
            raise ValueError(f"política desconhecida '{self.name}'; opções: {POLICIES}")


def _max_f1_threshold(yt: np.ndarray, ys: np.ndarray) -> float:
    """Threshold que maximiza F1 sobre a curva precision-recall (determinístico)."""
    precision, recall, thresh = precision_recall_curve(yt, ys)
    if thresh.size == 0:
        return 0.5
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    idx = int(np.argmax(f1[1:]))  # ignora o ponto (recall=0) sem threshold
    return float(thresh[min(idx, thresh.size - 1)])


def _cost_sensitive_threshold(
    yt: np.ndarray, ys: np.ndarray, cost_fn: float, cost_fp: float
) -> float:
    """Threshold de menor custo esperado: cost_fn*FN + cost_fp*FP por classe."""
    candidates = np.unique(np.concatenate(([0.5], np.quantile(ys, np.linspace(0, 1, 201)))))
    best_t, best_cost = 0.5, np.inf
    n = len(yt)
    for t in candidates:
        yp = ys >= t
        fn = int(((yt == 1) & ~yp).sum())
        fp = int(((yt == 0) & yp).sum())
        cost = (cost_fn * fn + cost_fp * fp) / max(n, 1)
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return best_t


def fit_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Sequence[str],
    policy: ThresholdPolicy,
) -> dict[str, float]:
    """Ajusta thresholds por classe — SOMENTE em dados de validation/calibration."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if policy.name == "fixed_0.5":
        return {name: 0.5 for name in class_names}
    if policy.name == "min_sensitivity_per_class":
        return optimize_thresholds(
            y_true, y_score, list(class_names), min_recall=policy.min_recall
        )
    thresholds: dict[str, float] = {}
    for idx, name in enumerate(class_names):
        yt, ys = y_true[:, idx], y_score[:, idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            thresholds[name] = 0.5  # classe degenerada: fallback neutro
            continue
        if policy.name == "max_f1_per_class":
            thresholds[name] = _max_f1_threshold(yt, ys)
        elif policy.name == "cost_sensitive":
            thresholds[name] = _cost_sensitive_threshold(
                yt, ys, policy.cost_fn, policy.cost_fp
            )
        else:  # pragma: no cover - guardado por __post_init__
            raise ValueError(f"política não implementada: {policy.name}")
    return thresholds


def apply_thresholds(
    y_score: np.ndarray,
    thresholds: dict[str, float],
    class_names: Sequence[str],
) -> np.ndarray:
    """Aplica thresholds congelados e retorna matriz de decisão (n, k) em {0, 1}."""
    y_score = np.asarray(y_score)
    decision = np.zeros_like(y_score, dtype=int)
    for idx, name in enumerate(class_names):
        decision[:, idx] = (y_score[:, idx] >= thresholds[name]).astype(int)
    return decision
