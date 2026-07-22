"""Calibração probabilística v3 (docs/rebuild_spec/07).

Princípios duros:

- calibração NUNCA corrige ausência de discriminação — pré-condições verificadas
  pelo chamador (ranking útil, estabilidade, suporte, sem leakage);
- calibradores ajustados SOMENTE na partição de calibração (independente do
  treino do modelo-base); seleção por NLL na MESMA partição — nunca no teste;
- ECE sozinho não é critério: reportar NLL, Brier, ECE global e por classe,
  e tabela de confiabilidade.

Escopos: binário/multilabel (temperature, Platt, beta, isotonic) e multiclasse
(temperature, vector, Dirichlet).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger("lewis.evaluation.calibration_v3")

_EPS = 1e-12


# ---------------------------------------------------------------------------
# métricas de calibração
# ---------------------------------------------------------------------------

def nll_binary(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def nll_multiclass(y_true: np.ndarray, proba: np.ndarray) -> float:
    proba = np.clip(proba, _EPS, 1.0)
    return float(-np.mean(np.log(proba[np.arange(len(y_true)), y_true])))


def brier_binary(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def brier_multiclass(y_true: np.ndarray, proba: np.ndarray) -> float:
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def ece_binary(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins: list[dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not sel.any():
            continue
        acc = float(y_true[sel].mean())
        conf = float(p[sel].mean())
        ece += (sel.sum() / len(p)) * abs(acc - conf)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()),
                     "acc": acc, "conf": conf})
    return float(ece), bins


def ece_classwise(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> dict[int, float]:
    out: dict[int, float] = {}
    for c in range(proba.shape[1]):
        yc = (y_true == c).astype(float)
        out[c], _ = ece_binary(yc, proba[:, c], n_bins=n_bins)
    return out


def evaluate_binary(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> dict[str, Any]:
    ece, bins = ece_binary(y_true, p, n_bins=n_bins)
    return {
        "nll": nll_binary(y_true, p),
        "brier": brier_binary(y_true, p),
        "ece": ece,
        "reliability_bins": bins,
    }


def evaluate_multiclass(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> dict[str, Any]:
    return {
        "nll": nll_multiclass(y_true, proba),
        "brier": brier_multiclass(y_true, proba),
        "ece_classwise": ece_classwise(y_true, proba, n_bins=n_bins),
    }


# ---------------------------------------------------------------------------
# calibradores binários
# ---------------------------------------------------------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class BinaryCalibrator:
    """Calibrador binário sobre p(classe positiva)."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.method == "temperature":
            return _sigmoid(_logit(p) / self.params["T"])
        if self.method == "platt":
            z = self.params["a"] * _logit(p) + self.params["b"]
            return _sigmoid(z)
        if self.method == "beta":
            z = self.params["a"] * np.log(np.clip(p, _EPS, 1.0)) \
                - self.params["b"] * np.log(np.clip(1 - p, _EPS, 1.0)) + self.params["c"]
            return _sigmoid(z)
        if self.method == "isotonic":
            return self.params["iso"].predict(p)
        if self.method == "identity":
            return np.asarray(p, dtype=np.float64)
        raise ValueError(f"método desconhecido: {self.method}")


def fit_binary_calibrator(p: np.ndarray, y: np.ndarray, method: str) -> BinaryCalibrator:
    """Ajusta calibrador binário na partição de calibração."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if method == "temperature":
        def loss(t: np.ndarray) -> float:
            return nll_binary(y, _sigmoid(_logit(p) / t[0]))
        res = minimize(loss, x0=[1.0], bounds=[(1e-3, 100.0)], method="L-BFGS-B")
        return BinaryCalibrator(method=method, params={"T": float(res.x[0])})
    if method == "platt":
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(_logit(p).reshape(-1, 1), y)
        return BinaryCalibrator(
            method=method,
            params={"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])},
        )
    if method == "beta":
        Xf = np.column_stack(
            [np.log(np.clip(p, _EPS, 1.0)), -np.log(np.clip(1 - p, _EPS, 1.0))]
        )
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(Xf, y)
        return BinaryCalibrator(
            method=method,
            params={
                "a": float(lr.coef_[0][0]),
                "b": float(lr.coef_[0][1]),
                "c": float(lr.intercept_[0]),
            },
        )
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p, y)
        return BinaryCalibrator(method=method, params={"iso": iso})
    if method == "identity":
        return BinaryCalibrator(method=method, params={})
    raise ValueError(f"método desconhecido: {method}")


# ---------------------------------------------------------------------------
# calibradores multiclasse
# ---------------------------------------------------------------------------

@dataclass
class MulticlassCalibrator:
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def transform(self, proba: np.ndarray) -> np.ndarray:
        logp = np.log(np.clip(proba, _EPS, 1.0))
        if self.method == "temperature":
            z = logp / self.params["T"]
        elif self.method == "vector":
            z = logp * self.params["w"] + self.params["b"]
        elif self.method == "dirichlet":
            z = logp @ self.params["W"] + self.params["b"]
        elif self.method == "identity":
            z = logp
        else:
            raise ValueError(f"método desconhecido: {self.method}")
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


def fit_multiclass_calibrator(
    proba: np.ndarray, y: np.ndarray, method: str
) -> MulticlassCalibrator:
    proba = np.asarray(proba, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    logp = np.log(np.clip(proba, _EPS, 1.0))
    if method == "temperature":
        def loss(t: np.ndarray) -> float:
            z = logp / t[0]
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return nll_multiclass(y, e / e.sum(axis=1, keepdims=True))
        res = minimize(loss, x0=[1.0], bounds=[(1e-3, 100.0)], method="L-BFGS-B")
        return MulticlassCalibrator(method=method, params={"T": float(res.x[0])})
    if method == "vector":
        lr = LogisticRegression(C=1e6, max_iter=2000, multi_class="multinomial")
        lr.fit(logp, y)
        return MulticlassCalibrator(
            method=method,
            params={"w": np.diag(lr.coef_), "b": lr.intercept_},
        )
    if method == "dirichlet":
        lr = LogisticRegression(C=1e6, max_iter=2000, multi_class="multinomial")
        lr.fit(logp, y)
        return MulticlassCalibrator(
            method=method, params={"W": lr.coef_.T, "b": lr.intercept_}
        )
    if method == "identity":
        return MulticlassCalibrator(method=method, params={})
    raise ValueError(f"método desconhecido: {method}")


# ---------------------------------------------------------------------------
# seleção (somente na partição de calibração)
# ---------------------------------------------------------------------------

BINARY_METHODS = ("identity", "temperature", "platt", "beta", "isotonic")
MULTICLASS_METHODS = ("identity", "temperature", "vector", "dirichlet")


def select_binary_calibrator(
    p: np.ndarray,
    y: np.ndarray,
    methods: tuple[str, ...] = BINARY_METHODS,
    isotonic_min_support: int = 50,
) -> tuple[BinaryCalibrator, dict[str, float]]:
    """Seleciona o calibrador de menor NLL na partição de calibração.

    NUNCA usar no teste. Isotonic exige suporte mínimo (pacientes/classe,
    verificado pelo chamador — aqui apenas contagem de amostras).
    """
    scores: dict[str, float] = {}
    best: BinaryCalibrator | None = None
    for method in methods:
        if method == "isotonic" and np.sum(y == 1) < isotonic_min_support:
            LOGGER.info("isotonic pulado: suporte insuficiente (%d < %d)",
                        int(np.sum(y == 1)), isotonic_min_support)
            continue
        cal = fit_binary_calibrator(p, y, method)
        nll = nll_binary(y, cal.transform(p))
        scores[method] = nll
        if best is None or nll < scores[best.method]:
            best = cal
    assert best is not None
    return best, scores


def select_multiclass_calibrator(
    proba: np.ndarray,
    y: np.ndarray,
    methods: tuple[str, ...] = MULTICLASS_METHODS,
) -> tuple[MulticlassCalibrator, dict[str, float]]:
    scores: dict[str, float] = {}
    best: MulticlassCalibrator | None = None
    for method in methods:
        cal = fit_multiclass_calibrator(proba, y, method)
        nll = nll_multiclass(y, cal.transform(proba))
        scores[method] = nll
        if best is None or nll < scores[best.method]:
            best = cal
    assert best is not None
    return best, scores
