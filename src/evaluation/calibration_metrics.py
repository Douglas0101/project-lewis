"""Métricas de calibração do avaliador canônico (ML Protocol v2, seção 5).

Adapta ``src/evaluation/calibration_v3.py`` (binagem/ECE/Brier/NLL) ao formato
multi-label do protocolo, mantendo **paridade numérica com o legado**
(``src/models/pretrain_evaluation.py``):

- mesma binagem (linspace 0–1, último bin fechado à direita);
- ECE/MCE macro = média não ponderada dos ECE/MCE por classe;
- fit de temperatura = mesmo objetivo do legado (NLL binário médio de
  ``logits / T``, ``minimize_scalar`` bounded (0.05, 20)) — reproduz T=0,3741
  do A2-full dentro da tolerância de reconciliação (1e-4);
- aplicação: ``sigmoid(logit(p) / T)`` com logits clipados em ±30 (idêntico
  ao legado); AUROC/PR-AUC são invariantes a T (monotonia) — testado.

ECE mede magnitude, não direção: a direção vem de ``T - 1`` e do reliability
diagram (``reliability`` em :func:`calibration_report`).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from src.evaluation.calibration_v3 import brier_binary, ece_binary, nll_binary
from src.evaluation.metric_definitions import bce_multilabel

_EPS = 1e-7
_LOGIT_CLIP = 30.0


def prob_to_logits(y_prob: np.ndarray) -> np.ndarray:
    """logit(p) com clip em 1e-7 (idêntico ao legado ``sigmoid_to_logits``)."""
    p = np.clip(np.asarray(y_prob, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def apply_temperature(y_prob: np.ndarray, temperature: float) -> np.ndarray:
    """sigmoid(logit/T) com clip de logits em ±30 (idêntico ao legado)."""
    logits = prob_to_logits(y_prob) / float(temperature)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -_LOGIT_CLIP, _LOGIT_CLIP)))


def _nll_from_logits(y_true: np.ndarray, logits: np.ndarray) -> float:
    """NLL binário médio por elemento na forma softplus (idêntico ao legado)."""
    z = np.clip(logits, -_LOGIT_CLIP, _LOGIT_CLIP)
    return float(np.mean(np.maximum(z, 0) - z * y_true + np.log1p(np.exp(-np.abs(z)))))


def fit_temperature_multilabel(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Temperatura escalar que minimiza o NLL multi-label (logits/T).

    Mesmo objetivo e mesmo otimizador do legado (``fit_temperature``):
    ``minimize_scalar`` bounded (0.05, 20).
    """
    y = np.asarray(y_true, dtype=np.float64)
    logits = prob_to_logits(y_prob)

    def objective(t: float) -> float:
        return _nll_from_logits(y, logits / t)

    result = minimize_scalar(objective, bounds=(0.05, 20.0), method="bounded")
    return float(result.x)


def _mce_from_bins(bins: list[dict]) -> Optional[float]:
    """MCE = maior |acc - conf| entre bins não vazios (None se todos vazios)."""
    gaps = [abs(b["acc"] - b["conf"]) for b in bins if b["n"] > 0]
    return float(max(gaps)) if gaps else None


def ece_stratified(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray,
    n_bins: int,
    class_names: Sequence[str],
) -> dict:
    """ECE macro e por classe restrito ao subconjunto ``mask`` (bool, shape (n,)).

    Uso normativo (ML Protocol v2, hipótese H2): calibração no estrato NORM=0 do
    pré-treino SCP-ECG — a temperatura global otimiza a marginal e pode deixar o
    estrato patológico descalibrado (T9.3, Tabela 4: ECE NORM=0 até 0,217).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"mask com shape {mask.shape} incompatível com y_true {y_true.shape}"
        )
    per_class: dict[str, dict] = {}
    eces: list[float] = []
    for idx, name in enumerate(class_names):
        yt = y_true[mask, idx].astype(np.float64)
        yp = y_prob[mask, idx]
        ece, _ = ece_binary(yt, yp, n_bins=n_bins)
        per_class[name] = {"ece": ece, "support": int(yt.sum())}
        eces.append(ece)
    return {
        "n_bins": int(n_bins),
        "n_samples": int(mask.sum()),
        "macro_ece": float(np.mean(eces)) if eces else None,
        "per_class": per_class,
    }


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
    class_names: Sequence[str],
) -> dict:
    """ECE/MCE/Brier/NLL/BCE macro e por classe + reliability bins.

    Retorna ``{"n_bins", "macro", "per_class", "reliability"}`` onde
    ``reliability.per_class[nome]`` é a lista de bins (lo/hi/n/acc/conf) —
    serializada diretamente como ``reliability.json`` pelo orquestrador.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    per_class: dict[str, dict] = {}
    reliability: dict[str, list] = {}
    eces: list[float] = []
    mces: list[float] = []
    briers: list[float] = []
    nlls: list[float] = []
    for idx, name in enumerate(class_names):
        yt = y_true[:, idx].astype(np.float64)
        yp = y_prob[:, idx]
        ece, bins = ece_binary(yt, yp, n_bins=n_bins)
        mce = _mce_from_bins(bins)
        brier = brier_binary(yt, yp)
        nll = nll_binary(yt, yp)
        per_class[name] = {"ece": ece, "mce": mce, "brier": brier, "nll": nll}
        reliability[name] = bins
        eces.append(ece)
        briers.append(brier)
        nlls.append(nll)
        if mce is not None:
            mces.append(mce)
    return {
        "n_bins": int(n_bins),
        "macro": {
            "ece": float(np.mean(eces)) if eces else None,
            "mce": float(np.mean(mces)) if mces else None,
            "brier": float(np.mean(briers)) if briers else None,
            "nll": float(np.mean(nlls)) if nlls else None,
            "bce": bce_multilabel(y_true, y_prob),
        },
        "per_class": per_class,
        "reliability": {"n_bins": int(n_bins), "per_class": reliability},
    }
