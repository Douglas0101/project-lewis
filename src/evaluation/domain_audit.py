"""Auditoria de atalhos de domínio (docs/rebuild_spec/06).

Pergunta central: o modelo aprendeu fisiologia ou a origem do arquivo?

- probe de identidade do dataset sobre embeddings (baseline esperado ≈ chance);
- métricas condicionais M(classe, dataset/paciente/qualidade);
- bateria contrafactual (invariância a fatores não fisiológicos).

Estados possíveis: ``DATASET_SHORTCUT_LEARNING``, ``INSUFFICIENT_EVIDENCE``,
``REVIEW_REQUIRED``. Nenhum estado de aprovação é emitido aqui isoladamente.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

LOGGER = logging.getLogger("lewis.evaluation.domain_audit")

DATASET_SHORTCUT_LEARNING = "DATASET_SHORTCUT_LEARNING"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
REVIEW_REQUIRED = "REVIEW_REQUIRED"


# ---------------------------------------------------------------------------
# probe de identidade do dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeResult:
    balanced_acc: float
    chance: float
    delta: float
    n_classes: int
    shortcut: bool
    per_fold: list[float]


def dataset_id_probe(
    embeddings: np.ndarray,
    dataset_ids: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    delta_threshold: float = 0.05,
) -> ProbeResult:
    """Classificador de auditoria: recupera ``dataset_id`` dos embeddings.

    Cross-validation por paciente (``groups``) quando fornecida; caso
    contrário, GroupKFold sobre os próprios índices (aproximação — reportar).
    ``delta_threshold``: probe acima de chance+δ com consistência entre folds
    indica atalho material (docs/rebuild_spec/06).
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    dataset_ids = np.asarray(dataset_ids)
    classes = np.unique(dataset_ids)
    n_classes = len(classes)
    if n_classes < 2:
        raise ValueError("probe exige ≥2 datasets")
    chance = 1.0 / n_classes
    if groups is None:
        groups = np.arange(len(embeddings))

    per_fold: list[float] = []
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(embeddings, dataset_ids, groups):
        if len(np.unique(dataset_ids[tr])) < n_classes or len(np.unique(dataset_ids[te])) < 1:
            continue
        clf = LogisticRegression(max_iter=2000)
        clf.fit(embeddings[tr], dataset_ids[tr])
        acc_fold = balanced_accuracy_score(dataset_ids[te], clf.predict(embeddings[te]))
        per_fold.append(float(acc_fold))

    if not per_fold:
        raise ValueError("probe sem folds válidos — verifique grupos/classes")
    acc = float(np.mean(per_fold))
    delta = acc - chance
    return ProbeResult(
        balanced_acc=acc,
        chance=chance,
        delta=delta,
        n_classes=n_classes,
        shortcut=delta > delta_threshold,
        per_fold=per_fold,
    )


# ---------------------------------------------------------------------------
# métricas condicionais
# ---------------------------------------------------------------------------

def conditional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: Any,
    class_names: list[str],
    by: str = "dataset",
) -> dict[str, dict[str, float]]:
    """M(classe, grupo): recall por classe dentro de cada grupo (dataset/paciente/qualidade).

    Nenhuma média global pode ocultar colapso em um domínio (rebuild_spec/06 §4).
    """
    out: dict[str, dict[str, float]] = {}
    groups = df[by].astype(str).to_numpy()
    for group_value in sorted(set(groups.tolist())):
        sel = groups == group_value
        per_class: dict[str, float] = {}
        for idx, name in enumerate(class_names):
            cls_sel = sel & (y_true == idx)
            if cls_sel.sum() == 0:
                per_class[name] = float("nan")
                continue
            per_class[name] = float((y_pred[cls_sel] == idx).mean())
        per_class["_support"] = float(sel.sum())
        out[group_value] = per_class
    return out


# ---------------------------------------------------------------------------
# bateria contrafactual
# ---------------------------------------------------------------------------

def _resample_500_250_500(x: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly

    down = resample_poly(x, up=1, down=2, axis=1)
    up = resample_poly(down, up=2, down=1, axis=1)
    return up[:, : x.shape[1], :].astype(np.float32)


def _edge_pad(x: np.ndarray, pad: int = 25) -> np.ndarray:
    core = x[:, pad:-pad, :]
    return np.pad(core, ((0, 0), (pad, pad), (0, 0)), mode="edge").astype(np.float32)


COUNTERFACTUAL_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "amplitude_x0.5": lambda x: (x * 0.5).astype(np.float32),
    "amplitude_x2.0": lambda x: (x * 2.0).astype(np.float32),
    "offset_+0.5": lambda x: (x + 0.5).astype(np.float32),
    "resample_500_250_500": _resample_500_250_500,
    "edge_pad_25": _edge_pad,
    "polarity_invert": lambda x: (-x).astype(np.float32),
}


def counterfactual_suite(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    transforms: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    noise_levels_snr_db: tuple[float, ...] = (30.0, 20.0),
    rng_seed: int = 0,
) -> dict[str, dict[str, float]]:
    """ΔP por intervenção não fisiológica + curva de ruído.

    ``predict_fn`` deve retornar probabilidades (n, C). Espera-se invariância
    para amplitude/offset/padding/resample/offset; sensibilidade apenas à
    fisiologia (rebuild_spec/06 §5).
    """
    transforms = transforms or COUNTERFACTUAL_TRANSFORMS
    base = predict_fn(X)
    rng = np.random.default_rng(rng_seed)
    out: dict[str, dict[str, float]] = {}
    for name, fn in transforms.items():
        p = predict_fn(fn(X))
        out[name] = {
            "delta_p_mean": float(np.abs(p - base).mean()),
            "delta_p_p95": float(np.percentile(np.abs(p - base), 95)),
            "argmax_flip_rate": float((p.argmax(axis=1) != base.argmax(axis=1)).mean()),
        }
    for snr_db in noise_levels_snr_db:
        signal_power = float(np.mean(X**2)) + 1e-12
        noise_power = signal_power / (10 ** (snr_db / 10))
        noisy = (X + rng.normal(0, np.sqrt(noise_power), X.shape)).astype(np.float32)
        p = predict_fn(noisy)
        out[f"noise_snr_{snr_db:.0f}db"] = {
            "delta_p_mean": float(np.abs(p - base).mean()),
            "delta_p_p95": float(np.percentile(np.abs(p - base), 95)),
            "argmax_flip_rate": float((p.argmax(axis=1) != base.argmax(axis=1)).mean()),
        }
    return out
