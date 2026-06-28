"""Monitoramento de drift do scaler de entrada."""

from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import ks_2samp


def compute_drift_metrics(batch: np.ndarray, scaler: Any) -> Dict[str, Any]:
    """Calcula métricas de drift entre um batch e a estatística do scaler.

    Args:
        batch: Array de forma (n_samples, n_features).
        scaler: Scaler treinado com atributos ``mean_`` e ``scale_``.

    Returns:
        Dicionário com mean_drift, scale_drift, ks_statistic e ks_pvalue.
    """
    mean_drift = np.abs(batch.mean(axis=0) - scaler.mean_) / scaler.scale_
    scale_drift = np.abs(batch.std(axis=0) - scaler.scale_) / scaler.scale_
    ks_stat, ks_pvalue = ks_2samp(
        batch.flatten(),
        np.random.normal(loc=scaler.mean_[0], scale=scaler.scale_[0], size=len(batch.flatten())),
    )
    return {
        "mean_drift": mean_drift.tolist(),
        "scale_drift": scale_drift.tolist(),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
    }


def check_drift(
    batch: np.ndarray, scaler: Any, thresholds: Optional[Dict[str, float]] = None
) -> bool:
    """Verifica se o batch está dentro dos limites de drift do scaler.

    Args:
        batch: Array de forma (n_samples, n_features).
        scaler: Scaler treinado com atributos ``mean_`` e ``scale_``.
        thresholds: Limites para mean, scale e psi. Padrão: mean=0.1, scale=0.15,
            psi=0.25.

    Returns:
        True se nenhum drift significativo for detectado, False caso contrário.
    """
    thresholds = thresholds or {"mean": 0.1, "scale": 0.15, "psi": 0.25}
    mean_drift = np.abs(batch.mean(axis=0) - scaler.mean_) / scaler.scale_
    scale_drift = np.abs(batch.std(axis=0) - scaler.scale_) / scaler.scale_
    safe = np.all(mean_drift < thresholds["mean"]) and np.all(scale_drift < thresholds["scale"])
    return bool(safe)
