"""Monitoramento de drift do scaler de entrada."""

from typing import Any, Dict, Optional

import numpy as np


def compute_drift_metrics(batch: np.ndarray, scaler: Any) -> Dict[str, Any]:
    """Calcula métricas de drift entre um batch e a estatística do scaler.

    Args:
        batch: Array de forma (n_samples, n_features).
        scaler: Scaler treinado com atributos ``mean_`` e ``scale_``.

    Returns:
        Dicionário com mean_drift e scale_drift.

    Raises:
        ValueError: Se ``batch`` não for um ndarray 2D não vazio ou se
            ``scaler`` não possuir os atributos ``mean_`` e ``scale_``.
    """
    if not isinstance(batch, np.ndarray) or batch.ndim != 2 or batch.shape[0] == 0:
        raise ValueError("batch deve ser ndarray 2D não vazio")
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise ValueError("scaler deve ter mean_ e scale_")
    mean_drift = np.abs(batch.mean(axis=0) - scaler.mean_) / scaler.scale_
    scale_drift = np.abs(batch.std(axis=0) - scaler.scale_) / scaler.scale_
    return {
        "mean_drift": mean_drift.tolist(),
        "scale_drift": scale_drift.tolist(),
    }


def check_drift(
    batch: np.ndarray, scaler: Any, thresholds: Optional[Dict[str, float]] = None
) -> bool:
    """Verifica se o batch está dentro dos limites de drift do scaler.

    Args:
        batch: Array de forma (n_samples, n_features).
        scaler: Scaler treinado com atributos ``mean_`` e ``scale_``.
        thresholds: Limites para mean e scale. Padrão: mean=0.1, scale=0.15.

    Returns:
        True se nenhum drift significativo for detectado, False caso contrário.
    """
    thresholds = thresholds or {"mean": 0.1, "scale": 0.15}
    metrics = compute_drift_metrics(batch, scaler)
    mean_drift = np.array(metrics["mean_drift"])
    scale_drift = np.array(metrics["scale_drift"])
    safe = np.all(mean_drift < thresholds["mean"]) and np.all(scale_drift < thresholds["scale"])
    return bool(safe)
