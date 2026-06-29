"""Normalizacao de sinal para o pipeline DSP do firmware."""

from __future__ import annotations

import numpy as np


def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Aplica Z-score em um array float32 (equivalente a lewis_zscore_normalize).

    Reproduz a implementacao C em precisao simples: media em uma passada,
    desvio padrao populacional em segunda passada e normalizacao in-place.
    """
    x = x.astype(np.float32)
    n = float(x.shape[0])
    mean = np.sum(x) / np.float32(n)
    diff = x - mean
    std = np.sqrt(np.sum(diff * diff) / np.float32(n))
    if std == 0.0:
        return np.zeros_like(x, dtype=np.float32)
    return (diff / std).astype(np.float32)
