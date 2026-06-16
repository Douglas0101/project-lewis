"""Normalizacao de sinal para o pipeline DSP do firmware."""

from __future__ import annotations

import numpy as np


def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Aplica Z-score em um array float32 (equivalente a lewis_zscore_normalize).

    Calcula media e desvio padrao da janela inteira e retorna
    (x - mean) / std. Se std == 0, retorna zeros.
    """
    x = x.astype(np.float64)
    mean = np.mean(x)
    std = np.std(x)
    if std == 0.0:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mean) / std).astype(np.float32)
