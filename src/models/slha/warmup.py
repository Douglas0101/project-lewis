"""Warmup leve para estimar memória e latência por amostra."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import psutil
import tensorflow as tf

from .exceptions import WarmupError

LOGGER = logging.getLogger("lewis.slha.warmup")


@dataclass(frozen=True)
class WarmupResult:
    batch_time_ms: float
    samples_per_second: float
    peak_ram_mb: float
    estimated_memory_per_sample_mb: float


def warmup_model(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 2,
    max_batches: int = 2,
    timeout_seconds: float = 30.0,
) -> WarmupResult:
    """Executa 1-2 batches com gradientes desabilitados para estimar recursos.

    Nunca modifica os pesos do modelo (usa tf.GradientTape não treinável).
    """
    if len(X) < batch_size:
        batch_size = max(1, len(X))

    process = psutil.Process()
    start_ram = process.memory_info().rss
    start_time = time.perf_counter()

    try:
        batches_run = 0
        for i in range(0, min(len(X), batch_size * max_batches), batch_size):
            x_batch = X[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            with tf.GradientTape(persistent=False, watch_accessed_variables=False):
                _ = model(tf.stop_gradient(x_batch), training=False)
            batches_run += 1
            if time.perf_counter() - start_time > timeout_seconds:
                LOGGER.warning("Warmup atingiu timeout de %.1fs", timeout_seconds)
                break
    except Exception as exc:
        raise WarmupError(f"Falha durante warmup: {exc}") from exc

    elapsed = time.perf_counter() - start_time
    peak_ram = process.memory_info().rss
    delta_ram_mb = max(0.0, (peak_ram - start_ram) / (1024 * 1024))
    samples = batches_run * batch_size

    if elapsed <= 0 or samples <= 0:
        raise WarmupError("Warmup não processou nenhuma amostra")

    estimated_per_sample = delta_ram_mb / samples if samples > 0 else 0.0

    return WarmupResult(
        batch_time_ms=round(elapsed * 1000 / batches_run, 2),
        samples_per_second=round(samples / elapsed, 2),
        peak_ram_mb=round(delta_ram_mb, 2),
        estimated_memory_per_sample_mb=round(estimated_per_sample, 4),
    )
