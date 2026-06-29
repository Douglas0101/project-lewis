"""SLHA — Sistema de Leitura de Hardware Automático (adaptado)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from .decision import decide_training_config
from .discovery import discover_hardware
from .monitor import ResourceMonitor
from .schemas import TrainingConfig
from .warmup import warmup_model

__all__ = [
    "auto_configure_training",
    "decide_training_config",
    "discover_hardware",
    "ResourceMonitor",
    "warmup_model",
]


def auto_configure_training(
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    model: tf.keras.Model,
    reference_batch_size: int = 64,
    log_dir: Optional[Path] = None,
) -> TrainingConfig:
    """Caminho feliz: discovery → warmup → decision.

    Parameters
    ----------
    X_sample : np.ndarray
        Pequeno subset de amostras para warmup (ex.: 8 amostras).
    y_sample : np.ndarray
        Labels correspondentes.
    model : tf.keras.Model
        Modelo a ser treinado.
    reference_batch_size : int
        Batch size desejado quando a memória permitir.
    log_dir : Path, optional
        Se informado, persiste logs estruturados de discovery, warmup e decision.

    Returns
    -------
    TrainingConfig
    """
    log_dir = Path(log_dir) if log_dir else None
    specs = discover_hardware(log_path=log_dir / "hardware_specs.json" if log_dir else None)
    warmup = warmup_model(
        model,
        X_sample,
        y_sample,
        batch_size=min(2, len(X_sample)),
        log_path=log_dir / "warmup_result.json" if log_dir else None,
    )
    config = decide_training_config(
        specs,
        estimated_memory_per_sample_mb=warmup.estimated_memory_per_sample_mb,
        reference_batch_size=reference_batch_size,
        log_path=log_dir / "training_config.json" if log_dir else None,
    )
    return config
