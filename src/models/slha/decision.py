"""Decision engine para configuração de treino TensorFlow/Keras."""

from __future__ import annotations

import logging
from typing import Literal

from .schemas import HardwareSpecs, TrainingConfig

LOGGER = logging.getLogger("lewis.slha.decision")

# Reserva 25% da RAM/VRAM para overhead do SO e do framework.
MEMORY_SAFETY_FACTOR = 0.75


def decide_training_config(
    specs: HardwareSpecs,
    estimated_memory_per_sample_mb: float = 8.0,
    reference_batch_size: int = 64,
) -> TrainingConfig:
    """Calcula configuração de treino com base nas specs e na memória estimada por amostra.

    Parameters
    ----------
    specs : HardwareSpecs
        Saída do discovery.
    estimated_memory_per_sample_mb : float
        Memória adicional estimada por amostra durante o treino (inclui ativações).
    reference_batch_size : int
        Batch size desejado quando a memória permitir.

    Returns
    -------
    TrainingConfig
    """
    has_gpu = specs.gpu.available and specs.gpu.count > 0
    accelerator: Literal["cpu", "gpu"] = "gpu" if has_gpu else "cpu"
    devices = 1

    if has_gpu:
        total_memory_mb = specs.gpu.devices[0].total_memory_mb
        precision: Literal["float32", "mixed_float16"] = (
            "mixed_float16" if _supports_mixed_precision(specs) else "float32"
        )
    else:
        total_memory_mb = int(specs.ram.available_gb * 1024)
        precision = "float32"

    usable_memory_mb = total_memory_mb * MEMORY_SAFETY_FACTOR
    batch_max = int(usable_memory_mb / max(estimated_memory_per_sample_mb, 0.1))
    batch_size = max(1, min(batch_max, reference_batch_size))

    num_workers = min(4, specs.cpu.logical_cores)
    pin_memory = has_gpu

    return TrainingConfig(
        accelerator=accelerator,
        strategy="single_device",
        devices=devices,
        batch_size=batch_size,
        precision=precision,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _supports_mixed_precision(specs: HardwareSpecs) -> bool:
    """Heurística: GPUs NVIDIA com compute capability >= 7.0 suportam FP16 bem."""
    if not specs.gpu.available or not specs.gpu.devices:
        return False
    dev = specs.gpu.devices[0]
    if not dev.compute_capability:
        # Se não conseguimos detectar, evitamos risco e usamos float32.
        return False
    try:
        major, _ = str(dev.compute_capability).split(".")
        return int(major) >= 7
    except Exception:
        return False
