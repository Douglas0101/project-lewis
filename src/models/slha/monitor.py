"""Callback Keras para monitoramento de recursos durante treino."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import psutil
import tensorflow as tf

from .schemas import ResourceLog

LOGGER = logging.getLogger("lewis.slha.monitor")


class ResourceMonitor(tf.keras.callbacks.Callback):
    """Loga uso de CPU/RAM/GPU a cada epoch sem interromper o treino."""

    def __init__(self, log_path: Optional[Path] = None, alert_cpu_threshold: float = 95.0):
        super().__init__()
        self.log_path = Path(log_path) if log_path else None
        self.alert_cpu_threshold = alert_cpu_threshold
        self._process = psutil.Process()

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        try:
            entry = self._build_log(epoch)
            LOGGER.info(
                "ResourceMonitor | epoch=%d | cpu=%.1f%% | ram=%.2fGB",
                entry.epoch,
                entry.cpu_percent,
                entry.ram_used_gb,
            )
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(entry.model_dump_json() + "\n")
        except Exception as exc:
            # Isolamento de falhas: nunca quebrar o treino.
            LOGGER.warning("ResourceMonitor falhou no epoch %d: %s", epoch, exc)

    def _build_log(self, epoch: int) -> ResourceLog:
        cpu_percent = self._process.cpu_percent(interval=None)
        # psutil pode retornar valores > 100 em CPUs multi-core; normalizar para escala 0-100.
        cpu_percent = max(0.0, min(100.0, cpu_percent))
        ram_used_gb = self._process.memory_info().rss / (1024**3)
        alerts = []
        if cpu_percent > self.alert_cpu_threshold:
            alerts.append(f"CPU acima de {self.alert_cpu_threshold}%")

        gpu_util = None
        gpu_mem_used = None
        gpu_mem_total = None
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                # tf não expõe utilização percentual facilmente sem pynvml.
                # Deixamos como None quando pynvml não está disponível.
                gpu_mem_total = self._try_gpu_memory_total()
        except Exception:
            pass

        return ResourceLog(
            epoch=epoch,
            cpu_percent=round(cpu_percent, 1),
            ram_used_gb=round(ram_used_gb, 2),
            gpu_utilization_percent=gpu_util,
            gpu_memory_used_mb=gpu_mem_used,
            gpu_memory_total_mb=gpu_mem_total,
            alerts=alerts,
        )

    def _try_gpu_memory_total(self) -> Optional[float]:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(info.total) / (1024 * 1024)
        except Exception:
            return None
