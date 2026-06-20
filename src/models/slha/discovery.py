"""Discovery de hardware via psutil e TensorFlow."""

from __future__ import annotations

import logging
import os
import platform
import socket
from typing import List

import psutil
import tensorflow as tf

from .exceptions import DiscoveryError
from .schemas import CPUInfo, DiskInfo, GPUDevice, GPUInfo, HardwareSpecs, RAMInfo

LOGGER = logging.getLogger("lewis.slha.discovery")


def _read_cpu() -> CPUInfo:
    try:
        info = {
            "physical_cores": psutil.cpu_count(logical=False) or 1,
            "logical_cores": psutil.cpu_count(logical=True) or 1,
            "max_freq_mhz": None,
            "architecture": platform.machine(),
            "flags": [],
        }
        freq = psutil.cpu_freq()
        if freq and freq.max:
            info["max_freq_mhz"] = float(freq.max)
    except Exception as exc:  # pragma: no cover
        raise DiscoveryError(f"Falha ao ler CPU: {exc}") from exc

    # flags SIMD (melhor esforço)
    try:
        if hasattr(os, "sysconf") and os.path.isdir("/proc"):
            with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("flags"):
                        raw = line.split(":", 1)[1].strip().split()
                        wanted = {"avx", "avx2", "avx512f", "sse4_2", "fma"}
                        info["flags"] = sorted([f for f in raw if f in wanted])
                        break
    except Exception:
        pass

    return CPUInfo(**info)


def _read_gpu() -> GPUInfo:
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as exc:
        LOGGER.warning("TensorFlow não conseguiu listar GPUs: %s", exc)
        return GPUInfo(available=False, count=0, devices=[])

    devices: List[GPUDevice] = []
    for idx, gpu in enumerate(gpus):
        try:
            details = tf.config.experimental.get_device_details(gpu)
            dev = GPUDevice(
                index=idx,
                name=details.get("device_name", "unknown"),
                total_memory_mb=details.get("memory_limit", 0) // (1024 * 1024),
                compute_capability=details.get("compute_capability"),
            )
            devices.append(dev)
        except Exception as exc:
            LOGGER.warning("Ignorando GPU %d devido a erro: %s", idx, exc)

    return GPUInfo(available=len(devices) > 0, count=len(devices), devices=devices)


def _read_ram() -> RAMInfo:
    mem = psutil.virtual_memory()
    return RAMInfo(
        total_gb=round(mem.total / (1024**3), 1),
        available_gb=round(mem.available / (1024**3), 1),
        percent_used=mem.percent,
    )


def _read_disk() -> DiskInfo:
    usage = psutil.disk_usage(".")
    return DiskInfo(
        total_gb=round(usage.total / (1024**3), 1),
        available_gb=round(usage.free / (1024**3), 1),
    )


def discover_hardware() -> HardwareSpecs:
    """Coleta especificações de hardware com graceful degradation."""
    try:
        return HardwareSpecs(
            hostname=socket.gethostname(),
            os=f"{platform.system()} {platform.release()}",
            cpu=_read_cpu(),
            gpu=_read_gpu(),
            ram=_read_ram(),
            disk=_read_disk(),
        )
    except Exception as exc:
        LOGGER.exception("Discovery falhou")
        raise DiscoveryError(str(exc)) from exc
