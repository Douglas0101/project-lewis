"""Tests for the SLHA decision engine."""

import pytest

from src.models.slha.decision import decide_training_config
from src.models.slha.schemas import CPUInfo, DiskInfo, GPUDevice, GPUInfo, HardwareSpecs, RAMInfo


def _cpu_only_specs(total_gb=16.0, available_gb=8.0):
    return HardwareSpecs(
        hostname="test",
        os="Linux test",
        cpu=CPUInfo(physical_cores=2, logical_cores=4, architecture="x86_64"),
        gpu=GPUInfo(available=False, count=0),
        ram=RAMInfo(total_gb=total_gb, available_gb=available_gb, percent_used=50.0),
        disk=DiskInfo(total_gb=100.0, available_gb=50.0),
    )


def test_decision_cpu_only_returns_valid_config():
    specs = _cpu_only_specs()
    config = decide_training_config(specs, estimated_memory_per_sample_mb=4.0)
    assert config.accelerator == "cpu"
    assert config.devices == 1
    assert config.batch_size >= 1
    assert config.precision == "float32"


def test_batch_size_never_below_one():
    specs = _cpu_only_specs(total_gb=1.0, available_gb=0.1)
    config = decide_training_config(specs, estimated_memory_per_sample_mb=500.0)
    assert config.batch_size == 1


def test_gpu_config_uses_mixed_precision_only_when_available():
    specs = _cpu_only_specs()
    specs.gpu = GPUInfo(
        available=True,
        count=1,
        devices=[GPUDevice(index=0, total_memory_mb=4096, name="Tesla T4")],
    )
    config = decide_training_config(specs, estimated_memory_per_sample_mb=8.0)
    assert config.accelerator == "gpu"
    assert config.precision in {"float32", "mixed_float16"}
