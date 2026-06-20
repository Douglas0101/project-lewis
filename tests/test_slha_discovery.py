import time

from src.models.slha.discovery import discover_hardware
from src.models.slha.schemas import HardwareSpecs


def test_discovery_returns_valid_specs():
    specs = discover_hardware()
    assert isinstance(specs, HardwareSpecs)
    assert specs.cpu.physical_cores >= 1
    assert specs.cpu.logical_cores >= specs.cpu.physical_cores
    assert specs.ram.total_gb > 0
    assert specs.ram.available_gb >= 0
    assert specs.disk.total_gb > 0


def test_discovery_runs_under_two_seconds():
    start = time.perf_counter()
    discover_hardware()
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0


def test_cpu_only_fallback_never_raises():
    # Mesmo sem GPU, a função deve retornar specs com gpu.available=False
    specs = discover_hardware()
    assert isinstance(specs.gpu.available, bool)
    if not specs.gpu.available:
        assert specs.gpu.count == 0
        assert specs.gpu.devices == []
