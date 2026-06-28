"""Quality Gates de firmware (Camada 8) — QG7 e QG9.

Estes testes dependem do artefato gerado por
`make -C firmware LEWIS_USE_TFLM=1 firmware-test`, que produz
`firmware/build/stm32f4/firmware_simulation_report.json`.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRMWARE_ROOT = PROJECT_ROOT / "firmware"
MAKEFILE = FIRMWARE_ROOT / "Makefile"


def _resolve_arm_toolchain() -> Path | None:
    """Descobre o diretorio raiz do toolchain ARM.

    1. Variavel de ambiente ``ARM_TOOLCHAIN``.
    2. Parse de ``ARM_DIR`` no ``firmware/Makefile``.
    3. Busca padrao em ``firmware/tools``.
    """
    env_toolchain = os.environ.get("ARM_TOOLCHAIN")
    if env_toolchain:
        path = Path(env_toolchain)
        if path.exists():
            return path

    if MAKEFILE.exists():
        text = MAKEFILE.read_text(encoding="utf-8")
        m = re.search(r"ARM_DIR\s*:=\s*(.+)", text)
        if m:
            raw = m.group(1).strip()
            # Makefile usa ``$(PROJECT_ROOT)/tools/...``
            raw = raw.replace("$(PROJECT_ROOT)", str(FIRMWARE_ROOT))
            raw = raw.replace("$(TOOLS_DIR)", str(FIRMWARE_ROOT / "tools"))
            path = Path(raw)
            if path.exists():
                return path

    # Fallback: procura por diretorios conhecidos em firmware/tools.
    tools_dir = FIRMWARE_ROOT / "tools"
    for prefix in (
        "xpack-arm-none-eabi-gcc-",
        "arm-gnu-toolchain-",
        "gcc-arm-none-eabi-",
    ):
        if tools_dir.exists():
            for candidate in tools_dir.iterdir():
                if candidate.is_dir() and candidate.name.startswith(prefix):
                    return candidate
    return None


@pytest.mark.qg7
class TestQG7Build:
    def test_firmware_build_zero_warnings(self):
        """QG7: build STM32F4 com -Werror deve terminar com sucesso."""
        if _resolve_arm_toolchain() is None:
            pytest.skip(
                "Toolchain ARM nao encontrado (defina ARM_TOOLCHAIN ou verifique firmware/tools)"
            )
        subprocess.run(
            ["make", "-C", str(FIRMWARE_ROOT), "LEWIS_USE_TFLM=1", "stm32f4"],
            check=True,
            timeout=300,
        )

    def test_flatbuffer_size_under_64kb(self):
        """QG7: cada FlatBuffer (stage1/stage2) deve ter menos de 64 KB."""
        quantized_dir = PROJECT_ROOT / "models" / "quantized"
        for name in ("stage1_int8_v2.0.tflite", "stage2_int8_v2.0.tflite"):
            path = quantized_dir / name
            if not path.exists():
                pytest.skip(f"Modelo nao encontrado: {path}")
            size = path.stat().st_size
            assert size < 64 * 1024, f"{name} tem {size} bytes (limite 64 KB)"


@pytest.mark.qg9
class TestQG9Runtime:
    def test_inference_latency_under_200ms(self, firmware_report):
        """QG9: latencia de inferencia < 200 ms/batimento @ 168 MHz."""
        assert firmware_report["checks"]["all_passed"], "Simulacao falhou"
        times = firmware_report["beat_times_ms"]
        assert len(times) >= 1
        for t in times:
            assert t < 200, f"Latencia {t} ms >= 200 ms"

    def test_tflm_arena_under_64kb(self, firmware_report):
        """QG9: arena TFLM usada deve ser < 64 KB."""
        text = firmware_report["uart_log_text"]
        m = re.search(r"Arena used:\s+(\d+)\s+bytes", text)
        assert m, "Arena used nao encontrada no log"
        arena_used = int(m.group(1))
        assert arena_used < 64 * 1024, f"Arena usada {arena_used} bytes (limite 64 KB)"

    def test_total_flash_under_512kb(self, firmware_report):
        """QG9: Flash total (text + data) deve ser < 512 KB."""
        elf = Path(firmware_report["firmware_elf"])
        toolchain_root = _resolve_arm_toolchain()
        if toolchain_root is None:
            pytest.skip(
                "Toolchain ARM nao encontrado (defina ARM_TOOLCHAIN ou verifique firmware/tools)"
            )
        size_bin = toolchain_root / "bin" / "arm-none-eabi-size"
        if not size_bin.exists():
            pytest.skip(f"arm-none-eabi-size nao encontrado em {size_bin}")
        result = subprocess.run(
            [str(size_bin), str(elf)],
            check=True,
            capture_output=True,
            text=True,
        )
        line = result.stdout.strip().splitlines()[-1]
        parts = line.split()
        text = int(parts[0])
        data = int(parts[1])
        flash_total = text + data
        assert flash_total < 512 * 1024, f"Flash total {flash_total} bytes (limite 512 KB)"
