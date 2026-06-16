"""Quality Gate QG12 — Limite de arena com 48 KB de RAM no Renode.

Verifica que o firmware emite ``INIT FAIL`` quando a RAM visivel e limitada
a 48 KB, pois a arena do TensorFlow Lite for Microcontrollers nao cabe.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRMWARE_ROOT = PROJECT_ROOT / "firmware"
RENODE_DIR = FIRMWARE_ROOT / "tools" / "renode-1.15.3"
RENODE_TEST_BIN = RENODE_DIR / "renode-test"
ROBOT_FILE = FIRMWARE_ROOT / "renode" / "arena_48k.robot"
UART_LOG = Path("/tmp/renode_lewis_uart.log")


def _ensure_renode() -> None:
    """Garante que o Renode esteja disponivel em firmware/tools."""
    if not RENODE_TEST_BIN.exists():
        pytest.skip(f"renode-test nao encontrado em {RENODE_TEST_BIN}")


def _find_python_with_robot() -> Path:
    """Retorna interpretador Python que possui robotframework."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        try:
            subprocess.run(
                [str(venv_python), "-c", "import robot"],
                check=True,
                capture_output=True,
                timeout=5,
            )
            return venv_python
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    return Path("python3")


@pytest.mark.qg12
@pytest.mark.slow
def test_arena_48k_init_fail() -> None:
    """QG12: com 48 KB de RAM a inicializacao TFLM deve falhar com INIT FAIL."""
    _ensure_renode()
    if not ROBOT_FILE.exists():
        pytest.skip(f"Arquivo robot nao encontrado: {ROBOT_FILE}")

    # Compila o firmware QG12: linker de 48 KB de RAM e arena de 16 KB.
    subprocess.run(
        ["make", "-C", str(FIRMWARE_ROOT), "LEWIS_USE_TFLM=1", "stm32f4-48k"],
        check=True,
        timeout=300,
    )

    bin_path = FIRMWARE_ROOT / "build" / "stm32f4" / "lewis_48k.bin"
    if not bin_path.exists():
        pytest.fail(f"Binario QG12 nao gerado: {bin_path}")

    if UART_LOG.exists():
        UART_LOG.unlink()

    python_runner = _find_python_with_robot()
    env = os.environ.copy()
    env["PATH"] = str(python_runner.parent) + os.pathsep + env.get("PATH", "")

    cmd = [
        str(RENODE_TEST_BIN),
        "--show-log",
        "-r",
        str(PROJECT_ROOT / "reports" / "renode_arena_48k"),
        str(ROBOT_FILE),
    ]

    # Executa a partir de firmware/ para manter a convencao de CWD dos
    # scripts .resc (os caminhos do binario sao absolutos).
    result = subprocess.run(
        cmd,
        cwd=FIRMWARE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        env=env,
    )

    output = result.stdout.decode("utf-8", errors="replace")
    tail = output[-4000:]

    if result.returncode != 0:
        # Se o firmware entrou em hard fault/CPU abort em vez de INIT FAIL,
        # marcamos o teste como BLOCKED para analise manual.
        lower_output = output.lower()
        hard_fault_indicators = [
            "cpu abort",
            "execute code outside ram",
            "hard fault",
            "hardfault",
        ]
        if any(ind in lower_output for ind in hard_fault_indicators):
            pytest.fail(
                "BLOCKED: firmware gerou hard fault/CPU abort em vez de INIT FAIL "
                f"com 48 KB RAM.\nSaida do Renode:\n{tail}"
            )
        raise RuntimeError(
            f"renode-test falhou para arena 48 KB (rc={result.returncode}):\n{tail}"
        )

    if not UART_LOG.exists():
        raise RuntimeError("Log UART nao foi criado pela simulacao QG12")

    log_text = UART_LOG.read_text(errors="replace")
    assert "INIT FAIL" in log_text, (
        "Log UART nao contem INIT FAIL apos simulacao com 48 KB de RAM.\n"
        f"Tail do log:\n{log_text[-2000:]}"
    )
