"""Quality Gate QG13 — Watchdog software para timeout de inferencia no Renode.

Verifica que o firmware detecta travamento (simulado pelo comando UART
WATCHDOG), emite ``WATCHDOG_TIMEOUT`` e reinicia o sistema.
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
ROBOT_FILE = FIRMWARE_ROOT / "renode" / "watchdog.robot"
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


@pytest.mark.qg13
@pytest.mark.slow
def test_watchdog_timeout_resets() -> None:
    """QG13: comando WATCHDOG deve disparar watchdog e logar WATCHDOG_TIMEOUT."""
    _ensure_renode()
    if not ROBOT_FILE.exists():
        pytest.skip(f"Arquivo robot nao encontrado: {ROBOT_FILE}")

    subprocess.run(
        ["make", "-C", str(FIRMWARE_ROOT), "LEWIS_USE_TFLM=1", "RENODE_SIMULATION=1", "stm32f4"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )

    bin_path = FIRMWARE_ROOT / "build" / "stm32f4" / "lewis.bin"
    if not bin_path.exists():
        pytest.skip(
            "Firmware nao compilado. Execute "
            "`make -C firmware RENODE_SIMULATION=1 LEWIS_USE_TFLM=1 stm32f4`"
        )

    if UART_LOG.exists():
        UART_LOG.unlink()

    python_runner = _find_python_with_robot()
    env = os.environ.copy()
    env["PATH"] = str(python_runner.parent) + os.pathsep + env.get("PATH", "")

    cmd = [
        str(RENODE_TEST_BIN),
        "--show-log",
        "-r",
        str(PROJECT_ROOT / "reports" / "renode_watchdog"),
        str(ROBOT_FILE),
    ]

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
        raise RuntimeError(f"renode-test falhou para watchdog (rc={result.returncode}):\n{tail}")

    if not UART_LOG.exists():
        raise RuntimeError("Log UART nao foi criado pela simulacao QG13")

    log_text = UART_LOG.read_text(errors="replace")
    assert "WATCHDOG_TIMEOUT" in log_text, (
        "Log UART nao contem WATCHDOG_TIMEOUT apos simulacao do watchdog.\n"
        f"Tail do log:\n{log_text[-2000:]}"
    )
