"""Quality Gate QG11 — Injecao de falha via periferico SPI dummy no Renode.

Verifica que o firmware reporta erro quando recebe um frame UART invalido
apos carregar um periferico SPI dummy.

Nota sobre fallback:
  Renode 1.15.3 pode nao suportar anexao direta de perifericos SPI a partir
de testes Robot (os comandos ``machine LoadPeripheralFromFile``,
``machine LoadPeripheral`` e injecao via Python costumam falhar nessa
versao). Quando isso ocorre, o teste prossegue injetando corrupcao de dados
pela UART e valida o tratamento graceful de erros (mensagem ``ERRO`` no log).
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
ROBOT_FILE = FIRMWARE_ROOT / "renode" / "fault_injection.robot"
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


@pytest.mark.qg11
@pytest.mark.slow
def test_fault_injection_dummy_spi() -> None:
    """QG11: frame UART invalido apos SPI dummy deve gerar 'ERRO'."""
    _ensure_renode()
    if not ROBOT_FILE.exists():
        pytest.skip(f"Arquivo robot nao encontrado: {ROBOT_FILE}")

    # Recompila o firmware garantindo RENODE_SIMULATION=1, pois um binario
    # antigo sem esse flag pode fazer o teste falhar.
    subprocess.run(
        [
            "make",
            "-C",
            str(FIRMWARE_ROOT),
            "LEWIS_USE_TFLM=1",
            "RENODE_SIMULATION=1",
            "stm32f4",
        ],
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
        str(PROJECT_ROOT / "reports" / "renode_fault_injection"),
        str(ROBOT_FILE),
    ]

    # Executa a partir de firmware/ para manter a convencao de CWD dos
    # scripts .resc (os caminhos do binario sao absolutos).
    result = subprocess.run(
        cmd,
        cwd=FIRMWARE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
        env=env,
    )

    if result.returncode != 0:
        tail = result.stdout.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            f"renode-test falhou para fault injection (rc={result.returncode}):\n{tail}"
        )
