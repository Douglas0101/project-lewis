"""Quality Gate QG10 — Fidelidade da inferencia UART no Renode.

Para cada batimento de ground-truth, o teste:
  1. Envia o frame ``<500×float32 little-endian>`` pela UART do STM32F4 emulado.
  2. Captura a resposta binaria ``<5×int8>`` do firmware.
  3. Compara a saida do firmware com a saida esperada do interpretador TFLite
     Python usando similaridade de cosseno e MAE sobre os valores dequantizados.

O teste exige que o firmware tenha sido compilado com:
    make -C firmware RENODE_SIMULATION=1 LEWIS_USE_TFLM=1 stm32f4
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRMWARE_ROOT = PROJECT_ROOT / "firmware"
RENODE_DIR = FIRMWARE_ROOT / "tools" / "renode-1.15.3"
RENODE_TEST_BIN = RENODE_DIR / "renode-test"
ROBOT_FILE = FIRMWARE_ROOT / "renode" / "fidelity.robot"
GROUND_TRUTH_DIR = PROJECT_ROOT / "tests" / "ground_truth"
UART_LOG = Path("/tmp/renode_lewis_uart.log")


def _load_quant_params():
    """Carrega parametros de quantizacao do modelo a partir do JSON exportado."""
    path = PROJECT_ROOT / "models" / "quantized" / "quantization_params.json"
    if not path.exists():
        raise FileNotFoundError(f"Parametros de quantizacao nao encontrados: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["input"], data["output"]


INPUT_QUANT, OUTPUT_QUANT = _load_quant_params()
INPUT_SCALE = INPUT_QUANT["scale"]
INPUT_ZERO_POINT = INPUT_QUANT["zero_point"]
OUTPUT_SCALE = OUTPUT_QUANT["scale"]
OUTPUT_ZERO_POINT = OUTPUT_QUANT["zero_point"]

# Limiares QG10.
MIN_COSINE_SIMILARITY = 0.99
MAX_MAE = 0.01

# Format constants for the UART response frame <5×int8>.
RESPONSE_LEN = 5
START_BYTE = ord("<")
END_BYTE = ord(">")


def _ensure_renode() -> None:
    """Garante que o Renode esteja disponivel em firmware/tools."""
    if RENODE_TEST_BIN.exists():
        return
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


def _build_resc_content(bin_path: Path, uart_log: Path) -> str:
    """Gera script .resc com path absoluto do firmware (requerido por renode-test)."""
    return f"""using sysbus
mach create "stm32f4discovery-lewis"
include @scripts/single-node/stm32f4_discovery.resc
sysbus LoadBinary @{bin_path} 0x08000000
cpu VectorTableOffset 0x08000000
$uart_log?="{uart_log}"
sysbus.uart4 CreateFileBackend $uart_log true
start
"""


def _run_robot_for_beat(idx: int) -> bytes:
    """Executa fidelity.robot para o batimento ``idx`` e retorna o log UART bruto."""
    _ensure_renode()
    if not RENODE_TEST_BIN.exists():
        pytest.skip(f"renode-test nao encontrado em {RENODE_TEST_BIN}")
    if not ROBOT_FILE.exists():
        pytest.skip(f"Arquivo robot nao encontrado: {ROBOT_FILE}")

    bin_path = FIRMWARE_ROOT / "build" / "stm32f4" / "lewis.bin"
    if not bin_path.exists():
        pytest.skip(
            "Firmware nao compilado. Execute "
            "`make -C firmware RENODE_SIMULATION=1 LEWIS_USE_TFLM=1 stm32f4`"
        )

    input_path = GROUND_TRUTH_DIR / f"ecg_input_{idx:02d}.bin"
    if not input_path.exists():
        pytest.skip(
            f"Ground-truth de entrada nao encontrado: {input_path}. "
            "Execute `python tests/ground_truth/generate_ground_truth.py`"
        )

    # Cada batimento usa seu proprio arquivo de log para evitar condicoes de
    # corrida e dados residuais entre execucoes.
    uart_log = Path(f"/tmp/renode_lewis_fidelity_{idx:02d}.log")
    if uart_log.exists():
        uart_log.unlink()

    with tempfile.TemporaryDirectory(prefix="lewis_fidelity_") as tmpdir:
        resc_path = Path(tmpdir) / "run.resc"
        resc_path.write_text(_build_resc_content(bin_path, uart_log), encoding="utf-8")

        python_runner = _find_python_with_robot()
        env = os.environ.copy()
        env["PATH"] = str(python_runner.parent) + os.pathsep + env.get("PATH", "")

        cmd = [
            str(RENODE_TEST_BIN),
            "--show-log",
            "-r",
            tmpdir,
            "--variable",
            f"RESC:{resc_path}",
            "--variable",
            f"INPUT_PATH:{input_path}",
            "--variable",
            f"UART_LOG:{uart_log}",
            str(ROBOT_FILE),
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=300,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"renode-test excedeu timeout para beat {idx}") from exc

        if result.returncode != 0:
            tail = result.stdout.decode("utf-8", errors="replace")[-3000:]
            raise RuntimeError(
                f"renode-test falhou para beat {idx} (rc={result.returncode}):\n{tail}"
            )

    if not uart_log.exists():
        raise RuntimeError(f"Log UART nao foi criado: {uart_log}")

    return uart_log.read_bytes()


def _extract_response_frame(log_bytes: bytes) -> np.ndarray:
    """Extrai a ultima resposta ``<5×int8>`` do log binario da UART."""
    # Procura pelo ultimo '<' que precede exatamente 5 bytes e '>'.
    for start in range(len(log_bytes) - 1, -1, -1):
        if log_bytes[start] == START_BYTE:
            end = start + RESPONSE_LEN + 1  # indice do terminador '>'
            if end < len(log_bytes) and log_bytes[end] == END_BYTE:
                return np.frombuffer(log_bytes[start + 1 : end], dtype=np.int8)
    raise ValueError("Frame de resposta '<5×int8>' nao encontrado no log UART")


def _dequantize(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Dequantiza array int8 para float32."""
    return (values.astype(np.float32) - zero_point) * scale


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity entre dois vetores 1D."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / norm)


@pytest.mark.qg10
@pytest.mark.slow
class TestFidelity:
    @pytest.fixture(scope="class", autouse=True)
    def _build_renode_firmware(self) -> None:
        """QG10 exige binario compilado com RENODE_SIMULATION=1."""
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

    @pytest.mark.parametrize("idx", [0, 1, 2, 3, 4])
    def test_beat_fidelity(self, idx: int) -> None:
        """QG10: saida do firmware deve ser proxima a ground-truth Python."""
        expected_path = GROUND_TRUTH_DIR / f"expected_output_{idx:02d}.bin"
        if not expected_path.exists():
            pytest.skip(
                f"Ground-truth de saida nao encontrado: {expected_path}. "
                "Execute `python tests/ground_truth/generate_ground_truth.py`"
            )

        expected_int8 = np.fromfile(expected_path, dtype=np.int8)
        assert expected_int8.shape == (5,), f"Formato inesperado: {expected_int8.shape}"

        log_bytes = _run_robot_for_beat(idx)
        firmware_int8 = _extract_response_frame(log_bytes)
        assert firmware_int8.shape == (5,), f"Resposta inesperada: {firmware_int8.shape}"

        expected_f32 = _dequantize(expected_int8, OUTPUT_SCALE, OUTPUT_ZERO_POINT)
        firmware_f32 = _dequantize(firmware_int8, OUTPUT_SCALE, OUTPUT_ZERO_POINT)

        cosine = _cosine_similarity(expected_f32, firmware_f32)
        mae = float(np.mean(np.abs(expected_f32 - firmware_f32)))

        assert cosine > MIN_COSINE_SIMILARITY, (
            f"Beat {idx}: cosine similarity {cosine:.6f} <= "
            f"{MIN_COSINE_SIMILARITY} (expected={expected_int8.tolist()}, "
            f"firmware={firmware_int8.tolist()})"
        )
        assert mae < MAX_MAE, (
            f"Beat {idx}: MAE {mae:.6f} >= {MAX_MAE} "
            f"(expected={expected_int8.tolist()}, firmware={firmware_int8.tolist()})"
        )
