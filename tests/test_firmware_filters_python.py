"""Testes de equivalencia entre filtros DSP em C e Python (QG16).

Este modulo valida que a cadeia de filtros bandpass -> notch implementada
em C (firmware/src/dsp/filter.c) produz a mesma saida da referencia Python
(tests/fixtures/dsp_filters.py), conforme exigido pelo Quality Gate 16.

Duas abordagens sao usadas:
  1. Verificar o relatorio JSON do harness nativo, que ja compara ambas as
     implementacoes em fixtures conhecidos.
  2. Compilar um runner C leve, gerar um sinal sintetico em Python e comparar
     a saida C com a saida Python ponto a ponto (RMSE < 1e-6).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRMWARE_ROOT = PROJECT_ROOT / "firmware"
HARNESS_SCRIPT = FIRMWARE_ROOT / "scripts" / "run_harness.py"
HARNESS_REPORT = FIRMWARE_ROOT / "test_harness_report.json"
RUNNER_C_SRC = PROJECT_ROOT / "tests" / "fixtures" / "filter_c_runner.c"
RUNNER_BIN_DIR = PROJECT_ROOT / "build" / "tests"
RUNNER_BIN = RUNNER_BIN_DIR / "filter_c_runner"

FS = 500.0


def _run_command(cmd: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """Executa um comando e retorna o resultado, capturando stdout/stderr."""
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )


def _run_harness_native() -> dict[str, Any]:
    """Roda o firmware harness em modo nativo e retorna o relatorio JSON.

    O script `run_harness.py` compila (se necessario) e executa o harness,
    gerando `firmware/test_harness_report.json`.
    """
    if not HARNESS_SCRIPT.exists():
        pytest.skip(f"Script do harness nao encontrado: {HARNESS_SCRIPT}")

    result = _run_command(
        [sys.executable, str(HARNESS_SCRIPT), "--mode", "native"],
        cwd=PROJECT_ROOT,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(
            f"Harness nativo falhou (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
        )

    if not HARNESS_REPORT.exists():
        pytest.fail(f"Relatorio do harness nao gerado: {HARNESS_REPORT}")

    return json.loads(HARNESS_REPORT.read_text(encoding="utf-8"))


def _find_harness_test(report: dict[str, Any], suite: str, name: str) -> dict[str, Any] | None:
    """Localiza o resultado de um teste especifico no relatorio do harness."""
    native = report.get("native") or {}
    for test in native.get("tests", []):
        if test.get("suite") == suite and test.get("name") == name:
            return test
    return None


@pytest.mark.qg16
@pytest.mark.integration
@pytest.mark.slow
def test_harness_filter_chain_vs_python_pass() -> None:
    """Verifica que o harness nativo reporta filter_chain_vs_python como PASS.

    O harness compara a saida da cadeia de filtros C contra a referencia Python
    em multiplos fixtures. Este teste apenas parseia o relatorio JSON.
    """
    report = _run_harness_native()
    test_result = _find_harness_test(report, "DSP", "filter_chain_vs_python")

    if test_result is None:
        pytest.fail("Teste 'DSP filter_chain_vs_python' nao encontrado no relatorio do harness")

    assert (
        test_result.get("status") == "PASS"
    ), f"filter_chain_vs_python falhou: {test_result.get('detail', 'sem detalhe')}"


def _build_filter_c_runner() -> Path:
    """Compila o runner C leve que aplica a cadeia de filtros a um sinal binario.

    O executavel e gerado em `build/tests/filter_c_runner`. Se o binario ja
    existir e for mais recente que os fontes, o build e pulado.
    """
    if not shutil.which("gcc"):
        pytest.skip("gcc nao disponivel para compilar o runner C")
    if not RUNNER_C_SRC.exists():
        pytest.skip(f"Fonte do runner C nao encontrado: {RUNNER_C_SRC}")

    RUNNER_BIN_DIR.mkdir(parents=True, exist_ok=True)

    deps = [
        RUNNER_C_SRC,
        FIRMWARE_ROOT / "src" / "dsp" / "filter.c",
        FIRMWARE_ROOT / "src" / "dsp" / "filter.h",
        FIRMWARE_ROOT / "src" / "dsp" / "filter_coeffs.h",
    ]
    if RUNNER_BIN.exists():
        bin_mtime = RUNNER_BIN.stat().st_mtime
        if all(bin_mtime > dep.stat().st_mtime for dep in deps if dep.exists()):
            return RUNNER_BIN

    cmd = [
        "gcc",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-std=gnu11",
        f"-I{FIRMWARE_ROOT / 'src'}",
        "-o",
        str(RUNNER_BIN),
        str(RUNNER_C_SRC),
        str(FIRMWARE_ROOT / "src" / "dsp" / "filter.c"),
        "-lm",
    ]
    result = _run_command(cmd, cwd=PROJECT_ROOT, timeout=60)
    if result.returncode != 0:
        pytest.fail(f"Falha ao compilar runner C:\n{result.stdout}\n{result.stderr}")

    return RUNNER_BIN


def _synthetic_ecg_signal(n_samples: int = 500, seed: int = 42) -> np.ndarray:
    """Gera um sinal sintetico de ECG para validar a cadeia de filtros.

    O sinal contem:
      - componente DC (atenuada pelo high-pass de 0.5 Hz);
      - batimento em ~1 Hz (componente util);
      - interferencia de 60 Hz (atenuada pelo notch);
      - ruido branco de baixa amplitude.

    Returns
    -------
    np.ndarray
        Sinal float32 com `n_samples` amostras a 500 Hz.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float32) / np.float32(FS)

    dc = np.float32(0.5)
    beat = np.float32(0.8) * np.sin(2.0 * np.pi * np.float32(1.2) * t)
    interference = np.float32(0.3) * np.sin(2.0 * np.pi * np.float32(60.0) * t)
    noise = rng.normal(loc=0.0, scale=0.01, size=n_samples).astype(np.float32)

    return dc + beat + interference + noise


@pytest.mark.qg16
def test_filter_chain_rmse_vs_c() -> None:
    """Compara saida do filtro C vs Python ponto a ponto (QG16).

    Gera um sinal sintetico, aplica a cadeia bandpass -> notch em Python e em C,
    e verifica que o RMSE entre as saidas e inferior a 1e-6.
    """
    runner = _build_filter_c_runner()

    # Importa a referencia Python equivalente a implementacao C.
    sys.path.insert(0, str(PROJECT_ROOT))
    from tests.fixtures.dsp_filters import filter_chain as py_filter_chain

    signal_in = _synthetic_ecg_signal(n_samples=500, seed=42)

    # Saida Python: bandpass -> notch.
    signal_py, _, _ = py_filter_chain(signal_in)
    signal_py = signal_py.astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_bin = tmp_path / "input.bin"
        output_bin = tmp_path / "output.bin"

        input_bin.write_bytes(signal_in.tobytes())

        result = _run_command(
            [str(runner), str(input_bin), str(output_bin)],
            cwd=PROJECT_ROOT,
            timeout=10,
        )
        if result.returncode != 0:
            pytest.fail(f"Runner C falhou:\n{result.stdout}\n{result.stderr}")

        if not output_bin.exists():
            pytest.fail("Arquivo de saida do runner C nao foi gerado")

        signal_c = np.frombuffer(output_bin.read_bytes(), dtype=np.float32)

    assert (
        signal_c.shape == signal_py.shape
    ), f"Shape divergente: Python {signal_py.shape} vs C {signal_c.shape}"

    rmse = float(np.sqrt(np.mean((signal_py - signal_c) ** 2)))
    assert rmse < 1e-6, f"RMSE entre filtros Python e C e {rmse:.3e} (limite 1e-6)"
