"""Quality Gate QG16 — Filtros DSP do firmware vs referencia Python/SciPy.

Compila um pequeno programa C nativo com os mesmos coeficientes SOS gerados
para o firmware, processa sinais sinteticos e compara saida C vs Python com
np.allclose(atol=1e-4).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.dsp_filters import (
    FS,
    bandpass_filter,
    filter_chain,
    notch_filter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRMWARE_SRC = PROJECT_ROOT / "firmware" / "src"
FILTER_C = FIRMWARE_SRC / "dsp" / "filter.c"
FILTER_H = FIRMWARE_SRC / "dsp" / "filter.h"
FILTER_COEFFS_H = FIRMWARE_SRC / "dsp" / "filter_coeffs.h"

C_TEST_SOURCE = r"""
#include "filter.h"
#include "filter_coeffs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int usage(const char* argv0)
{
    fprintf(stderr, "usage: %s <bandpass|notch|chain> <input.bin> <output.bin>\n", argv0);
    return 1;
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        return usage(argv[0]);
    }
    const char* mode = argv[1];
    const char* in_path = argv[2];
    const char* out_path = argv[3];

    FILE* fin = fopen(in_path, "rb");
    if (!fin) {
        fprintf(stderr, "ERRO: nao abriu %s\n", in_path);
        return 1;
    }

    float input[4096];
    size_t n = fread(input, sizeof(float), sizeof(input) / sizeof(float), fin);
    fclose(fin);

    float output[4096];
    lewis_filter_chain_t chain;
    lewis_biquad_state_t bp_state[LEWIS_FILTER_MAX_SECTIONS];
    lewis_biquad_state_t notch_state[LEWIS_FILTER_MAX_SECTIONS];
    lewis_biquad_cascade_t cascade;

    if (strcmp(mode, "chain") == 0) {
        lewis_filter_chain_init(&chain);
        lewis_filter_chain_reset(&chain);
        lewis_filter_chain_process(&chain, input, output, n);
    } else if (strcmp(mode, "bandpass") == 0) {
        lewis_biquad_init(&cascade, LEWIS_BANDPASS_coeffs, LEWIS_BANDPASS_SECTIONS, bp_state);
        lewis_biquad_reset(&cascade);
        lewis_biquad_process_block(&cascade, input, output, n);
    } else if (strcmp(mode, "notch") == 0) {
        lewis_biquad_init(&cascade, LEWIS_NOTCH_coeffs, LEWIS_NOTCH_SECTIONS, notch_state);
        lewis_biquad_reset(&cascade);
        lewis_biquad_process_block(&cascade, input, output, n);
    } else {
        return usage(argv[0]);
    }

    FILE* fout = fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "ERRO: nao criou %s\n", out_path);
        return 1;
    }
    fwrite(output, sizeof(float), n, fout);
    fclose(fout);
    return 0;
}
"""


@pytest.fixture(scope="module")
def native_filter_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compila o programa C de teste dos filtros e retorna o binario."""
    tmpdir = tmp_path_factory.mktemp("dsp_filter_test")
    c_path = tmpdir / "filter_test.c"
    c_path.write_text(C_TEST_SOURCE, encoding="utf-8")
    bin_path = tmpdir / "filter_test"

    cmd = [
        "gcc",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-std=gnu11",
        "-I",
        str(FIRMWARE_SRC),
        "-I",
        str(FIRMWARE_SRC / "dsp"),
        str(c_path),
        str(FILTER_C),
        "-lm",
        "-o",
        str(bin_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return bin_path


def _run_c_filter(bin_path: Path, mode: str, input_arr: np.ndarray) -> np.ndarray:
    """Executa o filtro C em um array float32 e retorna a saida."""
    with tempfile.TemporaryDirectory(prefix="lewis_dsp_") as tmpdir:
        in_path = Path(tmpdir) / "input.bin"
        out_path = Path(tmpdir) / "output.bin"
        in_path.write_bytes(input_arr.astype("<f4").tobytes())
        subprocess.run(
            [str(bin_path), mode, str(in_path), str(out_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        out_bytes = out_path.read_bytes()
    return np.frombuffer(out_bytes, dtype="<f4").copy()


def _synthetic_ecg_with_noise(n: int = 500, seed: int = 42) -> np.ndarray:
    """Gera sinal ECG-like com componentes em banda, fora de banda e 60 Hz."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    # batimento simulado: gaussiana + baixa frequencia (baseline) + linha 60 Hz + ruido
    qrs = 0.8 * np.exp(-200.0 * (t - 0.5) ** 2)
    baseline = 0.05 * np.sin(2.0 * np.pi * 0.3 * t)
    powerline = 0.1 * np.sin(2.0 * np.pi * 60.0 * t)
    noise = 0.02 * rng.normal(size=n)
    return (qrs + baseline + powerline + noise).astype(np.float32)


@pytest.mark.qg16
class TestDSPFilters:
    """Valida implementacao C dos filtros contra referencia Python."""

    def test_bandpass_vs_python(self, native_filter_bin: Path) -> None:
        """QG16.1: bandpass C == bandpass Python (causal SOS)."""
        x = _synthetic_ecg_with_noise(n=500)
        c_out = _run_c_filter(native_filter_bin, "bandpass", x)
        py_out, _ = bandpass_filter(x)
        assert c_out.shape == py_out.shape
        assert np.allclose(c_out, py_out, atol=1e-4)

    def test_notch_vs_python(self, native_filter_bin: Path) -> None:
        """QG16.2: notch C == notch Python (causal SOS)."""
        t = np.arange(500) / FS
        # sinal com componente 60 Hz proeminente
        x = (np.sin(2.0 * np.pi * 10.0 * t) + 0.5 * np.sin(2.0 * np.pi * 60.0 * t)).astype(
            np.float32
        )
        c_out = _run_c_filter(native_filter_bin, "notch", x)
        py_out, _ = notch_filter(x)
        assert c_out.shape == py_out.shape
        assert np.allclose(c_out, py_out, atol=1e-4)

    def test_chain_vs_python(self, native_filter_bin: Path) -> None:
        """QG16.3: cadeia bandpass->notch C == Python."""
        x = _synthetic_ecg_with_noise(n=500)
        c_out = _run_c_filter(native_filter_bin, "chain", x)
        py_out, _, _ = filter_chain(x)
        assert c_out.shape == py_out.shape
        assert np.allclose(c_out, py_out, atol=1e-4)

    def test_bandpass_attenuates_high_freq(self, native_filter_bin: Path) -> None:
        """QG16.4: componente 100 Hz e atenuado pelo bandpass."""
        t = np.arange(500) / FS
        x = np.sin(2.0 * np.pi * 100.0 * t).astype(np.float32)
        y = _run_c_filter(native_filter_bin, "bandpass", x)
        # energia residual de alta frequencia deve ser pequena
        rms_in = np.sqrt(np.mean(x.astype(np.float64) ** 2))
        rms_out = np.sqrt(np.mean(y.astype(np.float64) ** 2))
        assert rms_out < rms_in * 0.3, f"rms_in={rms_in}, rms_out={rms_out}"

    def test_notch_attenuates_60hz(self, native_filter_bin: Path) -> None:
        """QG16.5: notch remove componente 60 Hz."""
        t = np.arange(500) / FS
        x = np.sin(2.0 * np.pi * 60.0 * t).astype(np.float32)
        y = _run_c_filter(native_filter_bin, "notch", x)
        rms_in = np.sqrt(np.mean(x**2))
        rms_out = np.sqrt(np.mean(y**2))
        assert rms_out < rms_in * 0.3, f"rms_in={rms_in}, rms_out={rms_out}"

    def test_filter_state_resettable(self, native_filter_bin: Path) -> None:
        """QG16.6: resetar estado nao altera saida quando chamado antes do bloco."""
        x = _synthetic_ecg_with_noise(n=500)
        y1 = _run_c_filter(native_filter_bin, "chain", x)
        y2 = _run_c_filter(native_filter_bin, "chain", x)
        assert np.allclose(y1, y2, atol=1e-6)
