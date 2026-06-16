"""Quality Gate QG17 — Fidelidade do pipeline DSP filtrado do firmware.

Verifica que o caminho completo do firmware:
    ADC stub int8 -> dequantiza -> bandpass -> notch -> quantiza -> int8
produz o mesmo resultado do referencial Python equivalente.

O teste compila um pequeno programa C nativo que replica exatamente as
funcoes de quantizacao/desquantizacao e filtros do firmware, e compara a
saida int8 final com a referencia Python usando similaridade de cosseno e MAE.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.adc_stub import adc_stub_get_beat
from tests.fixtures.dsp_filters import filter_chain

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRMWARE_SRC = PROJECT_ROOT / "firmware" / "src"
QUANT_PARAMS_PATH = PROJECT_ROOT / "models" / "quantized" / "quantization_params.json"

C_DSP_PIPELINE_SOURCE = r"""
#include "dsp/adc_stub.h"
#include "dsp/filter.h"
#include "dsp/filter_coeffs.h"
#include "ml/quantization_params.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define INPUT_LEN 500

static void dequantize(const int8_t* in, float* out, size_t n)
{
    const float scale = LEWIS_QUANTIZATION_PARAMS_INPUT_SCALE;
    const int32_t zp = LEWIS_QUANTIZATION_PARAMS_INPUT_ZERO_POINT;
    for (size_t i = 0; i < n; ++i) {
        out[i] = ((float)in[i] - (float)zp) * scale;
    }
}

static void quantize(const float* in, int8_t* out, size_t n)
{
    const float scale = LEWIS_QUANTIZATION_PARAMS_INPUT_SCALE;
    const int32_t zp = LEWIS_QUANTIZATION_PARAMS_INPUT_ZERO_POINT;
    for (size_t i = 0; i < n; ++i) {
        float normalized = in[i] / scale;
        int32_t q = (int32_t)(normalized + (normalized >= 0.0f ? 0.5f : -0.5f)) + zp;
        if (q > 127) q = 127;
        else if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <idx> <output.bin>\n", argv[0]);
        return 1;
    }
    uint32_t idx = (uint32_t)atoi(argv[1]);
    const char* out_path = argv[2];

    int8_t raw[INPUT_LEN];
    float frame[INPUT_LEN];
    int8_t quantized[INPUT_LEN];
    lewis_filter_chain_t chain;

    lewis_adc_stub_get_beat(idx, raw);
    dequantize(raw, frame, INPUT_LEN);
    lewis_filter_chain_init(&chain);
    lewis_filter_chain_reset(&chain);
    lewis_filter_chain_process(&chain, frame, frame, INPUT_LEN);
    quantize(frame, quantized, INPUT_LEN);

    FILE* fout = fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "ERRO: nao criou %s\n", out_path);
        return 1;
    }
    fwrite(quantized, sizeof(int8_t), INPUT_LEN, fout);
    fclose(fout);
    return 0;
}
"""


def _load_quant_params():
    if not QUANT_PARAMS_PATH.exists():
        pytest.skip(f"Parametros de quantizacao nao encontrados: {QUANT_PARAMS_PATH}")
    data = json.loads(QUANT_PARAMS_PATH.read_text(encoding="utf-8"))
    return data["input"]


def _quantize(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Replica a quantizacao do firmware (arredondamento para longe do zero)."""
    normalized = values / scale
    rounded = np.where(normalized >= 0.0, np.floor(normalized + 0.5), np.ceil(normalized - 0.5))
    quantized = rounded + zero_point
    return np.clip(quantized, -128, 127).astype(np.int8)


@pytest.fixture(scope="module")
def dsp_pipeline_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compila programa C com o pipeline DSP completo do firmware."""
    tmpdir = tmp_path_factory.mktemp("dsp_pipeline_test")
    c_path = tmpdir / "dsp_pipeline.c"
    c_path.write_text(C_DSP_PIPELINE_SOURCE, encoding="utf-8")
    bin_path = tmpdir / "dsp_pipeline"

    cmd = [
        "gcc",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-std=gnu11",
        "-I",
        str(FIRMWARE_SRC),
        str(c_path),
        str(FIRMWARE_SRC / "dsp" / "adc_stub.c"),
        str(FIRMWARE_SRC / "dsp" / "filter.c"),
        "-lm",
        "-o",
        str(bin_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return bin_path


def _run_c_dsp_pipeline(bin_path: Path, idx: int) -> np.ndarray:
    """Executa o pipeline DSP C para o batimento idx e retorna int8[500]."""
    with tempfile.TemporaryDirectory(prefix="lewis_dsp_fidelity_") as tmpdir:
        out_path = Path(tmpdir) / "output.bin"
        subprocess.run(
            [str(bin_path), str(idx), str(out_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        out_bytes = out_path.read_bytes()
    return np.frombuffer(out_bytes, dtype=np.int8).copy()


def _python_filtered_beat(idx: int) -> np.ndarray:
    """Replica o pipeline DSP Python: adc_stub -> dequant -> filter -> quant."""
    q = _load_quant_params()
    scale = q["scale"]
    zero_point = q["zero_point"]

    raw = adc_stub_get_beat(idx)
    frame = (raw.astype(np.float32) - zero_point) * scale
    filtered, _, _ = filter_chain(frame)
    return _quantize(filtered, scale, zero_point)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a.astype(np.float64)) * np.linalg.norm(b.astype(np.float64))
    if norm == 0.0:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / norm)


@pytest.mark.qg17
class TestDSPPipelineFidelity:
    """Valida fidelidade do pipeline DSP filtrado (C vs Python)."""

    MIN_COSINE_SIMILARITY = 0.99
    MAX_MAE = 0.01

    @pytest.mark.parametrize("idx", [0, 1, 2, 3, 4])
    def test_filtered_beat_fidelity(self, dsp_pipeline_bin: Path, idx: int) -> None:
        """QG17: saida int8 do pipeline filtrado C == referencia Python."""
        c_out = _run_c_dsp_pipeline(dsp_pipeline_bin, idx)
        py_out = _python_filtered_beat(idx)

        assert c_out.shape == py_out.shape == (500,), f"Shape inesperado: {c_out.shape}"

        cosine = _cosine_similarity(c_out, py_out)
        mae = float(np.mean(np.abs(c_out.astype(np.float64) - py_out.astype(np.float64))))

        assert (
            cosine > self.MIN_COSINE_SIMILARITY
        ), f"Beat {idx}: cosine similarity {cosine:.6f} <= {self.MIN_COSINE_SIMILARITY}"
        assert mae < self.MAX_MAE, f"Beat {idx}: MAE {mae:.6f} >= {self.MAX_MAE}"
