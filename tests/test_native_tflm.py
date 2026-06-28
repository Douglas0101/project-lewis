"""Verifica que o firmware nativo usa TFLM real (nao stub) e bate com Python.

R1: a versao nativa deve ser vinculada com a biblioteca TFLM host e produzir
os mesmos resultados do interpretador Python de referencia (BUILTIN_REF).

No pipeline two-stage v2.0 o firmware responde com 3 int8 correspondentes
as classes S/V/F do Estagio 2 (zeros quando o Estagio 1 classifica como N).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from tests.fixtures.adc_stub import adc_stub_get_beat
from tests.fixtures.dsp_filters import filter_chain
from tests.fixtures.normalizer import zscore_normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRMWARE_DIR = PROJECT_ROOT / "firmware"
NATIVE_BIN = FIRMWARE_DIR / "build" / "native" / "lewis"
STAGE1_TFLITE = PROJECT_ROOT / "models" / "quantized" / "stage1_int8_v2.0.tflite"
STAGE2_TFLITE = PROJECT_ROOT / "models" / "quantized" / "stage2_int8_v2.0.tflite"
QPARAMS_PATH = PROJECT_ROOT / "models" / "quantized" / "quantization_params.json"


NUM_BEATS = 3
DEQUANT_ATOL = 1e-5
STUB_MARKER = "[inference] STUB"


def _read_quantization_params() -> dict:
    return json.loads(QPARAMS_PATH.read_text())


def _build_native_tflm() -> None:
    subprocess.run(
        ["make", "-C", str(FIRMWARE_DIR), "native-tflm"],
        check=True,
    )


def _run_native(timeout: float = 3.0) -> str:
    try:
        result = subprocess.run(
            [str(NATIVE_BIN)],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        return result.stdout
    except subprocess.TimeoutExpired as exc:
        # O firmware entra em loop infinito apos os beats; usamos o timeout
        # para encerrar a execucao e analisar a saida produzida.
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode()
        return stdout


def _parse_native_outputs(stdout: str) -> list[np.ndarray]:
    pattern = re.compile(
        r"Beat\s+(?P<idx>\d+):\s+\d+\s+ms\s*\(\d+\s+us\)\s*,\s*class=\w+\s*,\s*"
        r"output\s*=\s*\[(?P<values>[-\d,\s]+)\]"
    )
    outputs = []
    for match in pattern.finditer(stdout):
        values = [int(v.strip()) for v in match.group("values").split(",")]
        outputs.append(np.array(values, dtype=np.int8))
    return outputs


def _quantize_input(beat_float: np.ndarray, in_scale: float, in_zero_point: int) -> np.ndarray:
    """Reproduz a quantizacao int8 do firmware."""
    rounded = np.where(
        beat_float / in_scale >= 0.0,
        np.floor(beat_float / in_scale + 0.5),
        np.ceil(beat_float / in_scale - 0.5),
    )
    return np.clip(rounded.astype(np.int32) + in_zero_point, -128, 127).astype(np.int8)


def _run_python_reference(num_beats: int) -> list[np.ndarray]:
    """Executa o pipeline two-stage no Python usando BUILTIN_REF.

    Retorna os outputs dequantizados em float32 para comparacao justa com o
    firmware nativo (TFLM tambem dequantiza internamente para float32).
    """
    stage1 = tf.lite.Interpreter(
        model_path=str(STAGE1_TFLITE),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    stage1.allocate_tensors()
    stage1_input = stage1.get_input_details()[0]
    stage1_output = stage1.get_output_details()[0]

    stage2 = tf.lite.Interpreter(
        model_path=str(STAGE2_TFLITE),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    stage2.allocate_tensors()
    stage2_input = stage2.get_input_details()[0]
    stage2_output = stage2.get_output_details()[0]

    qparams = _read_quantization_params()
    in_scale = qparams["input"]["scale"]
    in_zero_point = qparams["input"]["zero_point"]
    out_scale = qparams["output"]["scale"]
    out_zero_point = qparams["output"]["zero_point"]

    outputs = []
    for idx in range(num_beats):
        beat = adc_stub_get_beat(idx)
        # Reproduz o pipeline DSP do firmware: dequantiza -> filtra -> zscore -> quantiza.
        beat_float = (beat.astype(np.float32) - in_zero_point) * in_scale
        beat_filtered, _, _ = filter_chain(beat_float)
        normalized = zscore_normalize(beat_filtered)
        beat_quantized = _quantize_input(normalized, in_scale, in_zero_point)
        tensor = beat_quantized.reshape(stage1_input["shape"]).astype(np.int8)

        stage1.set_tensor(stage1_input["index"], tensor)
        stage1.invoke()
        stage1_out = stage1.get_tensor(stage1_output["index"])[0].copy()

        if int(np.argmax(stage1_out)) == 1:
            stage2.set_tensor(stage2_input["index"], tensor)
            stage2.invoke()
            final_out = stage2.get_tensor(stage2_output["index"])[0].copy()
        else:
            final_out = np.zeros(stage2_output["shape"][1], dtype=np.int8)

        outputs.append((final_out.astype(np.float32) - out_zero_point) * out_scale)
    return outputs


def _dequantize(out_int8: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (out_int8.astype(np.float32) - zero_point) * scale


@pytest.mark.qg7
@pytest.mark.slow
class TestNativeTflm:
    @pytest.fixture(scope="class", autouse=True)
    def _build(self) -> None:
        _build_native_tflm()

    def test_binary_does_not_contain_stub_marker(self) -> None:
        binary = NATIVE_BIN.read_bytes()
        assert STUB_MARKER.encode() not in binary, "binario nativo contem string de stub"

    def test_runtime_does_not_emit_stub_marker(self) -> None:
        stdout = _run_native()
        assert STUB_MARKER not in stdout, "firmware nativo executou como stub"

    def test_outputs_match_python_reference(self) -> None:
        stdout = _run_native()
        fw_outputs = _parse_native_outputs(stdout)
        assert len(fw_outputs) == NUM_BEATS, f"esperava {NUM_BEATS} beats, obteve {len(fw_outputs)}"

        qparams = _read_quantization_params()
        out_scale = qparams["output"]["scale"]
        out_zero_point = qparams["output"]["zero_point"]

        py_outputs = _run_python_reference(len(fw_outputs))
        for idx, (fw_out, py_out) in enumerate(zip(fw_outputs, py_outputs)):
            fw_float = _dequantize(fw_out, out_scale, out_zero_point)
            assert np.allclose(fw_float, py_out, atol=DEQUANT_ATOL), (
                f"saida do beat {idx} diverge do Python de referencia: "
                f"fw={fw_float.tolist()}, py={py_out.tolist()}"
            )
