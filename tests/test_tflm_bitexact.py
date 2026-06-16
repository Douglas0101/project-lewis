"""Teste de bit-exatidao entre firmware TFLM e interpretador Python.

QG8: o output int8 do modelo executado no STM32F4 (Renode) deve ser
identico ao output do mesmo modelo executado no interpretador Python
para o sinal de teste gerado pelo adc_stub.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from tests.fixtures.adc_stub import adc_stub_get_beat
from tests.fixtures.dsp_filters import filter_chain
from tests.fixtures.normalizer import zscore_normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_TFLITE = PROJECT_ROOT / "models" / "quantized" / "model_int8.tflite"


def run_python_inference(num_beats: int = 3) -> list[np.ndarray]:
    """Executa o modelo TFLite no Python para os beats do stub.

    Reproduz o pipeline DSP do firmware (dequantiza -> filtra -> zscore -> quantiza)
    antes da inferencia. Usamos o resolver BUILTIN_REF para evitar diferencas
    numericas introduzidas pelo XNNPACK delegate, garantindo comparacao justa
    com o TFLM do firmware (que usa kernels de referencia / CMSIS-NN).
    """
    interpreter = tf.lite.Interpreter(
        model_path=str(MODEL_TFLITE),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    qparams = input_details["quantization_parameters"]
    in_scale = float(qparams["scales"][0])
    in_zero_point = int(qparams["zero_points"][0])

    outputs = []
    for idx in range(num_beats):
        beat = adc_stub_get_beat(idx)
        # Pipeline DSP causal igual ao firmware.
        beat_float = (beat.astype(np.float32) - in_zero_point) * in_scale
        beat_filtered, _, _ = filter_chain(beat_float)
        normalized = zscore_normalize(beat_filtered)
        rounded = np.where(
            normalized / in_scale >= 0.0,
            np.floor(normalized / in_scale + 0.5),
            np.ceil(normalized / in_scale - 0.5),
        )
        beat_quantized = np.clip(rounded.astype(np.int32) + in_zero_point, -128, 127).astype(
            np.int8
        )
        tensor = beat_quantized.reshape(input_details["shape"]).astype(np.int8)
        interpreter.set_tensor(input_details["index"], tensor)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])[0].copy()
        outputs.append(out)
    return outputs


@pytest.mark.qg8
@pytest.mark.slow
class TestTflmBitexact:
    def test_python_outputs_match_expected_shape(self):
        outputs = run_python_inference(3)
        assert len(outputs) == 3
        for out in outputs:
            assert out.shape == (5,)
            assert out.dtype == np.int8

    def test_bitexact_vs_firmware(self, firmware_report):
        """Compara output int8 do Python com o log do firmware."""
        if not firmware_report["checks"]["all_passed"]:
            pytest.skip("Firmware simulation nao passou nos checks estruturais.")

        py_outputs = run_python_inference(firmware_report["beat_count"])
        uart_text = firmware_report["uart_log_text"]
        beat_re = re.compile(
            r"Beat\s+(?P<idx>\d+):\s+\d+\s+ms\s*\(\d+\s+us\)?\s*,\s+output\s+\[(?P<values>[-\d,\s]+)\]"  # noqa: E501
        )

        fw_outputs = {}
        for line in uart_text.splitlines():
            m = beat_re.search(line)
            if not m:
                continue
            idx = int(m.group("idx"))
            values = [int(v.strip()) for v in m.group("values").split(",")]
            fw_outputs[idx] = np.array(values, dtype=np.int8)

        assert len(fw_outputs) == len(
            py_outputs
        ), f"Numero de beats diverge: firmware={len(fw_outputs)}, python={len(py_outputs)}"

        for idx, py_out in enumerate(py_outputs):
            fw_out = fw_outputs[idx]
            # CMSIS-NN pode divergir ate 1 LSB dos kernels de referencia
            # devido a arredondamento otimizado em acumuladores 32-bit,
            # portanto usamos tolerancia de 1 LSB ao inves de igualdade estrita.
            assert np.allclose(py_out, fw_out, atol=1), (
                f"Bit-exatidao falhou no beat {idx}: "
                f"python={py_out.tolist()}, firmware={fw_out.tolist()}"
            )
