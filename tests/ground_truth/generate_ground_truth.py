#!/usr/bin/env python3
"""Gera dataset versionado de ground-truth para teste de fidelidade QG10.

Para cada batimento sintetico (``idx``):
  1. Obtem 500 amostras int8 do ``tests.fixtures.adc_stub.adc_stub_get_beat``.
  2. Dequantiza int8 -> float32 com os parametros de entrada do modelo.
  3. Salva o sinal float32 little-endian em ``ecg_input_{idx:02d}.bin``.
  4. Re-quantiza float32 -> int8 usando a mesma formula do firmware.
  5. Executa inferencia no interpretador TFLite Python (resolver BUILTIN_REF).
  6. Salva os 5 logits/probabilidades int8 em ``expected_output_{idx:02d}.bin``.

O dataset versionado permite reproduzir o teste de fidelidade (QG10) de
forma deterministica, comparando a saida do firmware em Renode com a saida
de referencia gerada pelo interpretador Python.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Garante que o projeto raiz esteja no path para importar fixtures.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.fixtures.adc_stub import adc_stub_get_beat  # noqa: E402
from tests.fixtures.dsp_filters import filter_chain  # noqa: E402


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

MODEL_TFLITE = PROJECT_ROOT / "models" / "quantized" / "model_int8.tflite"
GROUND_TRUTH_DIR = Path(__file__).resolve().parent


def dequantize_int8_to_float32(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Dequantiza array int8 para float32."""
    return (values.astype(np.float32) - zero_point) * scale


def quantize_float32_to_int8(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Quantiza array float32 para int8 com o mesmo arredondamento do firmware C.

    O firmware arredonda para longe do zero (truncamento com +/- 0.5).
    """
    normalized = values / scale
    rounded = np.where(normalized >= 0.0, np.floor(normalized + 0.5), np.ceil(normalized - 0.5))
    quantized = rounded + zero_point
    return np.clip(quantized, -128, 127).astype(np.int8)


def save_float32_bin(path: Path, data: np.ndarray) -> None:
    """Salva array float32 como little-endian binario puro."""
    path.write_bytes(data.astype("<f4").tobytes())


def save_int8_bin(path: Path, data: np.ndarray) -> None:
    """Salva array int8 como binario puro."""
    path.write_bytes(data.astype(np.int8).tobytes())


def load_python_interpreter() -> tf.lite.Interpreter:
    """Carrega o modelo TFLite com resolver de referencia (BUILTIN_REF)."""
    if not MODEL_TFLITE.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {MODEL_TFLITE}")

    interpreter = tf.lite.Interpreter(
        model_path=str(MODEL_TFLITE),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    return interpreter


def generate_ground_truth(num_beats: int = 5) -> None:
    """Gera arquivos binarios de entrada e saida esperada para ``num_beats``."""
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    interpreter = load_python_interpreter()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    for idx in range(num_beats):
        beat_int8 = adc_stub_get_beat(idx)

        # 1. Dequantiza int8 -> float32 (mesmo que o host enviaria pela UART).
        beat_float32 = dequantize_int8_to_float32(beat_int8, INPUT_SCALE, INPUT_ZERO_POINT)

        # Salva entrada bruta (nao filtrada) — o firmware aplica filtros em runtime.
        input_path = GROUND_TRUTH_DIR / f"ecg_input_{idx:02d}.bin"
        save_float32_bin(input_path, beat_float32)

        # 2. Aplica pipeline DSP causal bandpass -> notch (igual ao firmware)
        #    para gerar a saida esperada.
        beat_filtered, _, _ = filter_chain(beat_float32)

        # 3. Re-quantiza float32 -> int8 com a mesma formula do firmware.
        input_quantized = quantize_float32_to_int8(beat_filtered, INPUT_SCALE, INPUT_ZERO_POINT)

        # 4. Inferencia TFLite Python com resolver de referencia.
        tensor = input_quantized.reshape(input_details["shape"]).astype(np.int8)
        interpreter.set_tensor(input_details["index"], tensor)
        interpreter.invoke()
        output_int8 = interpreter.get_tensor(output_details["index"])[0].copy()

        output_path = GROUND_TRUTH_DIR / f"expected_output_{idx:02d}.bin"
        save_int8_bin(output_path, output_int8)

        print(
            f"[ground-truth] idx={idx:02d} input={input_path} "
            f"output={output_path} out={output_int8.tolist()}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera dataset de ground-truth para teste de fidelidade QG10."
    )
    parser.add_argument(
        "--num-beats",
        type=int,
        default=5,
        help="Numero de batimentos sinteticos a gerar (padrao: 5).",
    )
    args = parser.parse_args()

    generate_ground_truth(args.num_beats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
