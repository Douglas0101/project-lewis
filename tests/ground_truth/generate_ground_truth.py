#!/usr/bin/env python3
"""Gera dataset versionado de ground-truth para teste de fidelidade QG10 (two-stage v2.0).

Para cada batimento sintetico (``idx``):
  1. Obtem 500 amostras int8 do ``tests.fixtures.adc_stub.adc_stub_get_beat``.
  2. Dequantiza int8 -> float32 com os parametros de entrada do modelo.
  3. Salva o sinal float32 little-endian em ``ecg_input_{idx:02d}.bin``.
  4. Aplica pipeline DSP (bandpass/notch/zscore) e re-quantiza float32 -> int8.
  5. Executa inferencia two-stage no interpretador TFLite Python (resolver BUILTIN_REF):
     - Estagio 1: N (0) vs Anormal (1).
     - Se Anormal, executa Estagio 2 (S/V/F) e salva os 3 logits int8.
     - Se N, salva 3 zeros (S/V/F ausentes).
  6. Salva a saida final de 3 int8 em ``expected_output_{idx:02d}.bin``.

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
from tests.fixtures.normalizer import zscore_normalize  # noqa: E402


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

STAGE1_TFLITE = PROJECT_ROOT / "models" / "quantized" / "stage1_int8_v2.0.tflite"
STAGE2_TFLITE = PROJECT_ROOT / "models" / "quantized" / "stage2_int8_v2.0.tflite"
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


def load_python_interpreter(model_path: Path) -> tf.lite.Interpreter:
    """Carrega o modelo TFLite com resolver de referencia (BUILTIN_REF)."""
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    return interpreter


def argmax_int8(values: np.ndarray) -> int:
    """Argmax compativel com a implementacao C."""
    return int(np.argmax(values))


def generate_ground_truth(num_beats: int = 5) -> None:
    """Gera arquivos binarios de entrada e saida esperada para ``num_beats``."""
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    stage1 = load_python_interpreter(STAGE1_TFLITE)
    stage2 = load_python_interpreter(STAGE2_TFLITE)

    stage1_input_details = stage1.get_input_details()[0]
    stage1_output_details = stage1.get_output_details()[0]
    stage2_input_details = stage2.get_input_details()[0]
    stage2_output_details = stage2.get_output_details()[0]

    for idx in range(num_beats):
        beat_int8 = adc_stub_get_beat(idx)

        # 1. Dequantiza int8 -> float32 (mesmo que o host enviaria pela UART).
        beat_float32 = dequantize_int8_to_float32(beat_int8, INPUT_SCALE, INPUT_ZERO_POINT)

        # Salva entrada bruta (nao filtrada) — o firmware aplica filtros em runtime.
        input_path = GROUND_TRUTH_DIR / f"ecg_input_{idx:02d}.bin"
        save_float32_bin(input_path, beat_float32)

        # 2. Aplica pipeline DSP causal bandpass -> notch -> zscore (igual ao firmware)
        #    para gerar a saida esperada.
        beat_filtered, _, _ = filter_chain(beat_float32)
        beat_normalized = zscore_normalize(beat_filtered)

        # 3. Re-quantiza float32 -> int8 com a mesma formula do firmware.
        input_quantized = quantize_float32_to_int8(beat_normalized, INPUT_SCALE, INPUT_ZERO_POINT)

        # 4. Estagio 1: N vs Anormal.
        tensor = input_quantized.reshape(stage1_input_details["shape"]).astype(np.int8)
        stage1.set_tensor(stage1_input_details["index"], tensor)
        stage1.invoke()
        stage1_out = stage1.get_tensor(stage1_output_details["index"])[0].copy()

        # 5. Se Anormal, executa Estagio 2 (S/V/F); caso contrario, saida e zero.
        stage1_cls = argmax_int8(stage1_out)
        if stage1_cls == 1:
            stage2.set_tensor(stage2_input_details["index"], tensor)
            stage2.invoke()
            final_out = stage2.get_tensor(stage2_output_details["index"])[0].copy()
        else:
            final_out = np.zeros(stage2_output_details["shape"][1], dtype=np.int8)

        output_path = GROUND_TRUTH_DIR / f"expected_output_{idx:02d}.bin"
        save_int8_bin(output_path, final_out)

        print(
            f"[ground-truth] idx={idx:02d} input={input_path} "
            f"output={output_path} stage1={stage1_cls} out={final_out.tolist()}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera dataset de ground-truth para teste de fidelidade QG10 (two-stage v2.0)."
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
