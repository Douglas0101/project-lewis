#!/usr/bin/env python3
"""Gera fixtures C para o firmware test harness a partir de referencias Python.

Reutiliza:
  - tests/ground_truth/generate_ground_truth.py  -> entradas e saidas do modelo TFLite.
  - tests/fixtures.dsp_filters / normalizer      -> saida esperada do filtro.
  - src.features.ampt_500hz                      -> picos R esperados.

Os headers gerados ficam em firmware/tests/fixtures/generated/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIRMWARE_ROOT = PROJECT_ROOT / "firmware"
OUTDIR = FIRMWARE_ROOT / "tests" / "fixtures" / "generated"
GROUND_TRUTH_DIR = PROJECT_ROOT / "tests" / "ground_truth"
GROUND_TRUTH_SCRIPT = GROUND_TRUTH_DIR / "generate_ground_truth.py"

# Garante imports do projeto.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.fixtures.adc_stub import adc_stub_get_beat  # noqa: E402
from tests.fixtures.dsp_filters import filter_chain  # noqa: E402
from tests.fixtures.normalizer import zscore_normalize  # noqa: E402
from src.features.ampt_500hz import AMPTDetector  # noqa: E402


def _load_quant_params() -> tuple[dict, dict]:
    path = PROJECT_ROOT / "models" / "quantized" / "quantization_params.json"
    if not path.exists():
        raise FileNotFoundError(f"Parametros de quantizacao nao encontrados: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["input"], data["output"]


def _quantize_float32_to_int8(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Replica a quantizacao do firmware (arredondamento para longe do zero)."""
    normalized = values / scale
    rounded = np.where(normalized >= 0.0, np.floor(normalized + 0.5), np.ceil(normalized - 0.5))
    quantized = rounded + zero_point
    return np.clip(quantized, -128, 127).astype(np.int8)


def _dequantize_int8_to_float32(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (values.astype(np.float32) - zero_point) * scale


def run_ground_truth(num_beats: int) -> None:
    """Executa o gerador de ground-truth do Python BUILTIN_REF."""
    if not GROUND_TRUTH_SCRIPT.exists():
        raise FileNotFoundError(f"Script de ground-truth nao encontrado: {GROUND_TRUTH_SCRIPT}")
    subprocess.run(
        [sys.executable, str(GROUND_TRUTH_SCRIPT), "--num-beats", str(num_beats)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _array_float_c(values: np.ndarray) -> str:
    return ", ".join(f"{v:.9g}f" for v in values)


def _array_int8_c(values: np.ndarray) -> str:
    return ", ".join(str(int(v)) for v in values)


def _array_uint32_c(values: np.ndarray) -> str:
    return ", ".join(f"{int(v)}u" for v in values)


def generate_pipeline_fixture(num_beats: int, input_quant: dict) -> str:
    """Gera header com inputs float32, inputs int8 processados e expected outputs int8 do modelo."""
    lines = [
        "#ifndef LEWIS_FIXTURE_PIPELINE_H",
        "#define LEWIS_FIXTURE_PIPELINE_H",
        "#include <stdint.h>",
        "",
        f"#define LEWIS_FIXTURE_PIPELINE_COUNT {num_beats}",
        "",
    ]

    scale = input_quant["scale"]
    zero_point = input_quant["zero_point"]

    for idx in range(num_beats):
        beat_int8 = adc_stub_get_beat(idx)
        beat_float = _dequantize_int8_to_float32(beat_int8, scale, zero_point)
        filtered, _, _ = filter_chain(beat_float)
        normalized = zscore_normalize(filtered)
        processed_int8 = _quantize_float32_to_int8(normalized, scale, zero_point)

        expected_path = GROUND_TRUTH_DIR / f"expected_output_{idx:02d}.bin"
        if not expected_path.exists():
            raise FileNotFoundError(f"Ground-truth de saida nao encontrado: {expected_path}")
        expected = np.frombuffer(expected_path.read_bytes(), dtype=np.int8)

        lines.append(f"static const float fixture_pipeline_input_{idx}[500] = {{")
        lines.append("    " + _array_float_c(beat_float))
        lines.append("};")
        lines.append(f"static const int8_t fixture_pipeline_input_{idx}_int8[500] = {{")
        lines.append("    " + _array_int8_c(processed_int8))
        lines.append("};")
        lines.append(f"static const int8_t fixture_pipeline_expected_{idx}[5] = {{")
        lines.append("    " + _array_int8_c(expected))
        lines.append("};")
        lines.append("")

    lines.append("#endif /* LEWIS_FIXTURE_PIPELINE_H */")
    return "\n".join(lines) + "\n"


def generate_dsp_fixture(num_beats: int, input_quant: dict) -> str:
    """Gera header com inputs float32 e saida filtrada esperada (Python)."""
    lines = [
        "#ifndef LEWIS_FIXTURE_DSP_H",
        "#define LEWIS_FIXTURE_DSP_H",
        "#include <stdint.h>",
        "",
        f"#define LEWIS_FIXTURE_DSP_COUNT {num_beats}",
        "",
    ]

    scale = input_quant["scale"]
    zero_point = input_quant["zero_point"]

    for idx in range(num_beats):
        beat_int8 = adc_stub_get_beat(idx)
        beat_float = _dequantize_int8_to_float32(beat_int8, scale, zero_point)
        filtered, _, _ = filter_chain(beat_float)
        normalized = zscore_normalize(filtered)

        lines.append(f"static const float fixture_dsp_input_{idx}[500] = {{")
        lines.append("    " + _array_float_c(beat_float))
        lines.append("};")
        lines.append(f"static const float fixture_dsp_expected_{idx}[500] = {{")
        lines.append("    " + _array_float_c(normalized))
        lines.append("};")
        lines.append("")

    lines.append("#endif /* LEWIS_FIXTURE_DSP_H */")
    return "\n".join(lines) + "\n"


def _synthetic_ecg(
    n_beats: int = 10,
    fs: float = 500.0,
    rr_ms: float = 800.0,
    noise_std: float = 0.005,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera sinal ECG sintetico com picos R conhecidos (mesma logica de test_r_peak_firmware.py)."""
    rng = np.random.default_rng(seed)
    rr_samples = int(round(rr_ms * fs / 1000.0))
    total_samples = rr_samples * n_beats + 500

    sig = np.zeros(total_samples, dtype=np.float64)
    r_peaks = np.array([rr_samples * i + rr_samples // 2 for i in range(n_beats)], dtype=np.int64)

    for rp in r_peaks:
        qrs_width = int(round(0.080 * fs))
        start = max(0, rp - qrs_width)
        end = min(total_samples, rp + qrs_width + 1)
        idx = np.arange(start, end)
        sig[idx] += 1.0 * np.exp(-0.5 * ((idx - rp) / (qrs_width / 3)) ** 2)

    for rp in r_peaks:
        tw_start = rp + int(round(0.25 * fs))
        tw_width = int(round(0.150 * fs))
        start = max(0, tw_start)
        end = min(total_samples, tw_start + tw_width)
        idx = np.arange(start, end)
        sig[idx] += 0.3 * np.exp(-0.5 * ((idx - tw_start - tw_width // 2) / (tw_width / 3)) ** 2)

    for rp in r_peaks:
        pw_start = rp - int(round(0.15 * fs))
        pw_width = int(round(0.10 * fs))
        start = max(0, pw_start)
        end = min(total_samples, pw_start + pw_width)
        idx = np.arange(start, end)
        sig[idx] += 0.15 * np.exp(-0.5 * ((idx - pw_start - pw_width // 2) / (pw_width / 3)) ** 2)

    sig += rng.normal(0.0, noise_std, total_samples)
    return sig.astype(np.float32), r_peaks.astype(np.uint32)


def generate_rpeak_fixture() -> str:
    """Gera header com sinal sintetico e picos esperados pelo AMPTDetector Python.

    O sinal e limitado a ~1700 amostras para caber na pilha do embarcado durante
    os testes do harness no Renode.
    """
    sig, _ = _synthetic_ecg(n_beats=3, rr_ms=800.0, noise_std=0.005, seed=42)
    det = AMPTDetector(fs=500.0)
    py_peaks = det.detect(sig.astype(np.float64))
    py_peaks_uint32 = py_peaks.astype(np.uint32)

    tol_samples = int(round(0.150 * 500.0))

    lines = [
        "#ifndef LEWIS_FIXTURE_RPEAK_H",
        "#define LEWIS_FIXTURE_RPEAK_H",
        "#include <stdint.h>",
        "#include <stddef.h>",
        "",
        f"#define LEWIS_FIXTURE_RPEAK_LEN {len(sig)}",
        f"#define LEWIS_FIXTURE_RPEAK_EXPECTED_COUNT {len(py_peaks_uint32)}",
        f"#define LEWIS_FIXTURE_RPEAK_TOL_SAMPLES {tol_samples}",
        "",
        "static const float fixture_rpeak_signal[" + str(len(sig)) + "] = {",
        "    " + _array_float_c(sig),
        "};",
        "",
        "static const uint32_t fixture_rpeak_expected[" + str(len(py_peaks_uint32)) + "] = {",
        "    " + _array_uint32_c(py_peaks_uint32),
        "};",
        "",
        "#endif /* LEWIS_FIXTURE_RPEAK_H */",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera fixtures C para o firmware test harness a partir de referencias Python."
    )
    parser.add_argument(
        "--num-beats",
        type=int,
        default=5,
        help="Numero de batimentos sinteticos para fixtures de pipeline/DSP (padrao: 5).",
    )
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    input_quant, _ = _load_quant_params()

    print(f"[fixtures] Gerando ground-truth Python com {args.num_beats} batimentos...")
    run_ground_truth(args.num_beats)

    print("[fixtures] Gerando fixture_pipeline.h...")
    (OUTDIR / "fixture_pipeline.h").write_text(
        generate_pipeline_fixture(args.num_beats, input_quant), encoding="utf-8"
    )

    print("[fixtures] Gerando fixture_dsp.h...")
    (OUTDIR / "fixture_dsp.h").write_text(
        generate_dsp_fixture(args.num_beats, input_quant), encoding="utf-8"
    )

    print("[fixtures] Gerando fixture_rpeak.h...")
    (OUTDIR / "fixture_rpeak.h").write_text(generate_rpeak_fixture(), encoding="utf-8")

    print(f"[fixtures] Headers gerados em {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
