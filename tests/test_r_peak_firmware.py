"""Quality Gate QG18 — Detector leve de R-peak em C vs referencia Python AMPT.

Verifica que a implementacao em ponto flutuante simples do firmware detecta
picos R de forma consistente com o AMPTDetector Python em um sinal sintetico.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.features.ampt_500hz import AMPTDetector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRMWARE_SRC = PROJECT_ROOT / "firmware" / "src"

C_DETECTOR_SOURCE = r"""
#include "dsp/r_peak_detector.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input_float32.bin> <output_peaks.bin>\n", argv[0]);
        return 1;
    }

    FILE* fin = fopen(argv[1], "rb");
    if (!fin) {
        fprintf(stderr, "ERRO: nao abriu %s\n", argv[1]);
        return 1;
    }
    fseek(fin, 0, SEEK_END);
    long bytes = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    size_t n = (size_t)bytes / sizeof(float);

    float* sig = (float*)malloc(n * sizeof(float));
    if (!sig) {
        fclose(fin);
        return 1;
    }
    if (fread(sig, sizeof(float), n, fin) != n) {
        free(sig);
        fclose(fin);
        return 1;
    }
    fclose(fin);

    size_t peaks[LEWIS_RPEAK_MAX_PEAKS];
    size_t n_peaks = 0;
    int rc = lewis_detect_r_peaks(sig, n, 500.0f, peaks, &n_peaks);
    free(sig);

    if (rc != 0) {
        fprintf(stderr, "ERRO: lewis_detect_r_peaks retornou %d\n", rc);
        return 1;
    }

    FILE* fout = fopen(argv[2], "wb");
    if (!fout) {
        fprintf(stderr, "ERRO: nao criou %s\n", argv[2]);
        return 1;
    }
    for (size_t i = 0; i < n_peaks; ++i) {
        uint32_t p = (uint32_t)peaks[i];
        fwrite(&p, sizeof(uint32_t), 1, fout);
    }
    fclose(fout);
    return 0;
}
"""


def _synthetic_ecg(
    n_beats: int = 10,
    fs: float = 500.0,
    rr_ms: float = 800.0,
    noise_std: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera sinal ECG sintetico com picos R conhecidos."""
    rr_samples = int(round(rr_ms * fs / 1000.0))
    total_samples = rr_samples * n_beats + 500

    sig = np.zeros(total_samples, dtype=np.float64)
    r_peaks = np.array(
        [rr_samples * i + rr_samples // 2 for i in range(n_beats)], dtype=np.int64
    )

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

    sig += np.random.normal(0, noise_std, total_samples)
    return sig.astype(np.float32), r_peaks


@pytest.fixture(scope="module")
def detector_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compila programa C com o detector leve de R-peaks."""
    tmpdir = tmp_path_factory.mktemp("rpeak_detector_test")
    c_path = tmpdir / "rpeak_detector.c"
    c_path.write_text(C_DETECTOR_SOURCE, encoding="utf-8")
    bin_path = tmpdir / "rpeak_detector"

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
        str(FIRMWARE_SRC / "dsp" / "r_peak_detector.c"),
        "-lm",
        "-o",
        str(bin_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return bin_path


def _run_c_detector(bin_path: Path, sig: np.ndarray) -> np.ndarray:
    """Executa o detector C no sinal e retorna array de indices de pico."""
    with tempfile.TemporaryDirectory(prefix="lewis_rpeak_") as tmpdir:
        in_path = Path(tmpdir) / "input.bin"
        out_path = Path(tmpdir) / "peaks.bin"
        in_path.write_bytes(sig.astype("<f4").tobytes())
        subprocess.run(
            [str(bin_path), str(in_path), str(out_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        out_bytes = out_path.read_bytes()
    return np.frombuffer(out_bytes, dtype="<u4").copy()


def _match_peaks(c_peaks: np.ndarray, py_peaks: np.ndarray, tol_samples: int) -> dict:
    """Calcula TP, FN, FP, Sens, PPV entre duas listas de picos."""
    matched_c = set()
    matched_py = set()
    tp = 0
    for i, cp in enumerate(c_peaks):
        for j, pp in enumerate(py_peaks):
            if j in matched_py:
                continue
            if abs(int(cp) - int(pp)) <= tol_samples:
                tp += 1
                matched_c.add(i)
                matched_py.add(j)
                break
    fn = len(py_peaks) - len(matched_py)
    fp = len(c_peaks) - len(matched_c)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {"TP": tp, "FN": fn, "FP": fp, "Sens": sens, "PPV": ppv}


@pytest.mark.qg18
class TestRPeakFirmware:
    MIN_SENS = 0.90
    MIN_PPV = 0.90

    def test_c_detector_matches_ampt_on_synthetic(self, detector_bin: Path) -> None:
        np.random.seed(42)
        sig, _ = _synthetic_ecg(n_beats=10, fs=500.0, rr_ms=800.0, noise_std=0.005)

        det = AMPTDetector(fs=500.0)
        py_peaks = det.detect(sig.astype(np.float64))
        c_peaks = _run_c_detector(detector_bin, sig)

        tol_samples = int(round(0.150 * 500.0))
        metrics = _match_peaks(c_peaks, py_peaks, tol_samples)

        assert metrics["Sens"] >= self.MIN_SENS, (
            f"Sensibilidade C vs AMPT = {metrics['Sens']:.3f} "
            f"(c={c_peaks.tolist()}, py={py_peaks.tolist()})"
        )
        assert metrics["PPV"] >= self.MIN_PPV, (
            f"PPV C vs AMPT = {metrics['PPV']:.3f} "
            f"(c={c_peaks.tolist()}, py={py_peaks.tolist()})"
        )

    def test_c_detector_finds_expected_peaks(self, detector_bin: Path) -> None:
        np.random.seed(43)
        sig, r_true = _synthetic_ecg(n_beats=8, fs=500.0, rr_ms=900.0, noise_std=0.005)
        c_peaks = _run_c_detector(detector_bin, sig)

        tol_samples = int(round(0.150 * 500.0))
        metrics = _match_peaks(c_peaks, r_true, tol_samples)

        assert metrics["Sens"] >= self.MIN_SENS, (
            f"Sensibilidade vs ground-truth = {metrics['Sens']:.3f}"
        )
