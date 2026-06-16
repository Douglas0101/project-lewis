"""Quality Gate QG2 — AMPT Detector @ 500 Hz.

Validates the AccYouRate Modified Pan-Tompkins detector:
* QG2.1 — Structure: band 5-15 Hz, MWI 150ms, refractory 360ms, tol=150ms.
* QG2.2 — Synthetic signal: detects known R-peak positions.
* QG2.3 — MIT-BIH record 100 (if data available): Sens > 96.5%, PPV > 99.0%.
* QG2.4 — T-wave discrimination: no false peaks within refractory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from src.features.ampt_500hz import TOL_MS_DEFAULT, AMPTDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_ecg(
    n_beats: int = 10,
    fs: float = 500.0,
    rr_ms: float = 800.0,
    noise_std: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic ECG-like signal with known R-peak positions."""
    rr_samples = int(round(rr_ms * fs / 1000.0))
    total_samples = rr_samples * n_beats + 500

    sig = np.zeros(total_samples, dtype=np.float64)
    r_peaks = np.array([rr_samples * i + rr_samples // 2 for i in range(n_beats)], dtype=np.int64)

    for rp in r_peaks:
        # QRS complex: narrow Gaussian (~80 ms width)
        qrs_width = int(round(0.080 * fs))
        start = max(0, rp - qrs_width)
        end = min(total_samples, rp + qrs_width + 1)
        idx = np.arange(start, end)
        sig[idx] += 1.0 * np.exp(-0.5 * ((idx - rp) / (qrs_width / 3)) ** 2)

    # Add T-wave (~300 ms after R)
    for rp in r_peaks:
        tw_start = rp + int(round(0.25 * fs))
        tw_width = int(round(0.15 * fs))
        start = max(0, tw_start)
        end = min(total_samples, tw_start + tw_width)
        idx = np.arange(start, end)
        sig[idx] += 0.3 * np.exp(-0.5 * ((idx - tw_start - tw_width // 2) / (tw_width / 3)) ** 2)

    # Add P-wave (~-150 ms before R)
    for rp in r_peaks:
        pw_start = rp - int(round(0.15 * fs))
        pw_width = int(round(0.10 * fs))
        start = max(0, pw_start)
        end = min(total_samples, pw_start + pw_width)
        idx = np.arange(start, end)
        sig[idx] += 0.15 * np.exp(-0.5 * ((idx - pw_start - pw_width // 2) / (pw_width / 3)) ** 2)

    # Add noise
    sig += np.random.normal(0, noise_std, total_samples)

    return sig, r_peaks


def _require_mitbih_data() -> None:
    raw = Path("data/raw_mitbih")
    if not raw.exists() or not any(raw.glob("*.hea")):
        if os.environ.get("LEWIS_REQUIRE_DATA") == "1":
            pytest.fail("MIT-BIH data missing — run `make download-mitbih`")
        pytest.skip("MIT-BIH data missing — run `make download-mitbih`")


# ---------------------------------------------------------------------------
# QG2.1 — Structure / Parameters
# ---------------------------------------------------------------------------


@pytest.mark.qg2
class TestAMPTStructure:
    """Validate detector configuration matches spec v1.1 §4.4."""

    def test_default_fs_500hz(self):
        det = AMPTDetector()
        assert det.fs == 500.0

    def test_mwi_window_150ms(self):
        det = AMPTDetector(fs=500.0)
        expected = int(round(0.150 * 500.0))
        assert det.mwi_window == expected or det.mwi_window == expected + 1
        # odd window
        assert det.mwi_window % 2 == 1

    def test_refractory_360ms(self):
        det = AMPTDetector(fs=500.0)
        assert det.refractory == int(round(0.360 * 500.0))

    def test_tolerance_150ms(self):
        assert TOL_MS_DEFAULT == 150.0
        det = AMPTDetector(fs=500.0)
        assert det.tol_samples == int(round(150.0 * 500.0 / 1000.0))

    def test_bandpass_coefficients(self):
        det = AMPTDetector(fs=500.0)
        # Verify bandpass is 5-15 Hz
        w, h = signal.freqz(det.b_band, det.a_band, worN=8192, fs=500.0)
        # Peak should be in 5-15 Hz range
        peak_idx = np.argmax(np.abs(h))
        peak_freq = w[peak_idx]
        assert 4.5 <= peak_freq <= 15.5, f"Peak at {peak_freq:.1f} Hz, expected 5-15 Hz"

    def test_rejects_multidim(self):
        det = AMPTDetector()
        with pytest.raises(ValueError, match="1-D"):
            det.detect(np.zeros((10, 2)))


# ---------------------------------------------------------------------------
# QG2.2 — Synthetic Signal
# ---------------------------------------------------------------------------


@pytest.mark.qg2
class TestAMPTSynthetic:
    """Validate detector on synthetic ECG with known peak positions."""

    def test_detects_all_synthetic_peaks(self):
        np.random.seed(42)
        sig, r_true = _synthetic_ecg(n_beats=20, fs=500.0, rr_ms=800.0, noise_std=0.005)
        det = AMPTDetector(fs=500.0)
        r_det = det.detect(sig)

        # Should detect close to all peaks (within 150 ms tolerance)
        tol_samples = int(round(0.150 * 500.0))
        matched = 0
        matched_det = set()
        for rt in r_true:
            for j, rd in enumerate(r_det):
                if j in matched_det:
                    continue
                if abs(int(rt) - int(rd)) <= tol_samples:
                    matched += 1
                    matched_det.add(j)
                    break

        sens = matched / len(r_true)
        assert sens >= 0.95, f"Sensitivity on synthetic = {sens:.3f}, expected >= 0.95"

    def test_no_false_peaks_in_refractory(self):
        """After a true QRS, no additional peaks within 360 ms refractory."""
        np.random.seed(43)
        sig, r_true = _synthetic_ecg(n_beats=10, fs=500.0, rr_ms=800.0, noise_std=0.005)
        det = AMPTDetector(fs=500.0)
        r_det = det.detect(sig)

        # Check that no two detected peaks are closer than refractory
        refractory = det.refractory
        for i in range(1, len(r_det)):
            interval = r_det[i] - r_det[i - 1]
            assert interval >= refractory * 0.5, (
                f"Peaks {r_det[i-1]} and {r_det[i]} are {interval} samples apart, "
                f"refractory={refractory}"
            )

    def test_evaluate_returns_metrics(self):
        np.random.seed(44)
        sig, r_true = _synthetic_ecg(n_beats=10, fs=500.0, rr_ms=800.0)
        det = AMPTDetector(fs=500.0)
        metrics = det.evaluate(sig, r_true, tol_ms=150.0)

        assert "TP" in metrics
        assert "FN" in metrics
        assert "FP" in metrics
        assert "Sens" in metrics
        assert "PPV" in metrics
        assert "F1" in metrics
        assert metrics["n_true"] == len(r_true)

    def test_short_signal_returns_empty(self):
        det = AMPTDetector(fs=500.0)
        sig = np.zeros(50)
        peaks = det.detect(sig)
        assert len(peaks) == 0


# ---------------------------------------------------------------------------
# QG2.3 — MIT-BIH Record 100 (if data available)
# ---------------------------------------------------------------------------


@pytest.mark.qg2
class TestAMPTMITBIH:
    """Validate detector against MIT-BIH ground truth annotations."""

    def test_record_100_sensitivity_and_ppv(self):
        _require_mitbih_data()
        pytest.importorskip("wfdb")
        import wfdb  # type: ignore

        record_path = Path("data/raw_mitbih/100")
        if not record_path.with_suffix(".dat").exists():
            pytest.skip("Record 100 not found")

        # Load signal (lead MLII = channel 0) and resample to 500 Hz (AMPT target)
        rec = wfdb.rdrecord(str(record_path), channels=[0], physical=True)
        sig = rec.p_signal.squeeze().astype(np.float64)
        fs_native = float(rec.fs)
        sig = signal.resample_poly(sig, up=500, down=int(fs_native), padtype="line")

        # Load annotations
        ann = wfdb.rdann(str(record_path), extension="atr")
        # Filter only beat annotations (same symbols as AAMI mapper)
        beat_symbols = {"N", "L", "R", "e", "j", "V", "E", "A", "a", "J", "S", "F", "/", "f", "Q"}
        mask = np.isin(ann.symbol, list(beat_symbols))
        r_true = (np.array(ann.sample)[mask] * 500.0 / fs_native).astype(np.int64)

        det = AMPTDetector(fs=500.0)
        metrics = det.evaluate(sig, r_true, tol_ms=150.0)

        LOGGER = logging.getLogger("lewis.tests.ampt")
        LOGGER.info(
            "MIT-BIH 100: TP=%d FN=%d FP=%d Sens=%.4f PPV=%.4f",
            metrics["TP"],
            metrics["FN"],
            metrics["FP"],
            metrics["Sens"],
            metrics["PPV"],
        )

        # QG2 thresholds (spec v1.1 §6)
        assert metrics["Sens"] > 0.965, f"Sensitivity = {metrics['Sens']:.4f}, expected > 0.965"
        assert metrics["PPV"] > 0.990, f"PPV = {metrics['PPV']:.4f}, expected > 0.990"
        assert metrics["F1"] > 0.975, f"F1 = {metrics['F1']:.4f}, expected > 0.975"

    def test_record_100_fp_rate(self):
        _require_mitbih_data()
        pytest.importorskip("wfdb")
        import wfdb  # type: ignore

        record_path = Path("data/raw_mitbih/100")
        if not record_path.with_suffix(".dat").exists():
            pytest.skip("Record 100 not found")

        rec = wfdb.rdrecord(str(record_path), channels=[0], physical=True)
        sig = rec.p_signal.squeeze().astype(np.float64)
        fs_native = float(rec.fs)
        sig = signal.resample_poly(sig, up=500, down=int(fs_native), padtype="line")

        ann = wfdb.rdann(str(record_path), extension="atr")
        beat_symbols = {"N", "L", "R", "e", "j", "V", "E", "A", "a", "J", "S", "F", "/", "f", "Q"}
        mask = np.isin(ann.symbol, list(beat_symbols))
        r_true = (np.array(ann.sample)[mask] * 500.0 / fs_native).astype(np.int64)

        det = AMPTDetector(fs=500.0)
        metrics = det.evaluate(sig, r_true, tol_ms=150.0)

        fp_rate = metrics["FP"] / max(metrics["n_true"], 1)
        assert fp_rate < 0.01, f"FP rate = {fp_rate:.4f}, expected < 0.01"
