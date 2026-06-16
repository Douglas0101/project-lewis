"""Quality Gate QG1 — ECGPreprocessor.

Validates:
* QG1.1 — Butterworth bandpass 0.5-40 Hz, order 4, filtfilt (zero-phase)
* QG1.2 — Detrend linear
* QG1.3 — Z-score global (mean ≈ 0, std ≈ 1)
* QG1.4 — No clipping (range ±5 mV)
* QG1.5 — Idempotency via lineage
* QG1.6 — DLQ for failures
* QG1.7 — Global stats mandatory when per_record=False
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from src.data.preprocessor import DEFAULT_CONFIG, ECGPreprocessor

LOGGER = logging.getLogger("lewis.tests.preprocessing")


@pytest.fixture
def dlq_path(tmp_path: Path) -> Path:
    """Isola a DLQ de pré-processamento em diretório temporário."""
    return tmp_path / "preprocess_failures.jsonl"


@pytest.fixture(autouse=True)
def _isolate_preprocessing_dirs(monkeypatch, tmp_path: Path) -> None:
    """Força ECGPreprocessor a escrever outputs em diretório temporário."""
    monkeypatch.setattr("src.data.preprocessor.PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr("src.data.preprocessor.LINEAGE_DIR", tmp_path / "lineage")


@pytest.mark.qg1
class TestPreprocessorStructure:
    """Validate preprocessor configuration and initialization."""

    def test_default_config(self):
        cfg = DEFAULT_CONFIG
        assert cfg["resample"]["target_fs"] == 500.0
        assert cfg["filter"]["lowcut"] == 0.5
        assert cfg["filter"]["highcut"] == 40.0
        assert cfg["filter"]["order"] == 4
        assert cfg["filter"]["implementation"] == "filtfilt"
        assert cfg["detrend"]["type"] == "linear"
        assert cfg["normalization"]["type"] == "zscore_global"

    def test_load_config_from_yaml(self, dlq_path):
        cfg_path = Path("config/preprocess_v1.0.yaml")
        if not cfg_path.exists():
            pytest.skip("config/preprocess_v1.0.yaml not found")
        pre = ECGPreprocessor(config_path=cfg_path, dlq_path=dlq_path)
        assert pre.fs == 500.0
        assert pre.lowcut == 0.5
        assert pre.highcut == 40.0
        assert pre.order == 4

    def test_fallback_config(self, dlq_path):
        pre = ECGPreprocessor(config_path=Path("nonexistent.yaml"), dlq_path=dlq_path)
        assert pre.fs == 500.0
        assert pre.lowcut == 0.5


@pytest.mark.qg1
class TestPreprocessorFilter:
    """Validate Butterworth bandpass filter."""

    def test_filter_zero_phase(self, dlq_path):
        """QG1: filtfilt produces zero-phase distortion."""
        pre = ECGPreprocessor(dlq_path=dlq_path)
        # Create signal with a single peak in the centre (non-periodic, avoids argmax ambiguity)
        fs = 500.0
        n = int(fs * 3)
        centre = n // 2
        x = np.zeros(n, dtype=np.float64)
        sigma = int(0.05 * fs)  # 50 ms Gaussian
        x[centre - sigma : centre + sigma + 1] = np.exp(
            -0.5 * ((np.arange(2 * sigma + 1) - sigma) / (sigma / 3)) ** 2
        )
        y = pre.filter(x)

        # filtfilt should preserve the peak position (zero-phase)
        peak_x = int(np.argmax(x))
        peak_y = int(np.argmax(y))
        assert abs(peak_x - peak_y) <= 3, f"Phase distortion: peak_x={peak_x}, peak_y={peak_y}"

    def test_filter_bandpass_attenuation(self, dlq_path):
        """QG1: Components outside 0.5-40 Hz should be attenuated."""
        pre = ECGPreprocessor(dlq_path=dlq_path)
        fs = 500.0
        t = np.arange(int(fs * 2)) / fs  # 2 seconds

        # Signal with 10 Hz (in-band) + 100 Hz (out-of-band)
        x = np.sin(2 * np.pi * 10.0 * t) + 0.5 * np.sin(2 * np.pi * 100.0 * t)
        y = pre.filter(x)

        # Power at 100 Hz should be attenuated
        freqs, psd_in = signal.welch(x, fs=fs, nperseg=256)
        _, psd_out = signal.welch(y, fs=fs, nperseg=256)

        mask_100 = (freqs > 80) & (freqs < 120)
        power_in_100 = np.mean(psd_in[mask_100])
        power_out_100 = np.mean(psd_out[mask_100])
        attenuation_db = -10 * np.log10(power_out_100 / power_in_100 + 1e-12)
        assert attenuation_db > 20.0, f"Attenuation at 100 Hz = {attenuation_db:.1f} dB"

    def test_filter_preserves_in_band(self, dlq_path):
        """QG1: 10 Hz signal should pass through with minimal attenuation."""
        pre = ECGPreprocessor(dlq_path=dlq_path)
        fs = 500.0
        t = np.arange(int(fs * 2)) / fs
        x = np.sin(2 * np.pi * 10.0 * t)
        y = pre.filter(x)

        # Peak amplitude should be preserved (~1.0)
        assert np.max(np.abs(y)) > 0.5, "In-band signal attenuated too much"


@pytest.mark.qg1
class TestPreprocessorDetrend:
    """Validate linear detrending."""

    def test_detrend_removes_linear_drift(self, dlq_path):
        pre = ECGPreprocessor(dlq_path=dlq_path)
        x = np.linspace(0, 1, 500) + np.sin(2 * np.pi * 5.0 * np.arange(500) / 500.0)
        y = pre.detrend(x)

        # Linear trend should be removed
        slope = np.polyfit(np.arange(len(y)), y, 1)[0]
        assert abs(slope) < 0.01, f"Linear slope after detrend = {slope:.4f}"

    def test_detrend_preserves_oscillation(self, dlq_path):
        pre = ECGPreprocessor(dlq_path=dlq_path)
        t = np.arange(500) / 500.0
        x = 2.0 * t + np.sin(2 * np.pi * 5.0 * t)
        y = pre.detrend(x)

        # Oscillatory component should remain
        assert np.std(y) > 0.3, "Oscillation removed by detrend"


@pytest.mark.qg1
class TestPreprocessorNormalize:
    """Validate z-score normalization."""

    def _per_record_pre(self, dlq_path: Path) -> ECGPreprocessor:
        """Helper: preprocessor with per-record normalization for isolated tests."""
        pre = ECGPreprocessor(dlq_path=dlq_path)
        pre.cfg["normalization"] = pre.cfg["normalization"].copy()
        pre.cfg["normalization"]["per_record"] = True
        pre.per_record = True
        return pre

    def test_zscore_mean_zero(self, dlq_path):
        pre = self._per_record_pre(dlq_path)
        x = np.random.randn(500) + 3.0  # mean = 3
        y = pre.normalize(x)

        assert abs(float(np.mean(y))) < 1e-6, f"Mean after z-score = {np.mean(y):.6f}"

    def test_zscore_std_one(self, dlq_path):
        pre = self._per_record_pre(dlq_path)
        x = np.random.randn(500) * 2.0  # std = 2
        y = pre.normalize(x)

        assert abs(float(np.std(y)) - 1.0) < 1e-4, f"Std after z-score = {np.std(y):.6f}"

    def test_zscore_with_eps(self, dlq_path):
        pre = self._per_record_pre(dlq_path)
        x = np.ones(500)  # std = 0
        y = pre.normalize(x)

        # Should not divide by zero (eps = 1e-12)
        assert not np.any(np.isinf(y))
        assert not np.any(np.isnan(y))

    def test_zscore_requires_global_stats(self, dlq_path):
        """QG1.7: normalize() must raise when per_record=False and stats are missing."""
        pre = ECGPreprocessor(dlq_path=dlq_path)
        with pytest.raises(ValueError, match="Estatísticas globais não definidas"):
            pre.normalize(np.random.randn(500))


@pytest.mark.qg1
class TestPreprocessorProcess:
    """Validate end-to-end processing pipeline."""

    def test_process_pipeline(self, dlq_path):
        pre = ECGPreprocessor(dlq_path=dlq_path)
        pre.cfg["normalization"] = pre.cfg["normalization"].copy()
        pre.cfg["normalization"]["per_record"] = True
        pre.per_record = True
        fs_orig = 360.0
        n_samples = int(fs_orig * 2)  # 2 seconds
        t = np.arange(n_samples) / fs_orig
        x = np.sin(2 * np.pi * 10.0 * t) + 0.1 * np.random.randn(n_samples)

        x_proc, metadata = pre.process(
            x,
            record_id="test_100",
            dataset="mitdb",
            fs_orig=fs_orig,
            raw_path=Path("data/raw_mitbih/test_100"),
            lead_name="MLII",
        )

        assert x_proc.dtype == np.float32
        assert len(x_proc) == int(round(n_samples * 500.0 / fs_orig))
        assert abs(float(np.mean(x_proc))) < 1e-6
        assert abs(float(np.std(x_proc)) - 1.0) < 1e-4
        assert metadata["dataset"] == "mitdb"
        assert metadata["record_id"] == "test_100"
        assert "input_range_mV" in metadata
        assert "output_range_mV" in metadata
        assert "duration_sec" in metadata

    def test_process_no_resample_when_fs_matches(self, dlq_path):
        pre = ECGPreprocessor(dlq_path=dlq_path)
        pre.cfg["normalization"] = pre.cfg["normalization"].copy()
        pre.cfg["normalization"]["per_record"] = True
        pre.per_record = True
        n_samples = 1000
        x = np.random.randn(n_samples)

        x_proc, _ = pre.process(
            x,
            record_id="test_500",
            dataset="mitdb",
            fs_orig=500.0,
            raw_path=Path("data/raw_mitbih/test_500"),
            lead_name="MLII",
        )

        assert len(x_proc) == n_samples

    def test_process_raises_on_invalid_input(self, dlq_path):
        pre = ECGPreprocessor(dlq_path=dlq_path)
        pre.cfg["normalization"] = pre.cfg["normalization"].copy()
        pre.cfg["normalization"]["per_record"] = True
        pre.per_record = True
        with pytest.raises(ValueError):
            pre.process(
                np.zeros((10, 2)),  # 2D not allowed
                record_id="test",
                dataset="mitdb",
                fs_orig=360.0,
                raw_path=Path("data/raw_mitbih/test"),
                lead_name="MLII",
            )

    def test_process_requires_global_stats(self, dlq_path):
        """QG1.7: process() must raise when per_record=False and stats are missing."""
        pre = ECGPreprocessor(dlq_path=dlq_path)
        with pytest.raises(ValueError, match="Estatísticas globais não definidas"):
            pre.process(
                np.random.randn(500),
                record_id="test_no_stats",
                dataset="mitdb",
                fs_orig=500.0,
                raw_path=Path("data/raw_mitbih/test_no_stats"),
                lead_name="MLII",
            )

    def test_process_failure_uses_isolated_dlq(self, dlq_path):
        """DLQ must be isolated from production path during tests."""
        pre = ECGPreprocessor(dlq_path=dlq_path)
        production_dlq = Path("data/.dlq/preprocess_failures.jsonl").resolve()

        # Force a failure by passing an unsupported 2-D signal.
        with pytest.raises(ValueError):
            pre.process(
                np.zeros((10, 2)),
                record_id="test_dlq_isolation",
                dataset="mitdb",
                fs_orig=360.0,
                raw_path=Path("data/raw_mitbih/test_dlq_isolation"),
                lead_name="MLII",
            )

        # Temporary DLQ should contain exactly one failure entry.
        assert dlq_path.exists(), "Temporary DLQ was not created"
        entries = [
            line for line in dlq_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        assert len(entries) == 1
        assert json.loads(entries[0])["record_id"] == "test_dlq_isolation"

        # Production DLQ must remain untouched.
        assert (
            not production_dlq.exists() or production_dlq.stat().st_size == 0
        ), "Production DLQ was populated by tests"


@pytest.mark.qg1
class TestPreprocessorGlobalStats:
    """Validate global z-score normalization."""

    def test_set_global_stats(self, dlq_path):
        pre = ECGPreprocessor(dlq_path=dlq_path)
        pre.set_global_stats(mean=0.5, std=2.0)
        x = np.random.randn(500)
        y = pre.normalize(x)

        expected = (x - 0.5) / 2.0
        np.testing.assert_array_almost_equal(y, expected, decimal=10)
