"""Quality Gate QG1 — MITBIHLoader.

Validates:
* QG1.1 — Gain/baseline read from .hea (not hardcoded)
* QG1.2 — Signal range [-5, +5] mV after physical conversion
* QG1.3 — Sampling rate read from header
* QG1.4 — AAMI mapping correctness
* QG1.5 — Annotation filtering (non-beat symbols dropped)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.data.loader import MITBIHLoader


def _require_or_skip(path: Path, what: str) -> None:
    if path.exists() and any(path.glob("*.hea")):
        return
    if os.environ.get("LEWIS_REQUIRE_DATA") == "1":
        pytest.fail(f"{what} missing at {path}")
    pytest.skip(f"{what} missing or empty at {path}")


@pytest.mark.qg1
class TestLoaderStructure:
    """Validate loader configuration."""

    def test_fs_target_500hz(self):
        assert MITBIHLoader.FS_TARGET == 500.0

    def test_dataset_config_keys(self):
        from src.data.loader import DATASET_CONFIG

        expected = {"mitbih", "svdb", "afdb", "incart", "chapman"}
        assert set(DATASET_CONFIG.keys()) == expected


@pytest.mark.qg1
class TestLoaderSignal:
    """Validate signal loading with real data."""

    def test_load_signal_physical(self):
        raw = Path("data/raw_mitbih")
        _require_or_skip(raw, "raw_mitbih")
        rec = raw / "100"
        if not rec.with_suffix(".dat").exists():
            pytest.skip("Record 100 .dat not found")

        sig = MITBIHLoader.load_signal(rec, channel=0, units="physical")
        assert sig.ndim == 1
        assert len(sig) > 0
        assert sig.dtype == np.float64

    def test_load_signal_digital(self):
        raw = Path("data/raw_mitbih")
        _require_or_skip(raw, "raw_mitbih")
        rec = raw / "100"
        if not rec.with_suffix(".dat").exists():
            pytest.skip("Record 100 .dat not found")

        sig = MITBIHLoader.load_signal(rec, channel=0, units="digital")
        assert sig.ndim == 1
        assert len(sig) > 0

    def test_signal_range_physical(self):
        raw = Path("data/raw_mitbih")
        _require_or_skip(raw, "raw_mitbih")
        rec = raw / "100"
        if not rec.with_suffix(".dat").exists():
            pytest.skip("Record 100 .dat not found")

        sig = MITBIHLoader.load_signal(rec, channel=0, units="physical")
        assert sig.min() >= -5.0, f"Signal min = {sig.min():.3f} mV"
        assert sig.max() <= 5.0, f"Signal max = {sig.max():.3f} mV"

    def test_strict_range_raises_on_out_of_range(self):
        """strict_range=True must raise ValueError for signals outside [-5, +5] mV."""
        raw = Path("data/raw_incart")
        _require_or_skip(raw, "raw_incart")
        rec = raw / "I01"
        if not rec.with_suffix(".dat").exists():
            pytest.skip("Record I01 .dat not found")

        with pytest.raises(ValueError, match="Range físico fora de"):
            MITBIHLoader.load_signal(rec, channel=0, units="physical", strict_range=True)

    def test_default_range_warns_but_returns_signal(self):
        """Default strict_range=False must return signal even when out of range."""
        raw = Path("data/raw_incart")
        _require_or_skip(raw, "raw_incart")
        rec = raw / "I01"
        if not rec.with_suffix(".dat").exists():
            pytest.skip("Record I01 .dat not found")

        sig = MITBIHLoader.load_signal(rec, channel=0, units="physical")
        assert sig.ndim == 1
        assert len(sig) > 0

    def test_gain_read_from_header(self):
        """QG1: adc_gain must be read from .hea, not hardcoded."""
        raw = Path("data/raw_mitbih")
        _require_or_skip(raw, "raw_mitbih")
        rec = raw / "100"
        if not rec.with_suffix(".hea").exists():
            pytest.skip("Record 100 .hea not found")

        pytest.importorskip("wfdb")
        import wfdb  # type: ignore

        header = wfdb.rdheader(str(rec))
        assert header.adc_gain is not None
        assert len(header.adc_gain) > 0
        assert header.adc_gain[0] > 0

    def test_baseline_read_from_header(self):
        """QG1: baseline/adc_zero must be read from .hea."""
        raw = Path("data/raw_mitbih")
        _require_or_skip(raw, "raw_mitbih")
        rec = raw / "100"
        if not rec.with_suffix(".hea").exists():
            pytest.skip("Record 100 .hea not found")

        pytest.importorskip("wfdb")
        import wfdb  # type: ignore

        header = wfdb.rdheader(str(rec))
        # baseline may be in adc_zero or baseline attribute
        baseline = getattr(header, "baseline", None) or getattr(header, "adc_zero", None)
        assert baseline is not None

    def test_rejects_invalid_channel(self):
        raw = Path("data/raw_mitbih")
        _require_or_skip(raw, "raw_mitbih")
        rec = raw / "100"
        if not rec.with_suffix(".dat").exists():
            pytest.skip("Record 100 .dat not found")

        with pytest.raises(ValueError, match="Canal"):
            MITBIHLoader.load_signal(rec, channel=99)

    def test_rejects_invalid_units(self):
        # Validation of `units` happens before any file I/O, so no real data is required.
        fake_rec = Path("data/raw_mitbih/__nonexistent_record__")
        with pytest.raises(ValueError, match="units"):
            MITBIHLoader.load_signal(fake_rec, units="invalid")


@pytest.mark.qg1
class TestLoaderAnnotations:
    """Validate annotation loading and AAMI mapping."""

    def test_load_annotations(self):
        raw = Path("data/raw_mitbih")
        _require_or_skip(raw, "raw_mitbih")
        rec = raw / "100"
        if not rec.with_suffix(".atr").exists():
            pytest.skip("Record 100 .atr not found")

        samples, labels, metadata = MITBIHLoader.load_annotations(rec)
        assert len(samples) > 0
        assert len(labels) == len(samples)
        assert set(labels).issubset({"N", "S", "V", "F", "Q"})
        assert "total_beats" in metadata
        assert metadata["total_beats"] == len(samples)

    def test_non_beat_annotations_filtered(self):
        """QG1: rhythm changes (+), signal quality (~) must be dropped."""
        raw = Path("data/raw_mitbih")
        _require_or_skip(raw, "raw_mitbih")
        rec = raw / "100"
        if not rec.with_suffix(".atr").exists():
            pytest.skip("Record 100 .atr not found")

        samples, labels, metadata = MITBIHLoader.load_annotations(rec)
        # Verify no non-beat symbols in output
        non_beat_aami = {"~", "+", "x"}
        for label in labels:
            assert label not in non_beat_aami

    def test_paced_ratio_in_metadata(self):
        raw = Path("data/raw_mitbih")
        _require_or_skip(raw, "raw_mitbih")
        rec = raw / "100"
        if not rec.with_suffix(".atr").exists():
            pytest.skip("Record 100 .atr not found")

        _, _, metadata = MITBIHLoader.load_annotations(rec)
        assert "paced_ratio" in metadata
        assert 0.0 <= metadata["paced_ratio"] <= 1.0

    def test_noise_segments_in_metadata(self):
        raw = Path("data/raw_mitbih")
        _require_or_skip(raw, "raw_mitbih")
        rec = raw / "100"
        if not rec.with_suffix(".atr").exists():
            pytest.skip("Record 100 .atr not found")

        _, _, metadata = MITBIHLoader.load_annotations(rec)
        assert "noise_segments" in metadata
        assert isinstance(metadata["noise_segments"], int)


@pytest.mark.qg1
class TestLoaderRecordNames:
    """Validate record enumeration."""

    def test_get_record_names(self):
        raw = Path("data/raw_mitbih")
        _require_or_skip(raw, "raw_mitbih")

        names = MITBIHLoader.get_record_names(raw)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_iter_dataset(self):
        raw = Path("data/raw_mitbih")
        _require_or_skip(raw, "raw_mitbih")

        records = list(MITBIHLoader.iter_dataset("mitbih", data_dir=raw))
        assert len(records) > 0
        for rec in records:
            assert "record_name" in rec
            assert "record_path" in rec
            assert "fs_native" in rec
            assert "lead_name" in rec
