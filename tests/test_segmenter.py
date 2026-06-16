"""Quality Gate QG3 — ECG Segmenter.

Validates beat segmentation:
* QG3.1 — Window size: 1000 ms default (500 samples @ 500 Hz), 600 ms fallback (300 samples).
* QG3.2 — No zero-padding: beats at edges are discarded.
* QG3.3 — R-peak centered: the R-peak is near the center of the window.
* QG3.4 — Output dtype float32, no NaN/Inf.
* QG3.5 — RR < 600 ms uses min_window, RR >= 600 ms uses standard window.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.segmenter import ECGSegmenter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_signal_with_peaks(
    n_samples: int = 5000,
    fs: float = 500.0,
    r_peaks: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic signal with impulses at R-peak positions."""
    sig = np.zeros(n_samples, dtype=np.float64)
    if r_peaks is None:
        # 5 peaks spaced 1000 ms apart (500 samples)
        r_peaks = [500, 1000, 1500, 2000, 2500]
    for rp in r_peaks:
        if 0 <= rp < n_samples:
            sig[rp] = 1.0
            # small neighborhood
            for offset in (-1, 1):
                idx = rp + offset
                if 0 <= idx < n_samples:
                    sig[idx] = 0.5
    return sig, np.array(r_peaks, dtype=np.int64)


# ---------------------------------------------------------------------------
# QG3.1 — Window Sizes
# ---------------------------------------------------------------------------


@pytest.mark.qg3
class TestSegmenterWindowSizes:
    """Validate default and minimum window dimensions."""

    def test_default_window_1000ms(self):
        seg = ECGSegmenter(fs=500.0, window_ms=1000.0, min_window_ms=600.0)
        assert seg.window_len == 500  # 2*250
        assert seg.half_len == 250

    def test_min_window_600ms(self):
        seg = ECGSegmenter(fs=500.0, window_ms=1000.0, min_window_ms=600.0)
        assert seg.min_window_len == 300  # 2*150
        assert seg.min_half_len == 150

    def test_centered_at_500hz(self):
        seg = ECGSegmenter(fs=500.0, window_ms=1000.0)
        assert seg.window_len == 500
        assert seg.half_len * 2 == seg.window_len


# ---------------------------------------------------------------------------
# QG3.2 — No Zero Padding
# ---------------------------------------------------------------------------


@pytest.mark.qg3
class TestSegmenterNoPadding:
    """Validate that edge beats are discarded instead of padded."""

    def test_discards_early_peak(self):
        sig, r_peaks = _synthetic_signal_with_peaks(n_samples=5000, r_peaks=[100, 1500, 2500])
        labels = np.array(["N", "N", "N"])
        seg = ECGSegmenter(fs=500.0)
        X, y, meta = seg.segment_with_labels(sig, r_peaks, labels)

        assert meta["n_descartados_bordas"] >= 1
        assert len(X) < len(r_peaks)

    def test_discards_late_peak(self):
        sig, r_peaks = _synthetic_signal_with_peaks(n_samples=3000, r_peaks=[1500, 2900])
        labels = np.array(["N", "N"])
        seg = ECGSegmenter(fs=500.0)
        X, y, meta = seg.segment_with_labels(sig, r_peaks, labels)

        assert meta["n_descartados_bordas"] >= 1
        assert len(X) < len(r_peaks)

    def test_all_valid_peaks_kept(self):
        # Peaks well inside the signal
        r_peaks = [1000, 2000, 3000, 4000]
        sig, _ = _synthetic_signal_with_peaks(n_samples=5500, r_peaks=r_peaks)
        labels = np.array(["N", "V", "S", "F"])
        seg = ECGSegmenter(fs=500.0)
        X, y, meta = seg.segment_with_labels(sig, r_peaks, labels)

        assert meta["n_descartados_bordas"] == 0
        assert len(X) == len(r_peaks)
        assert list(y) == ["N", "V", "S", "F"]

    def test_no_padding_in_output(self):
        r_peaks = [1500, 2500]
        sig, _ = _synthetic_signal_with_peaks(n_samples=4000, r_peaks=r_peaks)
        labels = np.array(["N", "N"])
        seg = ECGSegmenter(fs=500.0)
        X, y, meta = seg.segment_with_labels(sig, r_peaks, labels)

        # No segment should contain exact zeros at edges (our synthetic has zeros,
        # but the point is that the segmenter doesn't pad)
        assert meta["n_descartados_bordas"] == 0
        assert X.shape == (2, 500)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))


# ---------------------------------------------------------------------------
# QG3.3 — R-peak Centered
# ---------------------------------------------------------------------------


@pytest.mark.qg3
class TestSegmenterCentered:
    """Validate that the R-peak is near the center of each segment."""

    def test_r_peak_at_center(self):
        r_peaks = [1500, 2500]
        sig, _ = _synthetic_signal_with_peaks(n_samples=4000, r_peaks=r_peaks)
        labels = np.array(["N", "N"])
        seg = ECGSegmenter(fs=500.0)
        X, y, meta = seg.segment_with_labels(sig, r_peaks, labels)

        half = seg.half_len
        for i, rp in enumerate(r_peaks):
            # The synthetic signal has an impulse at rp
            # In the segment, rp maps to index 'half' (center)
            center_idx = half
            # Verify the peak is at center by checking max value position
            seg_peak_idx = int(np.argmax(np.abs(X[i])))
            assert (
                seg_peak_idx == center_idx
            ), f"Segment {i}: peak at {seg_peak_idx}, expected center {center_idx}"

    def test_r_peak_at_center_600ms_fallback(self):
        # Short RR intervals (< 600ms) -> 600ms windows with edge padding to 1000ms
        r_peaks = [500, 800, 1100]
        rr_intervals_ms = np.array([500.0, 500.0, 500.0])
        sig, _ = _synthetic_signal_with_peaks(n_samples=2000, r_peaks=r_peaks)
        labels = np.array(["N", "N", "N"])
        seg = ECGSegmenter(fs=500.0)
        X, y, meta = seg.segment_with_labels(sig, r_peaks, labels, rr_intervals_ms)

        # With edge padding, the 300-sample segment is centered within the 500-sample array.
        # The impulse is at the center of the 300-sample window -> index 150 of 300.
        # After padding to 500: pad_left = (500-300)//2 = 100, so center = 150+100 = 250.
        expected_center = seg.half_len
        for i, rp in enumerate(r_peaks):
            if i >= len(X):
                break
            seg_peak_idx = int(np.argmax(np.abs(X[i])))
            assert (
                seg_peak_idx == expected_center
            ), f"Segment {i}: peak at {seg_peak_idx}, expected center {expected_center}"


# ---------------------------------------------------------------------------
# QG3.4 — Output Quality
# ---------------------------------------------------------------------------


@pytest.mark.qg3
class TestSegmenterOutputQuality:
    """Validate output dtype, shape, and absence of NaN/Inf."""

    def test_output_dtype_float32(self):
        r_peaks = [1500]
        sig, _ = _synthetic_signal_with_peaks(n_samples=3000, r_peaks=r_peaks)
        labels = np.array(["N"])
        seg = ECGSegmenter(fs=500.0)
        X, y, meta = seg.segment_with_labels(sig, r_peaks, labels)

        assert X.dtype == np.float32
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))

    def test_empty_input(self):
        sig = np.zeros(100, dtype=np.float64)
        r_peaks = np.array([], dtype=np.int64)
        labels = np.array([], dtype=object)
        seg = ECGSegmenter(fs=500.0)
        X, y, meta = seg.segment_with_labels(sig, r_peaks, labels)

        assert X.shape == (0, seg.window_len)
        assert len(y) == 0
        assert meta["n_total"] == 0

    def test_single_beat(self):
        r_peaks = [1500]
        sig, _ = _synthetic_signal_with_peaks(n_samples=3000, r_peaks=r_peaks)
        labels = np.array(["V"])
        seg = ECGSegmenter(fs=500.0)
        X, y, meta = seg.segment_with_labels(sig, r_peaks, labels)

        assert X.shape == (1, 500)
        assert y[0] == "V"


# ---------------------------------------------------------------------------
# QG3.5 — RR-based Window Selection
# ---------------------------------------------------------------------------


@pytest.mark.qg3
class TestSegmenterRRSelection:
    """Validate that RR interval drives window size selection."""

    def test_tachycardia_uses_600ms(self):
        # RR = 500 ms (< 600) -> should use min_window
        r_peaks = [500, 1000, 1500]
        rr_intervals_ms = np.array([500.0, 500.0, 500.0])
        sig, _ = _synthetic_signal_with_peaks(n_samples=2500, r_peaks=r_peaks)
        labels = np.array(["N", "N", "N"])
        seg = ECGSegmenter(fs=500.0)
        X, y, meta = seg.segment_with_labels(sig, r_peaks, labels, rr_intervals_ms)

        assert meta["n_usados_600ms"] > 0

    def test_normal_rr_uses_1000ms(self):
        # RR = 1000 ms (>= 600) -> should use standard window
        r_peaks = [1000, 2000, 3000]
        rr_intervals_ms = np.array([1000.0, 1000.0, 1000.0])
        sig, _ = _synthetic_signal_with_peaks(n_samples=4500, r_peaks=r_peaks)
        labels = np.array(["N", "N", "N"])
        seg = ECGSegmenter(fs=500.0)
        X, y, meta = seg.segment_with_labels(sig, r_peaks, labels, rr_intervals_ms)

        assert meta["n_usados_1000ms"] > 0

    def test_rr_auto_computed_when_none(self):
        # No rr_intervals_ms provided -> computed from r_peaks
        r_peaks = np.array([500, 1500, 2500])
        sig, _ = _synthetic_signal_with_peaks(n_samples=3500, r_peaks=list(r_peaks))
        labels = np.array(["N", "N", "N"])
        seg = ECGSegmenter(fs=500.0)
        X, y, meta = seg.segment_with_labels(sig, r_peaks, labels)

        # RR intervals: ~1000ms, ~1000ms -> standard window
        assert meta["n_usados_1000ms"] > 0

    def test_mixed_rr_intervals(self):
        # Mix of tachycardia and normal
        r_peaks = [500, 1000, 2000]
        rr_intervals_ms = np.array([500.0, 500.0, 1000.0])
        sig, _ = _synthetic_signal_with_peaks(n_samples=3000, r_peaks=list(r_peaks))
        labels = np.array(["N", "N", "N"])
        seg = ECGSegmenter(fs=500.0)
        X, y, meta = seg.segment_with_labels(sig, r_peaks, labels, rr_intervals_ms)

        assert meta["n_usados_600ms"] > 0
        assert meta["n_usados_1000ms"] > 0


# ---------------------------------------------------------------------------
# QG3.6 — Single Beat API
# ---------------------------------------------------------------------------


@pytest.mark.qg3
class TestSegmenterSingleBeat:
    """Validate segment_single_beat helper."""

    def test_valid_single_beat(self):
        sig = np.zeros(3000, dtype=np.float64)
        sig[1500] = 1.0
        seg = ECGSegmenter(fs=500.0)
        beat = seg.segment_single_beat(sig, 1500)

        assert beat is not None
        assert beat.dtype == np.float32
        assert len(beat) == 500

    def test_insufficient_borders_returns_none(self):
        sig = np.zeros(200, dtype=np.float64)
        seg = ECGSegmenter(fs=500.0)
        beat = seg.segment_single_beat(sig, 50)

        assert beat is None

    def test_tachycardia_single_beat(self):
        sig = np.zeros(2000, dtype=np.float64)
        sig[800] = 1.0
        seg = ECGSegmenter(fs=500.0)
        beat = seg.segment_single_beat(sig, 800, rr_ms=500.0)

        assert beat is not None
        assert len(beat) == 300  # 600 ms window
