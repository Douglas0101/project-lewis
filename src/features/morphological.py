"""ECG morphological features — QRS width, ST slope, amplitudes.

Regras mandatórias (ecg-preprocessing-pipeline + Camada-03 spec §3.6):
- QRS width via envelope method (onset 300ms before, offset 150ms after, threshold 50% |R|)
- ST slope: J+60ms → J+80ms (ACC/AHA/HRS standard)
- q_depth: min in 100ms before R
- t_amplitude: max in 300ms after R
- qrs_area: trapezoid of |QRS|
- Sem NaN/Inf — marcar NaN apenas se onset ≥ offset (falha de detecção)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy import signal

LOGGER = logging.getLogger("lewis.camada03.morphological")


class MorphologicalFeatures:
    """Extract morphological features from ECG beat segments.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz. Default 500.0.
    """

    def __init__(self, fs: float = 500.0):
        self.fs = fs

    def _envelope(self, seg: np.ndarray, fs: float) -> np.ndarray:
        """Compute signal envelope using Hilbert transform (bandpass 5-30 Hz)."""
        # Bandpass 5-30 Hz for envelope
        nyq = fs / 2.0
        b, a = signal.butter(2, [5.0 / nyq, 30.0 / nyq], btype="band")
        filtered = signal.filtfilt(b, a, seg)
        # Analytic signal envelope
        analytic = signal.hilbert(filtered)
        return np.abs(analytic)

    def _find_qrs_onset_offset(
        self,
        seg: np.ndarray,
        r_idx: int,
        r_amp: float,
        fs: float,
    ) -> tuple[int, int]:
        """Find QRS onset and offset using envelope method.

        Returns (onset, offset) sample indices within segment.
        If detection fails, returns (-1, -1).
        """
        envelope = self._envelope(seg, fs)
        threshold = 0.5 * abs(r_amp)

        # Onset: search 300ms before R, last point where envelope < threshold
        onset_search_start = max(0, r_idx - int(round(0.300 * fs)))
        onset_region = envelope[onset_search_start:r_idx]
        onset = onset_search_start
        if len(onset_region) > 0:
            below = np.where(onset_region < threshold)[0]
            if len(below) > 0:
                onset = onset_search_start + below[-1]
            else:
                onset = onset_search_start

        # Offset: search 150ms after R, first point where envelope < threshold
        offset_search_end = min(len(seg), r_idx + int(round(0.150 * fs)) + 1)
        offset_region = envelope[r_idx:offset_search_end]
        offset = offset_search_end - 1
        if len(offset_region) > 0:
            below = np.where(offset_region < threshold)[0]
            if len(below) > 0:
                offset = r_idx + below[0]
            else:
                offset = offset_search_end - 1

        if onset >= offset:
            return -1, -1
        return onset, offset

    def extract(
        self,
        segments: np.ndarray,
        fs: Optional[float] = None,
        r_idx: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Extract morphological features for each segment.

        Parameters
        ----------
        segments : np.ndarray
            Array shape (n_segments, window_len), float32, centered on R-peak.
        fs : float, optional
            Override sampling frequency.
        r_idx : int, optional
            Index of R-peak in segment. If None, uses argmax(abs(seg)).

        Returns
        -------
        List[Dict[str, float]]
            One dict per segment with keys:
            {
                "r_amplitude": float,      # mV
                "q_depth": float,          # mV (negative)
                "t_amplitude": float,      # mV
                "qrs_width_ms": float,     # ms (NaN if detection fails)
                "qrs_area": float,         # mV·s
                "st_slope_mV_s": float,    # mV/s
                "j_point": int,            # sample index
            }
        """
        fs = fs if fs is not None else self.fs
        n_segments = segments.shape[0]
        if n_segments == 0:
            return []

        # Default r_idx: center of segment (for 1000ms window @ 500Hz = 250)
        if r_idx is None:
            r_idx = segments.shape[1] // 2

        features: List[Dict[str, float]] = []

        for i in range(n_segments):
            seg = segments[i]

            # 1. R-peak amplitude
            actual_r_idx = int(np.argmax(np.abs(seg)))
            r_amplitude = float(seg[actual_r_idx])

            # 2. Q depth: min in 100ms before R
            q_start = max(0, actual_r_idx - int(round(0.100 * fs)))
            q_depth = float(np.min(seg[q_start:actual_r_idx])) if actual_r_idx > q_start else 0.0

            # 3. T amplitude: max in 300ms after R
            t_end = min(len(seg), actual_r_idx + int(round(0.300 * fs)) + 1)
            t_amplitude = float(np.max(seg[actual_r_idx:t_end])) if t_end > actual_r_idx else 0.0

            # 4. QRS onset/offset via envelope method
            onset, offset = self._find_qrs_onset_offset(seg, actual_r_idx, r_amplitude, fs)

            if onset >= 0 and offset > onset:
                qrs_width_ms = (offset - onset) / fs * 1000.0
                j_point = offset
                # 12. QRS area
                qrs_area = float(np.trapezoid(np.abs(seg[onset:offset]), dx=1.0 / fs))
            else:
                qrs_width_ms = np.nan
                j_point = actual_r_idx
                qrs_area = np.nan
                LOGGER.debug("Segment %d: QRS onset/offset detection failed", i)

            # 9-11. ST slope: J+60ms → J+80ms
            st_start = j_point + int(round(0.060 * fs))
            st_end = j_point + int(round(0.080 * fs))
            if st_end <= len(seg) and st_start < st_end:
                st_seg = seg[st_start:st_end]
                if len(st_seg) >= 2:
                    # Linear fit: slope in mV/sample
                    slope_per_sample = float(np.polyfit(np.arange(len(st_seg)), st_seg, 1)[0])
                    st_slope_mV_s = slope_per_sample * fs  # convert to mV/s
                else:
                    st_slope_mV_s = 0.0
            else:
                st_slope_mV_s = 0.0

            features.append(
                {
                    "r_amplitude": r_amplitude,
                    "q_depth": q_depth,
                    "t_amplitude": t_amplitude,
                    "qrs_width_ms": qrs_width_ms,
                    "qrs_area": qrs_area,
                    "st_slope_mV_s": st_slope_mV_s,
                    "j_point": j_point,
                }
            )

        n_valid = sum(1 for f in features if not np.isnan(f["qrs_width_ms"]))
        LOGGER.info("Morphological: %d segments, %d valid QRS widths", n_segments, n_valid)
        return features
