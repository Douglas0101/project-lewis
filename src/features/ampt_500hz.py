"""AMPT detector @ 500 Hz — AccYouRate Modified Pan-Tompkins (Neri et al., 2023).

Reference: https://github.com/Accyourate-Group-S-p-A/acy_ampt

Regras mandatórias (ecg-preprocessing-pipeline + spec v1.1 §4.4):
- Bandpass: 5–15 Hz (Butterworth 2nd order, filtfilt zero-phase)
- Derivative: 5-point (Pan-Tompkins standard)
- Squaring: point-by-point
- Moving Window Integration (MWI): 150 ms = 75 amostras @ 500 Hz
- Refratariedade / T-wave discrimination: 360 ms = 180 amostras @ 500 Hz
- Adaptive thresholding (SPKF, NPKF, THRESHOLDF1, THRESHOLDF2)
- Search-back for missed QRS (1.66 × RR_average1)
- Tolerância de detecção: 150 ms (75 amostras @ 500 Hz)

NÃO usar biosppy, heartpy, neurokit2 — implementar do zero para auditabilidade.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal

LOGGER = logging.getLogger("lewis.camada03.ampt")

# Tolerância padrão AAMI/PhysioNet para detecção de batimento
TOL_MS_DEFAULT = 150.0


class AMPTDetector:
    """AccYouRate Modified Pan-Tompkins detector.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz. Default 500.0.
    """

    def __init__(self, fs: float = 500.0):
        self.fs = fs
        self.tol_samples = int(round(TOL_MS_DEFAULT * fs / 1000.0))

        # 1. Bandpass: 5–15 Hz, 2nd order Butterworth, zero-phase filtfilt
        self.b_band, self.a_band = signal.butter(2, [5.0 / (fs / 2), 15.0 / (fs / 2)], btype="band")

        # 4. MWI window: 150 ms
        self.mwi_window = int(round(0.150 * fs))
        if self.mwi_window % 2 == 0:
            self.mwi_window += 1  # ensure odd for symmetric convolution

        # 5. Refractory / T-wave discrimination: 360 ms
        self.refractory = int(round(0.360 * fs))

        # Search-back factor
        self.search_back_factor = 1.66

        LOGGER.debug(
            "AMPTDetector fs=%.1f | tol=%d samples | mwi=%d | refractory=%d",
            fs,
            self.tol_samples,
            self.mwi_window,
            self.refractory,
        )

    def _derivative(self, x: np.ndarray) -> np.ndarray:
        """5-point derivative (Pan-Tompkins standard)."""
        # h = [-1, -2, 0, 2, 1] / 8  (normalized for unit gain at DC)
        h = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / 8.0
        return np.convolve(x, h, mode="same")

    def _squaring(self, x: np.ndarray) -> np.ndarray:
        """Point-by-point squaring."""
        return x**2

    def _mwi(self, x: np.ndarray) -> np.ndarray:
        """Moving window integration with rectangular window (150 ms)."""
        window = np.ones(self.mwi_window) / self.mwi_window
        return np.convolve(x, window, mode="same")

    def _adaptive_thresholds(
        self,
        mwi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute adaptive thresholds THRESHOLDF1 and THRESHOLDF2.

        Returns
        -------
        thresh1, thresh2 : np.ndarray
            Threshold arrays same length as mwi.
        """
        # Initialize thresholds
        spkf = 0.0  # Signal peak (QRS) estimate
        npkf = 0.0  # Noise peak estimate
        thresh1 = np.zeros_like(mwi)
        thresh2 = np.zeros_like(mwi)

        # Use first 2 seconds to initialize SPKF and NPKF
        init_len = min(int(2.0 * self.fs), len(mwi) // 4)
        if init_len > 10:
            init_max = np.max(mwi[:init_len])
            spkf = init_max * 0.25
            npkf = init_max * 0.125

        for i in range(len(mwi)):
            thresh1[i] = npkf + 0.25 * (spkf - npkf)
            thresh2[i] = 0.5 * thresh1[i]

        return thresh1, thresh2

    def detect(self, sig: np.ndarray) -> np.ndarray:
        """Detect R-peaks using AMPT algorithm.

        Parameters
        ----------
        sig : np.ndarray
            ECG signal 1-D, already preprocessed (filtered, detrended, normalized).

        Returns
        -------
        np.ndarray
            Sample indices of detected R-peaks.
        """
        if sig.ndim != 1:
            raise ValueError(f"sig must be 1-D, got shape {sig.shape}")

        n = len(sig)
        if n < self.mwi_window * 2:
            LOGGER.warning("Signal too short for AMPT: %d samples", n)
            return np.array([], dtype=np.int64)

        # 1. Bandpass filter (zero-phase)
        bp = signal.filtfilt(self.b_band, self.a_band, sig)

        # 2. 5-point derivative
        der = self._derivative(bp)

        # 3. Squaring
        sq = self._squaring(der)

        # 4. Moving window integration
        mwi = self._mwi(sq)

        # 5. Adaptive thresholding + peak detection
        peaks = self._detect_peaks_adaptive(mwi)

        # 6. T-wave discrimination (slope check within refractory window)
        peaks = self._t_wave_discrimination(bp, peaks)

        LOGGER.info("AMPT detected %d peaks from %d samples", len(peaks), n)
        return np.array(peaks, dtype=np.int64)

    def _detect_peaks_adaptive(self, mwi: np.ndarray) -> List[int]:
        """Adaptive thresholding with search-back.

        Implements the Pan-Tompkins adaptive threshold logic:
        - Track signal peaks (SPKF) and noise peaks (NPKF)
        - Two thresholds: THRESHOLDF1 (primary), THRESHOLDF2 (search-back)
        - Search-back detects missed QRS complexes
        """
        n = len(mwi)
        peaks: List[int] = []
        rrs: List[int] = []  # RR intervals in samples

        # Initialize
        spkf = 0.0
        npkf = 0.0
        rr_average1 = 0.0  # Average RR (last 8 beats)

        # Initialize from first 2 seconds
        init_len = min(int(2.0 * self.fs), n // 4)
        if init_len > 10:
            init_max = float(np.max(mwi[:init_len]))
            spkf = init_max * 0.25
            npkf = init_max * 0.125
            rr_average1 = int(0.8 * self.fs)  # ~800 ms default

        i = 0
        last_peak = -self.refractory

        while i < n:
            # Find local max in a small window
            window_end = min(i + self.refractory, n)
            local_max_idx = int(np.argmax(mwi[i:window_end])) + i
            local_max_val = float(mwi[local_max_idx])

            thresh1 = npkf + 0.25 * (spkf - npkf)
            thresh2 = 0.5 * thresh1

            if local_max_val > thresh1:
                # Potential QRS
                if local_max_idx - last_peak >= self.refractory:
                    peaks.append(local_max_idx)
                    # Update RR
                    if len(peaks) >= 2:
                        rr = peaks[-1] - peaks[-2]
                        rrs.append(rr)
                        # Update RR averages (last 8)
                        recent_rrs = rrs[-8:]
                        rr_average1 = float(np.mean(recent_rrs))
                        # RR average2: only RRs within 92%-116% of rr_average1
                        valid_rrs = [
                            r for r in recent_rrs if 0.92 * rr_average1 <= r <= 1.16 * rr_average1
                        ]
                        if valid_rrs:
                            _ = float(np.mean(valid_rrs))  # mantido para clareza do algoritmo
                    # Update SPKF
                    spkf = 0.125 * local_max_val + 0.875 * spkf
                    last_peak = local_max_idx
                    i = local_max_idx + 1
                else:
                    # Within refractory — likely T-wave
                    npkf = 0.125 * local_max_val + 0.875 * npkf
                    i = local_max_idx + 1
            elif local_max_val > thresh2 and len(peaks) > 0:
                # Search-back: missed QRS?
                expected_rr = self.search_back_factor * rr_average1
                if local_max_idx - last_peak > expected_rr:
                    peaks.append(local_max_idx)
                    if len(peaks) >= 2:
                        rr = peaks[-1] - peaks[-2]
                        rrs.append(rr)
                        recent_rrs = rrs[-8:]
                        rr_average1 = float(np.mean(recent_rrs))
                        valid_rrs = [
                            r for r in recent_rrs if 0.92 * rr_average1 <= r <= 1.16 * rr_average1
                        ]
                        if valid_rrs:
                            _ = float(np.mean(valid_rrs))  # mantido para clareza do algoritmo
                    spkf = 0.125 * local_max_val + 0.875 * spkf
                    last_peak = local_max_idx
                    i = local_max_idx + 1
                else:
                    npkf = 0.125 * local_max_val + 0.875 * npkf
                    i = local_max_idx + 1
            else:
                # Noise peak
                if local_max_val > 0:
                    npkf = 0.125 * local_max_val + 0.875 * npkf
                i = window_end

        return peaks

    def _t_wave_discrimination(self, bp: np.ndarray, peaks: List[int]) -> List[int]:
        """Discriminate T-waves using slope check within refractory window.

        For peaks that occur within the refractory period of a previous peak,
        keep the one with the steeper bandpass slope (QRS) and discard the
        other (T-wave).
        """
        if len(peaks) < 2:
            return peaks

        win = max(2, int(round(0.040 * self.fs)))

        def slope_at(idx: int) -> float:
            if idx - win < 0 or idx + win >= len(bp):
                return 0.0
            return float(np.max(np.abs(np.diff(bp[idx - win : idx + win + 1]))))

        filtered: List[int] = [peaks[0]]
        for current in peaks[1:]:
            previous = filtered[-1]
            if current - previous < self.refractory:
                # Overlapping candidates — keep the steeper one.
                if slope_at(current) > slope_at(previous):
                    filtered[-1] = current
                # else discard current
            else:
                filtered.append(current)

        return filtered

    def evaluate(
        self,
        sig: np.ndarray,
        r_true: np.ndarray,
        tol_ms: float = TOL_MS_DEFAULT,
    ) -> Dict[str, float]:
        """Evaluate detector against ground-truth annotations.

        Parameters
        ----------
        sig : np.ndarray
            ECG signal.
        r_true : np.ndarray
            Ground-truth R-peak sample indices.
        tol_ms : float
            Tolerance in milliseconds for matching peaks.

        Returns
        -------
        dict
            {"TP": int, "FN": int, "FP": int, "Sens": float, "PPV": float, "F1": float}
        """
        r_det = self.detect(sig)
        tol_samples = int(round(tol_ms * self.fs / 1000.0))

        tp = 0
        matched_det = set()
        matched_true = set()

        for i, rt in enumerate(r_true):
            for j, rd in enumerate(r_det):
                if j in matched_det:
                    continue
                if abs(int(rt) - int(rd)) <= tol_samples:
                    tp += 1
                    matched_det.add(j)
                    matched_true.add(i)
                    break

        fn = len(r_true) - len(matched_true)
        fp = len(r_det) - len(matched_det)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) > 0 else 0.0

        return {
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "Sens": sens,
            "PPV": ppv,
            "F1": f1,
            "tol_ms": tol_ms,
            "n_det": len(r_det),
            "n_true": len(r_true),
        }
