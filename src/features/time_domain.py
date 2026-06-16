"""ECG time-domain features — RR-interval based metrics.

Regras mandatórias (ecg-preprocessing-pipeline + Camada-03 spec §3.5):
- rr_prev, rr_next, rr_ratio, rr_local_mean, rr_local_std, rmssd, heart_rate
- Todos em unidades fisiológicas (ms, BPM, adimensional)
- Sem NaN/Inf em output
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

LOGGER = logging.getLogger("lewis.camada03.time_domain")


class TimeDomainFeatures:
    """Extract time-domain features from R-peak positions.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz. Default 500.0.
    """

    def __init__(self, fs: float = 500.0):
        self.fs = fs

    def extract(
        self,
        r_peaks: np.ndarray,
        fs: Optional[float] = None,
    ) -> List[Dict[str, float]]:
        """Extract time-domain features for each beat.

        Parameters
        ----------
        r_peaks : np.ndarray
            Sample indices of R-peaks.
        fs : float, optional
            Override sampling frequency.

        Returns
        -------
        List[Dict[str, float]]
            One dict per beat with keys:
            {
                "rr_prev": float,       # ms
                "rr_next": float,       # ms
                "rr_ratio": float,      # adimensional
                "rr_local_mean": float, # ms (5-beat window)
                "rr_local_std": float,  # ms (5-beat window)
                "rmssd": float,         # ms
                "heart_rate": float,    # BPM
            }
        """
        fs = fs if fs is not None else self.fs
        r_peaks = np.asarray(r_peaks, dtype=np.int64)
        n_beats = len(r_peaks)

        if n_beats == 0:
            return []

        # RR intervals in samples, then convert to ms
        rr_samples = np.diff(r_peaks)
        rr_ms = rr_samples / fs * 1000.0

        features: List[Dict[str, float]] = []

        for i in range(n_beats):
            # rr_prev: interval BEFORE this beat (beat i-1 → i)
            rr_prev = float(rr_ms[i - 1]) if i > 0 else 0.0

            # rr_next: interval AFTER this beat (beat i → i+1)
            rr_next = float(rr_ms[i]) if i < len(rr_ms) else 0.0

            # rr_ratio: prev / next
            rr_ratio = rr_prev / rr_next if rr_next > 0 else 1.0

            # Local window: max(0, i-2) to min(n-1, i+2) → up to 5 beats
            local_start = max(0, i - 2)
            local_end = min(len(rr_ms), i + 3)  # rr_ms has n_beats-1 elements
            local_rrs = rr_ms[local_start:local_end]

            rr_local_mean = float(np.mean(local_rrs)) if len(local_rrs) > 0 else 0.0
            rr_local_std = float(np.std(local_rrs)) if len(local_rrs) > 1 else 0.0

            # RMSSD: root mean square of successive differences
            # Window: max(0, i-4) to min(n-1, i) → up to 5 successive intervals
            rmssd_start = max(0, i - 4)
            rmssd_end = min(len(rr_ms), i + 1)
            rmssd_rrs = rr_ms[rmssd_start:rmssd_end]
            if len(rmssd_rrs) >= 2:
                diffs = np.diff(rmssd_rrs)
                rmssd = float(np.sqrt(np.mean(diffs**2)))
            else:
                rmssd = 0.0

            # Heart rate: 60000 / rr_prev (BPM)
            heart_rate = 60000.0 / rr_prev if rr_prev > 0 else 0.0

            features.append(
                {
                    "rr_prev": rr_prev,
                    "rr_next": rr_next,
                    "rr_ratio": rr_ratio,
                    "rr_local_mean": rr_local_mean,
                    "rr_local_std": rr_local_std,
                    "rmssd": rmssd,
                    "heart_rate": heart_rate,
                }
            )

        LOGGER.info(
            "Time-domain: %d beats, mean HR=%.1f BPM",
            n_beats,
            np.mean([f["heart_rate"] for f in features if f["heart_rate"] > 0]) or 0.0,
        )
        return features
