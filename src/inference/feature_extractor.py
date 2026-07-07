"""Feature extractor for inference-time MLP inputs.

Extracts the same 13 morphological + time-domain features used during training
of ``stage1_mlp_features_v2.1.keras``.  The extractor is intentionally light so
that it can be ported to C for the firmware (QG8/QG9).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from src.features.morphological import MorphologicalFeatures
from src.features.time_domain import TimeDomainFeatures

LOGGER = logging.getLogger("lewis.inference.feature_extractor")

FEATURE_NAMES: List[str] = [
    "rr_prev",
    "rr_next",
    "rr_ratio",
    "rr_local_mean",
    "rr_local_std",
    "rmssd",
    "heart_rate",
    "r_amplitude",
    "q_depth",
    "t_amplitude",
    "qrs_width_ms",
    "qrs_area",
    "st_slope_mV_s",
    "qrs_asymmetry_index",
    "t_r_ratio",
    "qrs_raggedness",
]


class FeatureExtractor:
    """Extract features from ECG beat segments.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz. Default 500.0.
    """

    def __init__(self, fs: float = 500.0):
        self.fs = fs
        self.morph = MorphologicalFeatures(fs=fs)
        self.temporal = TimeDomainFeatures(fs=fs)

    def extract_from_segments(
        self,
        segments: np.ndarray,
        r_peaks: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract features from raw beat segments.

        Parameters
        ----------
        segments : np.ndarray
            Beat segments with shape ``(n, 500)`` or ``(n, 500, 1)``.
        r_peaks : np.ndarray, optional
            Global R-peak sample indices for the whole record.  If provided,
            time-domain (RR) features are computed from them; otherwise RR
            features are zero-filled and a warning is emitted.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with one array per feature in ``FEATURE_NAMES``.
        """
        segments = np.asarray(segments, dtype=np.float32)
        if segments.ndim == 3:
            segments = segments[..., 0]
        if segments.ndim != 2:
            raise ValueError(
                f"segments must be (n, 500) or (n, 500, 1); got {segments.shape}"
            )

        n = segments.shape[0]

        # Morphological features from individual segments.
        morph_list = self.morph.extract(segments, fs=self.fs)

        # Time-domain features require R-peak positions in the full record.
        if r_peaks is not None and len(r_peaks) == n:
            temporal_list = self.temporal.extract(r_peaks, fs=self.fs)
        else:
            if r_peaks is not None and len(r_peaks) != n:
                LOGGER.warning(
                    "r_peaks length (%d) differs from segments (%d); "
                    "zero-filling RR features",
                    len(r_peaks),
                    n,
                )
            elif r_peaks is None:
                LOGGER.warning(
                    "No r_peaks provided; RR features are zero-filled. "
                    "Supply pre-computed features for best accuracy."
                )
            temporal_list = [
                {
                    "rr_prev": 0.0,
                    "rr_next": 0.0,
                    "rr_ratio": 1.0,
                    "rr_local_mean": 0.0,
                    "rr_local_std": 0.0,
                    "rmssd": 0.0,
                    "heart_rate": 0.0,
                }
                for _ in range(n)
            ]

        features: Dict[str, np.ndarray] = {}
        for name in FEATURE_NAMES:
            if name in {
                "rr_prev",
                "rr_next",
                "rr_ratio",
                "rr_local_mean",
                "rr_local_std",
                "rmssd",
                "heart_rate",
            }:
                features[name] = np.array([t[name] for t in temporal_list], dtype=np.float32)
            else:
                features[name] = np.array([m[name] for m in morph_list], dtype=np.float32)

        return features

    @staticmethod
    def features_to_array(
        features: Dict[str, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Stack feature dict into a 2-D array.

        Parameters
        ----------
        features : dict[str, np.ndarray]
            Dictionary of feature arrays.
        feature_names : list[str], optional
            Order of columns. Defaults to ``FEATURE_NAMES``.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples, n_features)``.
        """
        names = feature_names or FEATURE_NAMES
        return np.column_stack([features[name] for name in names]).astype(np.float32)
