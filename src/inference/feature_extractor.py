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
        record_ids: Optional[np.ndarray] = None,
        precomputed_temporal: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract features from raw beat segments.

        Parameters
        ----------
        segments : np.ndarray
            Beat segments with shape ``(n, 500)`` or ``(n, 500, 1)``.
        r_peaks : np.ndarray, optional
            Global R-peak sample indices for each beat. Used together with
            ``record_ids`` to compute time-domain features when
            ``precomputed_temporal`` is not provided.
        record_ids : np.ndarray, optional
            Identifier of the source record for each beat.
        precomputed_temporal : list[dict], optional
            Pre-computed time-domain features (rr_prev, rr_next, etc.) for
            each beat. When provided, it takes precedence and avoids
            re-computing RR features from sparse R-peak subsets.

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

        # Time-domain features: prefer pre-computed, fall back to extraction.
        if precomputed_temporal is not None:
            if len(precomputed_temporal) != n:
                raise ValueError(
                    f"precomputed_temporal length ({len(precomputed_temporal)}) "
                    f"differs from segments ({n})"
                )
            temporal_list = precomputed_temporal
        elif r_peaks is not None and len(r_peaks) == n:
            if record_ids is not None and len(record_ids) == n:
                temporal_list = self._extract_temporal_per_record(r_peaks, record_ids)
            else:
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
                    "No r_peaks or precomputed_temporal provided; RR features are zero-filled. "
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

    def _extract_temporal_per_record(
        self,
        r_peaks: np.ndarray,
        record_ids: np.ndarray,
    ) -> List[Dict[str, float]]:
        """Compute time-domain features grouped by record, preserving input order.

        .. warning::
            This helper assumes ``r_peaks`` contains *consecutive* R-peaks of
            each record. If the input is a sparse subset (e.g. only abnormal
            beats), the computed RR features will be wrong. Use
            ``precomputed_temporal`` in that case.
        """
        r_peaks = np.asarray(r_peaks, dtype=np.int64)
        record_ids = np.asarray(record_ids)
        order = np.arange(len(record_ids))
        temporal: Dict[int, Dict[str, float]] = {}
        for rec in np.unique(record_ids):
            mask = record_ids == rec
            idx = order[mask]
            # Sort by r_peak within record to get correct RR sequence.
            sort_order = np.argsort(r_peaks[idx])
            sorted_idx = idx[sort_order]
            sorted_r_peaks = r_peaks[sorted_idx]
            feats_sorted = self.temporal.extract(sorted_r_peaks, fs=self.fs)
            for original_pos, feat in zip(sorted_idx, feats_sorted):
                temporal[int(original_pos)] = feat
        return [temporal[i] for i in range(len(record_ids))]

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
