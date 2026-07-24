"""Leakage-free E06 features for AAMI fusion beats.

AAMI class ``F`` is a fusion of a ventricular and a normal beat.  This module
implements the first reopened E06 hypothesis: stateless direct QRS morphology.
It deliberately contains no label-fitted state, sampling, scaling, imputation,
or future-beat context.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FusionMorphologyConfig(BaseModel):
    """Validated extraction configuration for E06R-H1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fs_hz: float = Field(default=500.0, gt=0.0)
    qrs_pre_ms: float = Field(default=120.0, gt=0.0)
    qrs_post_ms: float = Field(default=180.0, gt=0.0)
    crest_short_ms: float = Field(default=180.0, gt=0.0)
    crest_long_ms: float = Field(default=400.0, gt=0.0)
    s_search_ms: float = Field(default=120.0, gt=0.0)
    epsilon: float = Field(default=1.0e-8, gt=0.0)

    @model_validator(mode="after")
    def validate_windows(self) -> FusionMorphologyConfig:
        """Keep the feature windows nested and within the 1000 ms beat."""
        qrs_total = self.qrs_pre_ms + self.qrs_post_ms
        if self.crest_short_ms > self.crest_long_ms:
            raise ValueError("crest_short_ms must not exceed crest_long_ms")
        if qrs_total > self.crest_long_ms:
            raise ValueError("QRS window must fit inside crest_long_ms")
        if self.crest_long_ms > 1000.0:
            raise ValueError("crest_long_ms must fit inside the 1000 ms beat")
        return self


class E06FeatureDefinition(BaseModel):
    """One immutable feature-contract entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    family: Literal["direct_qrs_morphology"] = "direct_qrs_morphology"
    units: str
    requires_previous_context: bool = False
    requires_future_context: bool = False


class E06FeatureSchema(BaseModel):
    """Hashed schema manifest for one E06 representation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    mode: Literal["causal"]
    features: tuple[E06FeatureDefinition, ...]
    extraction_config: FusionMorphologyConfig
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


_FEATURE_DEFINITIONS = (
    E06FeatureDefinition(name="qrs_peak_to_peak", units="mV"),
    E06FeatureDefinition(name="qrs_signed_area", units="mV*s"),
    E06FeatureDefinition(name="qrs_abs_area", units="mV*s"),
    E06FeatureDefinition(name="qrs_energy", units="mV^2"),
    E06FeatureDefinition(name="qrs_activity", units="mV"),
    E06FeatureDefinition(name="qrs_mobility", units="ratio"),
    E06FeatureDefinition(name="qrs_max_upslope", units="mV/s"),
    E06FeatureDefinition(name="qrs_max_downslope", units="mV/s"),
    E06FeatureDefinition(name="qrs_s_amplitude", units="mV"),
    E06FeatureDefinition(name="qrs_r_s_ratio", units="ratio"),
    E06FeatureDefinition(name="qrs_zero_crossings", units="count"),
    E06FeatureDefinition(name="qrs_centroid_ms", units="ms"),
    E06FeatureDefinition(name="qrs_early_energy_fraction", units="ratio"),
    E06FeatureDefinition(name="qrs_late_energy_fraction", units="ratio"),
    E06FeatureDefinition(name="crest_factor_180", units="ratio"),
    E06FeatureDefinition(name="crest_factor_400", units="ratio"),
    E06FeatureDefinition(name="beat_energy_400", units="mV^2"),
    E06FeatureDefinition(name="beat_skewness_400", units="ratio"),
    E06FeatureDefinition(name="beat_kurtosis_400", units="ratio"),
)

DIRECT_MORPHOLOGY_FEATURE_NAMES = tuple(feature.name for feature in _FEATURE_DEFINITIONS)


def build_direct_morphology_schema(
    config: FusionMorphologyConfig | None = None,
) -> E06FeatureSchema:
    """Build the immutable, content-addressed E06R-H1 schema."""
    resolved = config or FusionMorphologyConfig()
    payload = {
        "version": "e06r-h1-v1",
        "mode": "causal",
        "features": [feature.model_dump(mode="json") for feature in _FEATURE_DEFINITIONS],
        "extraction_config": resolved.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return E06FeatureSchema(
        version="e06r-h1-v1",
        mode="causal",
        features=_FEATURE_DEFINITIONS,
        extraction_config=resolved,
        schema_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _as_signal_matrix(signals: np.ndarray) -> np.ndarray:
    values = np.asarray(signals, dtype=np.float32)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"signals must be 2-D or 3-D with one channel; got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("signals contain NaN or Inf")
    return values


def _centered_windows(
    signals: np.ndarray,
    r_peaks: np.ndarray,
    pre_samples: int,
    post_samples: int,
) -> np.ndarray:
    if pre_samples <= 0 or post_samples <= 0:
        raise ValueError("window lengths must be positive")
    starts = r_peaks - pre_samples
    ends = r_peaks + post_samples
    if np.any(starts < 0) or np.any(ends > signals.shape[1]):
        raise ValueError("requested feature window would require prohibited zero padding")
    offsets = np.arange(-pre_samples, post_samples, dtype=np.int64)
    indices = r_peaks[:, np.newaxis] + offsets[np.newaxis, :]
    rows = np.arange(signals.shape[0], dtype=np.int64)[:, np.newaxis]
    return signals[rows, indices]


def _milliseconds_to_samples(milliseconds: float, fs_hz: float) -> int:
    try:
        return int(round(milliseconds * fs_hz / 1000.0))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("invalid time-to-sample conversion") from error


def _baseline_center(windows: np.ndarray, fs_hz: float) -> np.ndarray:
    baseline_samples = max(5, _milliseconds_to_samples(40.0, fs_hz))
    baseline_samples = min(baseline_samples, windows.shape[1])
    baseline = np.median(windows[:, :baseline_samples], axis=1, keepdims=True)
    return windows - baseline


def _crest_factor(windows: np.ndarray, epsilon: float) -> np.ndarray:
    peak = np.max(np.abs(windows), axis=1)
    rms = np.sqrt(np.mean(np.square(windows), axis=1))
    return peak / np.maximum(rms, epsilon)


def extract_direct_morphology(
    signals: np.ndarray,
    r_peak_indices: np.ndarray,
    config: FusionMorphologyConfig | None = None,
) -> np.ndarray:
    """Extract causal, stateless morphology features from aligned beat windows.

    No padding is performed.  A caller must supply the R-peak position inside
    every 1000 ms segment so that the same contract can be reproduced in
    firmware and online inference.
    """
    resolved = config or FusionMorphologyConfig()
    values = _as_signal_matrix(signals)
    r_peaks = np.asarray(r_peak_indices, dtype=np.int64)
    if r_peaks.ndim != 1:
        raise ValueError("r_peak_indices must be one-dimensional")
    if values.shape[0] != r_peaks.shape[0]:
        raise ValueError("signals and r_peak_indices must contain the same number of beats")

    fs_hz = resolved.fs_hz
    qrs_pre = _milliseconds_to_samples(resolved.qrs_pre_ms, fs_hz)
    qrs_post = _milliseconds_to_samples(resolved.qrs_post_ms, fs_hz)
    short_half = _milliseconds_to_samples(resolved.crest_short_ms / 2.0, fs_hz)
    long_half = _milliseconds_to_samples(resolved.crest_long_ms / 2.0, fs_hz)
    s_search = _milliseconds_to_samples(resolved.s_search_ms, fs_hz)

    qrs = _baseline_center(
        _centered_windows(values, r_peaks, qrs_pre, qrs_post),
        fs_hz,
    )
    short_window = _baseline_center(
        _centered_windows(values, r_peaks, short_half, short_half),
        fs_hz,
    )
    long_window = _baseline_center(
        _centered_windows(values, r_peaks, long_half, long_half),
        fs_hz,
    )

    epsilon = resolved.epsilon
    qrs_diff = np.diff(qrs, axis=1)
    qrs_energy_by_sample = np.square(qrs)
    total_energy = np.sum(qrs_energy_by_sample, axis=1)
    r_amplitude = qrs[:, qrs_pre]
    s_end = min(qrs.shape[1], qrs_pre + s_search + 1)
    s_amplitude = np.min(qrs[:, qrs_pre:s_end], axis=1)
    peak_to_peak = np.ptp(qrs, axis=1)
    s_denominator = np.maximum(
        np.abs(s_amplitude),
        np.maximum(0.05 * peak_to_peak, epsilon),
    )

    smoothed_cumsum = np.pad(
        np.cumsum(qrs, axis=1),
        ((0, 0), (1, 0)),
        mode="constant",
    )
    smooth_width = 5
    smoothed = (
        smoothed_cumsum[:, smooth_width:] - smoothed_cumsum[:, :-smooth_width]
    ) / smooth_width
    zero_crossings = np.sum(
        np.signbit(smoothed[:, 1:]) != np.signbit(smoothed[:, :-1]),
        axis=1,
    )

    offsets_ms = np.arange(-qrs_pre, qrs_post, dtype=np.float64) / fs_hz * 1000.0
    abs_qrs = np.abs(qrs)
    abs_sum = np.sum(abs_qrs, axis=1)
    centroid_ms = np.sum(abs_qrs * offsets_ms[np.newaxis, :], axis=1) / np.maximum(abs_sum, epsilon)

    early_energy = np.sum(qrs_energy_by_sample[:, :qrs_pre], axis=1)
    late_energy = np.sum(qrs_energy_by_sample[:, qrs_pre:], axis=1)
    long_mean = np.mean(long_window, axis=1, keepdims=True)
    long_centered = long_window - long_mean
    long_std = np.sqrt(np.mean(np.square(long_centered), axis=1))
    standardized = long_centered / np.maximum(long_std[:, np.newaxis], epsilon)

    output = np.column_stack(
        [
            peak_to_peak,
            np.sum(qrs, axis=1) / fs_hz,
            np.sum(abs_qrs, axis=1) / fs_hz,
            np.mean(qrs_energy_by_sample, axis=1),
            np.sum(np.abs(qrs_diff), axis=1),
            np.sqrt(np.mean(np.square(qrs_diff), axis=1))
            / np.maximum(np.sqrt(np.mean(qrs_energy_by_sample, axis=1)), epsilon),
            np.max(qrs_diff, axis=1) * fs_hz,
            np.abs(np.min(qrs_diff, axis=1) * fs_hz),
            s_amplitude,
            r_amplitude / s_denominator,
            zero_crossings,
            centroid_ms,
            early_energy / np.maximum(total_energy, epsilon),
            late_energy / np.maximum(total_energy, epsilon),
            _crest_factor(short_window, epsilon),
            _crest_factor(long_window, epsilon),
            np.mean(np.square(long_window), axis=1),
            np.mean(np.power(standardized, 3), axis=1),
            np.mean(np.power(standardized, 4), axis=1) - 3.0,
        ]
    )
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    return output.astype(np.float32, copy=False)
