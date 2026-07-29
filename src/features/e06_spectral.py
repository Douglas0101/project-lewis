"""Spectral QRS representation for reopened E06 H9."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.signal import welch

# numpy 2.0 renomeou np.trapz -> np.trapezoid; o projeto está pinado em numpy <2.0
_trapezoid = getattr(np, "trapezoid", np.trapz)


class SpectralQRSConfig(BaseModel):
    """Immutable spectral-QRS configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fs_hz: float = Field(default=500.0, gt=0.0)
    qrs_pre_ms: float = Field(default=120.0, gt=0.0)
    qrs_post_ms: float = Field(default=180.0, gt=0.0)
    nperseg: int = Field(default=64, ge=16, le=256)
    epsilon: float = Field(default=1.0e-8, gt=0.0)

    @model_validator(mode="after")
    def validate_window(self) -> SpectralQRSConfig:
        if self.qrs_pre_ms + self.qrs_post_ms > 500.0:
            raise ValueError("spectral QRS window must not exceed 500 ms")
        return self


class SpectralQRSFeatureDefinition(BaseModel):
    """One spectral feature contract entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    family: Literal["spectral_qrs"] = "spectral_qrs"
    units: str
    requires_previous_context: bool = False
    requires_future_context: bool = False


class SpectralQRSSchema(BaseModel):
    """Content-addressed H9 schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    mode: Literal["causal"]
    features: tuple[SpectralQRSFeatureDefinition, ...]
    extraction_config: SpectralQRSConfig
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


_SPECTRAL_DEFINITIONS = (
    SpectralQRSFeatureDefinition(name="spectral_centroid", units="Hz"),
    SpectralQRSFeatureDefinition(name="spectral_bandwidth", units="Hz"),
    SpectralQRSFeatureDefinition(name="spectral_flatness", units="ratio"),
    SpectralQRSFeatureDefinition(name="spectral_rolloff_85", units="Hz"),
    SpectralQRSFeatureDefinition(name="spectral_peak_freq", units="Hz"),
    SpectralQRSFeatureDefinition(name="spectral_peak_power", units="dB"),
    SpectralQRSFeatureDefinition(name="spectral_power_0_10hz", units="ratio"),
    SpectralQRSFeatureDefinition(name="spectral_power_10_20hz", units="ratio"),
    SpectralQRSFeatureDefinition(name="spectral_power_20_40hz", units="ratio"),
    SpectralQRSFeatureDefinition(name="spectral_power_40_60hz", units="ratio"),
    SpectralQRSFeatureDefinition(name="spectral_power_60_100hz", units="ratio"),
    SpectralQRSFeatureDefinition(name="spectral_power_ratio_low_high", units="ratio"),
    SpectralQRSFeatureDefinition(name="spectral_entropy", units="nats"),
)

SPECTRAL_QRS_FEATURE_NAMES = tuple(feature.name for feature in _SPECTRAL_DEFINITIONS)


def _to_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} is not an integer") from error


def _milliseconds_to_samples(milliseconds: float, fs_hz: float) -> int:
    try:
        return int(round(milliseconds * fs_hz / 1000.0))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("invalid spectral time-to-sample conversion") from error


def build_spectral_qrs_schema(
    config: SpectralQRSConfig | None = None,
) -> SpectralQRSSchema:
    """Build the immutable H9 feature schema."""
    resolved = config or SpectralQRSConfig()
    payload = {
        "version": "e06r-h9-v1",
        "mode": "causal",
        "features": [feature.model_dump(mode="json") for feature in _SPECTRAL_DEFINITIONS],
        "extraction_config": resolved.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return SpectralQRSSchema(
        version="e06r-h9-v1",
        mode="causal",
        features=_SPECTRAL_DEFINITIONS,
        extraction_config=resolved,
        schema_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _signal_matrix(signals: np.ndarray) -> np.ndarray:
    values = np.asarray(signals, dtype=np.float32)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"signals must be 2-D or single-channel 3-D; got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("spectral-QRS signals contain NaN or Inf")
    return values


def _extract_qrs_windows(
    signals: np.ndarray,
    config: SpectralQRSConfig,
) -> np.ndarray:
    values = _signal_matrix(signals)
    center = values.shape[1] // 2
    pre = _milliseconds_to_samples(config.qrs_pre_ms, config.fs_hz)
    post = _milliseconds_to_samples(config.qrs_post_ms, config.fs_hz)
    start = center - pre
    end = center + post
    if start < 0 or end > values.shape[1]:
        raise ValueError("spectral window would require prohibited padding")
    windows = values[:, start:end].astype(np.float64, copy=True)
    baseline_count = min(20, windows.shape[1])
    windows -= np.median(windows[:, :baseline_count], axis=1, keepdims=True)
    return windows


def _band_power(
    psd: np.ndarray,
    freqs: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return np.zeros(psd.shape[0], dtype=np.float64)
    return _trapezoid(psd[:, mask], freqs[mask], axis=1)


def extract_spectral_qrs_features(
    signals: np.ndarray,
    config: SpectralQRSConfig | None = None,
) -> np.ndarray:
    """Compute spectral features from the QRS window."""
    resolved = config or SpectralQRSConfig()
    windows = _extract_qrs_windows(signals, resolved)
    n_samples = windows.shape[1]
    nperseg = min(resolved.nperseg, n_samples)

    freqs, psd = welch(
        windows,
        fs=resolved.fs_hz,
        nperseg=nperseg,
        axis=1,
    )
    psd = psd.astype(np.float64, copy=True)
    psd_sum = np.sum(psd, axis=1, keepdims=True)
    psd_normalized = psd / np.maximum(psd_sum, resolved.epsilon)

    centroid = np.sum(psd_normalized * freqs[np.newaxis, :], axis=1)
    bandwidth = np.sqrt(
        np.sum(
            psd_normalized * np.square(freqs[np.newaxis, :] - centroid[:, np.newaxis]),
            axis=1,
        )
    )
    flatness = np.exp(
        np.mean(np.log(np.maximum(psd_normalized, resolved.epsilon)), axis=1)
    ) / np.maximum(
        np.mean(psd_normalized, axis=1),
        resolved.epsilon,
    )
    cumulative = np.cumsum(psd, axis=1)
    total = cumulative[:, -1]
    rolloff_target = 0.85 * total
    rolloff_idx = np.argmax(cumulative >= rolloff_target[:, np.newaxis], axis=1)
    rolloff = freqs[rolloff_idx]
    peak_idx = np.argmax(psd, axis=1)
    peak_freq = freqs[peak_idx]
    peak_power = 10.0 * np.log10(
        np.maximum(
            psd_normalized[np.arange(windows.shape[0]), peak_idx],
            resolved.epsilon,
        )
    )

    power_0_10 = _band_power(psd, freqs, 0.0, 10.0)
    power_10_20 = _band_power(psd, freqs, 10.0, 20.0)
    power_20_40 = _band_power(psd, freqs, 20.0, 40.0)
    power_40_60 = _band_power(psd, freqs, 40.0, 60.0)
    power_60_100 = _band_power(psd, freqs, 60.0, 100.0)
    total_power = np.maximum(psd_sum[:, 0], resolved.epsilon)
    ratio_low_high = (power_0_10 + power_10_20) / np.maximum(
        power_40_60 + power_60_100,
        resolved.epsilon,
    )
    entropy = -np.sum(
        psd_normalized * np.log(np.maximum(psd_normalized, resolved.epsilon)),
        axis=1,
    )

    output = np.column_stack(
        [
            centroid,
            bandwidth,
            flatness,
            rolloff,
            peak_freq,
            peak_power,
            power_0_10 / total_power,
            power_10_20 / total_power,
            power_20_40 / total_power,
            power_40_60 / total_power,
            power_60_100 / total_power,
            ratio_low_high,
            entropy,
        ]
    )
    if not np.isfinite(output).all():
        raise RuntimeError("spectral QRS features contain NaN or Inf")
    return output.astype(np.float32, copy=False)
