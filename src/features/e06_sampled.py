"""Sampled QRS shape representation for reopened E06 H4."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SampledQRSConfig(BaseModel):
    """Immutable sampled-morphology configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fs_hz: float = Field(default=500.0, gt=0.0)
    qrs_pre_ms: float = Field(default=120.0, gt=0.0)
    qrs_post_ms: float = Field(default=180.0, gt=0.0)
    sample_step_ms: float = Field(default=20.0, gt=0.0)
    baseline_ms: float = Field(default=40.0, gt=0.0)
    epsilon: float = Field(default=1.0e-8, gt=0.0)

    @model_validator(mode="after")
    def validate_grid(self) -> SampledQRSConfig:
        total = self.qrs_pre_ms + self.qrs_post_ms
        if total > 500.0:
            raise ValueError("sampled QRS window must not exceed 500 ms")
        if self.sample_step_ms > total:
            raise ValueError("sample_step_ms must fit inside the QRS window")
        if self.baseline_ms > total:
            raise ValueError("baseline_ms must fit inside the QRS window")
        return self


class SampledQRSFeatureDefinition(BaseModel):
    """One sampled-shape feature contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    family: Literal["sampled_qrs_morphology"] = "sampled_qrs_morphology"
    units: str
    requires_previous_context: bool = False
    requires_future_context: bool = False


class SampledQRSSchema(BaseModel):
    """Content-addressed H4 schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    mode: Literal["causal"]
    features: tuple[SampledQRSFeatureDefinition, ...]
    extraction_config: SampledQRSConfig
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _to_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} is not an integer") from error


def _milliseconds_to_samples(milliseconds: float, fs_hz: float) -> int:
    try:
        return int(round(milliseconds * fs_hz / 1000.0))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("invalid sampled-QRS time conversion") from error


def _grid(config: SampledQRSConfig) -> tuple[np.ndarray, np.ndarray]:
    pre = _milliseconds_to_samples(config.qrs_pre_ms, config.fs_hz)
    post = _milliseconds_to_samples(config.qrs_post_ms, config.fs_hz)
    step = _milliseconds_to_samples(config.sample_step_ms, config.fs_hz)
    if step <= 0:
        raise ValueError("sample_step_ms resolves to zero samples")
    sample_offsets = np.arange(-pre, post, step, dtype=np.int64)
    offset_ms = sample_offsets.astype(np.float64) / config.fs_hz * 1000.0
    return sample_offsets, offset_ms


def _offset_tag(milliseconds: float) -> str:
    rounded = _to_int(np.rint(milliseconds), "sample offset")
    prefix = "m" if rounded < 0 else "p"
    return f"{prefix}{abs(rounded):03d}"


def _definitions(config: SampledQRSConfig) -> tuple[SampledQRSFeatureDefinition, ...]:
    _, offsets_ms = _grid(config)
    definitions: list[SampledQRSFeatureDefinition] = []
    for prefix, units in (
        ("qrs_raw", "mV"),
        ("qrs_norm", "ratio"),
        ("qrs_diff_norm", "ratio"),
    ):
        definitions.extend(
            SampledQRSFeatureDefinition(
                name=f"{prefix}_{_offset_tag(offset)}ms",
                units=units,
            )
            for offset in offsets_ms
        )
    return tuple(definitions)


def build_sampled_qrs_schema(
    config: SampledQRSConfig | None = None,
) -> SampledQRSSchema:
    """Build the immutable H4 feature schema."""
    resolved = config or SampledQRSConfig()
    definitions = _definitions(resolved)
    payload = {
        "version": "e06r-h4-v1",
        "mode": "causal",
        "features": [feature.model_dump(mode="json") for feature in definitions],
        "extraction_config": resolved.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return SampledQRSSchema(
        version="e06r-h4-v1",
        mode="causal",
        features=definitions,
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
        raise ValueError("sampled-QRS signals contain NaN or Inf")
    return values


def extract_sampled_qrs_morphology(
    signals: np.ndarray,
    config: SampledQRSConfig | None = None,
) -> np.ndarray:
    """Sample raw, amplitude-normalized and derivative-normalized QRS shape."""
    resolved = config or SampledQRSConfig()
    values = _signal_matrix(signals)
    offsets, _ = _grid(resolved)
    center = values.shape[1] // 2
    indices = center + offsets
    if np.any(indices < 0) or np.any(indices >= values.shape[1]):
        raise ValueError("sampled QRS grid would require prohibited padding")

    pre = _milliseconds_to_samples(resolved.qrs_pre_ms, resolved.fs_hz)
    post = _milliseconds_to_samples(resolved.qrs_post_ms, resolved.fs_hz)
    qrs = values[:, center - pre : center + post].astype(np.float64, copy=True)
    baseline_count = min(
        _milliseconds_to_samples(resolved.baseline_ms, resolved.fs_hz),
        qrs.shape[1],
    )
    qrs -= np.median(qrs[:, :baseline_count], axis=1, keepdims=True)
    local_indices = offsets + pre
    raw_samples = qrs[:, local_indices]

    amplitude = np.max(np.abs(qrs), axis=1, keepdims=True)
    normalized = qrs / np.maximum(amplitude, resolved.epsilon)
    normalized_samples = normalized[:, local_indices]
    derivative = np.gradient(normalized, axis=1)
    derivative_scale = np.max(np.abs(derivative), axis=1, keepdims=True)
    derivative /= np.maximum(derivative_scale, resolved.epsilon)
    derivative_samples = derivative[:, local_indices]

    output = np.column_stack([raw_samples, normalized_samples, derivative_samples])
    if not np.isfinite(output).all():
        raise RuntimeError("sampled QRS features contain NaN or Inf")
    return output.astype(np.float32, copy=False)
