"""E06 H4: sampled QRS morphology contracts."""

from __future__ import annotations

import numpy as np

from src.features.e06_sampled import (
    SampledQRSConfig,
    build_sampled_qrs_schema,
    extract_sampled_qrs_morphology,
)


def _beats() -> np.ndarray:
    x = np.linspace(-1.0, 1.0, 500, dtype=np.float32)
    narrow = np.exp(-((x / 0.035) ** 2)).astype(np.float32)
    wide = (
        0.8 * np.exp(-(((x + 0.025) / 0.080) ** 2)) - 0.7 * np.exp(-(((x - 0.070) / 0.070) ** 2))
    ).astype(np.float32)
    return np.stack([narrow, wide, 0.5 * (narrow + wide)])


def test_sampled_qrs_schema_is_causal_and_deterministic() -> None:
    config = SampledQRSConfig(sample_step_ms=20.0)
    first = build_sampled_qrs_schema(config)
    second = build_sampled_qrs_schema(config)

    assert first.version == "e06r-h4-v1"
    assert first.mode == "causal"
    assert first.schema_sha256 == second.schema_sha256
    assert all(not feature.requires_future_context for feature in first.features)


def test_sampled_qrs_features_preserve_shape_and_are_finite() -> None:
    signals = _beats()
    config = SampledQRSConfig(sample_step_ms=20.0)

    features = extract_sampled_qrs_morphology(signals, config)
    schema = build_sampled_qrs_schema(config)

    expected_shape = (3, len(schema.features))
    assert features.shape == expected_shape
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert not np.array_equal(features[0], features[1])
    assert not np.array_equal(features[0], features[2])


def test_sampled_qrs_is_amplitude_robust_in_normalized_block() -> None:
    signals = _beats()[:1]
    config = SampledQRSConfig(sample_step_ms=20.0)
    schema = build_sampled_qrs_schema(config)
    names = [feature.name for feature in schema.features]
    normalized = [index for index, name in enumerate(names) if name.startswith("qrs_norm_")]

    original = extract_sampled_qrs_morphology(signals, config)
    scaled = extract_sampled_qrs_morphology(signals * 3.0, config)

    assert np.allclose(original[:, normalized], scaled[:, normalized], atol=1.0e-6)


def test_sampled_qrs_uses_segment_center_contract() -> None:
    signals = _beats()
    shifted = np.roll(signals, 40, axis=1)

    original = extract_sampled_qrs_morphology(signals)
    shifted_features = extract_sampled_qrs_morphology(shifted)

    assert not np.array_equal(original, shifted_features)
