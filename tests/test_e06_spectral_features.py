"""E06 H9: spectral QRS features contracts."""

from __future__ import annotations

import numpy as np

from src.features.e06_spectral import (
    build_spectral_qrs_schema,
    extract_spectral_qrs_features,
)


def _beats() -> np.ndarray:
    x = np.linspace(-1.0, 1.0, 500, dtype=np.float32)
    narrow = np.exp(-((x / 0.035) ** 2)).astype(np.float32)
    wide = (
        0.8 * np.exp(-(((x + 0.025) / 0.080) ** 2)) - 0.7 * np.exp(-(((x - 0.070) / 0.070) ** 2))
    ).astype(np.float32)
    return np.stack([narrow, wide, 0.5 * (narrow + wide)])


def test_spectral_schema_is_causal_and_deterministic() -> None:
    first = build_spectral_qrs_schema()
    second = build_spectral_qrs_schema()

    assert first.version == "e06r-h9-v1"
    assert first.mode == "causal"
    assert first.schema_sha256 == second.schema_sha256
    assert all(not feature.requires_future_context for feature in first.features)


def test_spectral_features_are_finite_and_differentiate_shapes() -> None:
    signals = _beats()

    features = extract_spectral_qrs_features(signals)
    schema = build_spectral_qrs_schema()

    expected_shape = (3, len(schema.features))
    assert features.shape == expected_shape
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert not np.array_equal(features[0], features[1])
    assert not np.array_equal(features[0], features[2])


def test_spectral_features_are_amplitude_robust() -> None:
    signals = _beats()[:1]

    original = extract_spectral_qrs_features(signals)
    scaled = extract_spectral_qrs_features(signals * 3.0)

    assert np.allclose(original, scaled, atol=1.0e-6)


def test_spectral_features_use_segment_center_contract() -> None:
    signals = _beats()
    shifted = np.roll(signals, 40, axis=1)

    original = extract_spectral_qrs_features(signals)
    shifted_features = extract_spectral_qrs_features(shifted)

    assert not np.array_equal(original, shifted_features)
