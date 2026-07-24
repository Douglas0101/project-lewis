"""E06 reopened: direct morphology feature-family contracts."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.e06_fusion import (
    DIRECT_MORPHOLOGY_FEATURE_NAMES,
    FusionMorphologyConfig,
    build_direct_morphology_schema,
    extract_direct_morphology,
)


def _synthetic_beats() -> tuple[np.ndarray, np.ndarray]:
    signals = np.zeros((3, 500), dtype=np.float32)
    r_peaks = np.array([250, 250, 250], dtype=np.int64)

    signals[0, 250] = 2.0
    signals[0, 265] = -1.0

    signals[1, 245:256] = np.linspace(-1.0, 2.0, 11, dtype=np.float32)
    signals[1, 256:276] = np.linspace(2.0, -1.2, 20, dtype=np.float32)

    signals[2] = 0.5 * (signals[0] + signals[1])
    return signals, r_peaks


def test_direct_morphology_schema_is_causal_and_versioned() -> None:
    schema = build_direct_morphology_schema(FusionMorphologyConfig())

    assert schema.version == "e06r-h1-v1"
    assert schema.mode == "causal"
    assert [feature.name for feature in schema.features] == list(DIRECT_MORPHOLOGY_FEATURE_NAMES)
    assert all(not feature.requires_previous_context for feature in schema.features)
    assert all(not feature.requires_future_context for feature in schema.features)
    assert len(schema.schema_sha256) == 64


def test_direct_morphology_is_finite_deterministic_and_float32() -> None:
    signals, r_peaks = _synthetic_beats()

    first = extract_direct_morphology(signals, r_peaks)
    second = extract_direct_morphology(signals[..., np.newaxis], r_peaks)

    expected_shape = (3, len(DIRECT_MORPHOLOGY_FEATURE_NAMES))
    assert first.shape == expected_shape
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_direct_morphology_peak_and_rs_formulas() -> None:
    signals, r_peaks = _synthetic_beats()
    features = extract_direct_morphology(signals[:1], r_peaks[:1])
    by_name = dict(zip(DIRECT_MORPHOLOGY_FEATURE_NAMES, features[0], strict=True))

    assert by_name["qrs_peak_to_peak"] == pytest.approx(3.0)
    assert by_name["qrs_s_amplitude"] == pytest.approx(-1.0)
    assert by_name["qrs_r_s_ratio"] == pytest.approx(2.0)
    assert by_name["qrs_abs_area"] > 0.0
    assert by_name["qrs_energy"] > 0.0


def test_direct_morphology_rejects_windows_that_require_padding() -> None:
    signals = np.zeros((1, 500), dtype=np.float32)

    with pytest.raises(ValueError, match="padding"):
        extract_direct_morphology(signals, np.array([10], dtype=np.int64))


def test_direct_morphology_rejects_invalid_input_contract() -> None:
    with pytest.raises(ValueError, match="same number"):
        extract_direct_morphology(
            np.zeros((2, 500), dtype=np.float32),
            np.array([250], dtype=np.int64),
        )

    with pytest.raises(ValueError, match="2-D or 3-D"):
        extract_direct_morphology(
            np.zeros((500,), dtype=np.float32),
            np.array([250], dtype=np.int64),
        )
