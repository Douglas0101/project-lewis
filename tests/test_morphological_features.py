"""Unit tests for robust morphological feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.morphological import MorphologicalFeatures


def _make_segment_with_qrs(fs: float = 500.0, width_ms: float = 80.0) -> tuple[np.ndarray, int]:
    """Create a synthetic beat segment with a QRS-like waveform."""
    window_len = 500
    seg = np.zeros(window_len, dtype=np.float32)
    r_idx = window_len // 2
    half_width_samples = int(round((width_ms / 1000.0) * fs / 2.0))
    onset = r_idx - half_width_samples
    offset = r_idx + half_width_samples
    if onset >= 0 and offset <= window_len:
        seg[onset:offset] = np.sin(np.linspace(0, np.pi, offset - onset)).astype(np.float32)
    seg[r_idx] = 1.0
    return seg, r_idx


def test_qrs_width_within_valid_range():
    morph = MorphologicalFeatures(fs=500.0)
    seg, r_idx = _make_segment_with_qrs(width_ms=80.0)
    feats = morph.extract(seg[np.newaxis, :], fs=500.0, r_idx=r_idx)
    assert len(feats) == 1
    assert not np.isnan(feats[0]["qrs_width_ms"])
    assert 20.0 <= feats[0]["qrs_width_ms"] <= 180.0
    assert not np.isnan(feats[0]["qrs_area"])
    assert feats[0]["qrs_area"] > 0.0


def test_qrs_width_invalid_returns_nan():
    morph = MorphologicalFeatures(fs=500.0)
    seg = np.zeros(500, dtype=np.float32)
    seg[250] = 1.0
    feats = morph.extract(seg[np.newaxis, :], fs=500.0, r_idx=250)
    assert np.isnan(feats[0]["qrs_width_ms"])
    assert np.isnan(feats[0]["qrs_area"])


def test_new_features_present():
    morph = MorphologicalFeatures(fs=500.0)
    seg, r_idx = _make_segment_with_qrs(width_ms=80.0)
    feats = morph.extract(seg[np.newaxis, :], fs=500.0, r_idx=r_idx)
    assert "qrs_asymmetry_index" in feats[0]
    assert "t_r_ratio" in feats[0]
    assert "qrs_raggedness" in feats[0]
    assert not np.isnan(feats[0]["qrs_raggedness"])


def test_physiological_width_validation():
    """Very narrow synthetic QRS should be marked NaN."""
    morph = MorphologicalFeatures(fs=500.0)
    seg, r_idx = _make_segment_with_qrs(width_ms=10.0)
    feats = morph.extract(seg[np.newaxis, :], fs=500.0, r_idx=r_idx)
    assert np.isnan(feats[0]["qrs_width_ms"])
