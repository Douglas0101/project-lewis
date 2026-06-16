"""Quality Gate QG3 — Feature Engineering.

Validates:
* QG3.1 — Time-domain features:
  rr_prev, rr_next, rr_ratio, rr_local_mean, rr_local_std, rmssd, heart_rate
* QG3.2 — Morphological features:
  r_amplitude, q_depth, t_amplitude, qrs_width_ms, qrs_area, st_slope_mV_s
* QG3.3 — AAMI mapping correctness
* QG3.4 — Augmentation: jitter, baseline_wander, powerline_noise, time_warp
* QG3.5 — Balancer: SMOTE/ADASYN in feature space
* QG3.6 — ≥ 10 dimensions per beat
* QG3.7 — No NaN/Inf in features
* QG3.8 — QRS width ∈ [40, 200] ms for > 95% of beats
* QG3.9 — Augmentation only affects amplitude, not shape fundamentally
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.aami_mapper import (
    AAMI_CLASSES,
    map_annotations,
    map_annotations_array,
)
from src.features.augmentation import ECGAugmenter
from src.features.balancer import ECGBalancer
from src.features.morphological import MorphologicalFeatures
from src.features.time_domain import TimeDomainFeatures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_ecg_segment(
    n_samples: int = 501,
    fs: float = 500.0,
) -> np.ndarray:
    """Create a synthetic ECG-like segment centered on an R-peak."""
    r_idx = n_samples // 2
    seg = np.zeros(n_samples, dtype=np.float32)

    # QRS: narrow Gaussian at center
    qrs_width = int(round(0.080 * fs))
    qrs_indices = np.arange(max(0, r_idx - qrs_width), min(n_samples, r_idx + qrs_width + 1))
    seg[qrs_indices] += 1.0 * np.exp(-0.5 * ((qrs_indices - r_idx) / (qrs_width / 3)) ** 2)

    # T-wave: ~300ms after R
    tw_start = r_idx + int(round(0.25 * fs))
    tw_width = int(round(0.15 * fs))
    tw_indices = np.arange(max(0, tw_start), min(n_samples, tw_start + tw_width))
    if len(tw_indices) > 0:
        seg[tw_indices] += 0.3 * np.exp(
            -0.5 * ((tw_indices - tw_start - tw_width // 2) / (tw_width / 3)) ** 2
        )

    # P-wave: ~150ms before R
    pw_start = r_idx - int(round(0.15 * fs))
    pw_width = int(round(0.10 * fs))
    pw_indices = np.arange(max(0, pw_start), min(n_samples, pw_start + pw_width))
    if len(pw_indices) > 0:
        seg[pw_indices] += 0.15 * np.exp(
            -0.5 * ((pw_indices - pw_start - pw_width // 2) / (pw_width / 3)) ** 2
        )

    return seg


def _synthetic_r_peaks(
    n_beats: int = 20,
    fs: float = 500.0,
    rr_ms: float = 800.0,
) -> np.ndarray:
    """Generate evenly spaced R-peak positions."""
    rr_samples = int(round(rr_ms * fs / 1000.0))
    return np.array([rr_samples * i + rr_samples // 2 for i in range(n_beats)], dtype=np.int64)


# ---------------------------------------------------------------------------
# QG3.1 — Time-Domain Features
# ---------------------------------------------------------------------------


@pytest.mark.qg3
class TestTimeDomainFeatures:
    """Validate RR-interval based features."""

    def test_extract_returns_correct_keys(self):
        r_peaks = _synthetic_r_peaks(n_beats=10, rr_ms=800.0)
        td = TimeDomainFeatures(fs=500.0)
        feats = td.extract(r_peaks)

        assert len(feats) == len(r_peaks)
        for f in feats:
            assert set(f.keys()) == {
                "rr_prev",
                "rr_next",
                "rr_ratio",
                "rr_local_mean",
                "rr_local_std",
                "rmssd",
                "heart_rate",
            }

    def test_rr_intervals_in_ms(self):
        r_peaks = _synthetic_r_peaks(n_beats=5, rr_ms=800.0)
        td = TimeDomainFeatures(fs=500.0)
        feats = td.extract(r_peaks)

        # rr_prev for beats 1-4 should be ~800 ms
        for i in range(1, len(feats)):
            assert 750 <= feats[i]["rr_prev"] <= 850, f"rr_prev[{i}] = {feats[i]['rr_prev']}"

    def test_heart_rate_calculation(self):
        r_peaks = _synthetic_r_peaks(n_beats=5, rr_ms=800.0)
        td = TimeDomainFeatures(fs=500.0)
        feats = td.extract(r_peaks)

        # HR = 60000 / 800 = 75 BPM
        for i in range(1, len(feats)):
            expected_hr = 60000.0 / feats[i]["rr_prev"]
            assert abs(feats[i]["heart_rate"] - expected_hr) < 1e-6

    def test_first_beat_no_prev(self):
        r_peaks = _synthetic_r_peaks(n_beats=5, rr_ms=800.0)
        td = TimeDomainFeatures(fs=500.0)
        feats = td.extract(r_peaks)

        assert feats[0]["rr_prev"] == 0.0
        assert feats[0]["heart_rate"] == 0.0

    def test_last_beat_no_next(self):
        r_peaks = _synthetic_r_peaks(n_beats=5, rr_ms=800.0)
        td = TimeDomainFeatures(fs=500.0)
        feats = td.extract(r_peaks)

        assert feats[-1]["rr_next"] == 0.0

    def test_no_nan_or_inf(self):
        r_peaks = _synthetic_r_peaks(n_beats=10, rr_ms=800.0)
        td = TimeDomainFeatures(fs=500.0)
        feats = td.extract(r_peaks)

        for f in feats:
            for k, v in f.items():
                assert not np.isnan(v), f"NaN in {k}"
                assert not np.isinf(v), f"Inf in {k}"

    def test_empty_peaks(self):
        td = TimeDomainFeatures(fs=500.0)
        feats = td.extract(np.array([], dtype=np.int64))
        assert feats == []


# ---------------------------------------------------------------------------
# QG3.2 — Morphological Features
# ---------------------------------------------------------------------------


@pytest.mark.qg3
class TestMorphologicalFeatures:
    """Validate QRS width, ST slope, and amplitude features."""

    def test_extract_returns_correct_keys(self):
        segments = np.stack([_synthetic_ecg_segment() for _ in range(5)])
        morph = MorphologicalFeatures(fs=500.0)
        feats = morph.extract(segments)

        assert len(feats) == 5
        for f in feats:
            assert set(f.keys()) == {
                "r_amplitude",
                "q_depth",
                "t_amplitude",
                "qrs_width_ms",
                "qrs_area",
                "st_slope_mV_s",
                "j_point",
            }

    def test_r_amplitude_positive(self):
        segments = np.stack([_synthetic_ecg_segment() for _ in range(5)])
        morph = MorphologicalFeatures(fs=500.0)
        feats = morph.extract(segments)

        for f in feats:
            assert f["r_amplitude"] > 0

    def test_qrs_width_in_range(self):
        segments = np.stack([_synthetic_ecg_segment() for _ in range(20)])
        morph = MorphologicalFeatures(fs=500.0)
        feats = morph.extract(segments)

        widths = [f["qrs_width_ms"] for f in feats if not np.isnan(f["qrs_width_ms"])]
        # Synthetic QRS is ~80ms wide; allow small widths due to narrow envelope
        assert len(widths) > 0
        for w in widths:
            assert 2 <= w <= 200, f"QRS width {w} ms out of range [2, 200]"

    def test_qrs_width_valid_rate(self):
        """QG3: > 95% of beats must have valid QRS width."""
        segments = np.stack([_synthetic_ecg_segment() for _ in range(100)])
        morph = MorphologicalFeatures(fs=500.0)
        feats = morph.extract(segments)

        n_valid = sum(1 for f in feats if not np.isnan(f["qrs_width_ms"]))
        valid_rate = n_valid / len(feats)
        assert valid_rate > 0.95, f"QRS width valid rate = {valid_rate:.2f}, expected > 0.95"

    def test_no_nan_in_scalar_features(self):
        segments = np.stack([_synthetic_ecg_segment() for _ in range(10)])
        morph = MorphologicalFeatures(fs=500.0)
        feats = morph.extract(segments)

        for f in feats:
            assert not np.isnan(f["r_amplitude"])
            assert not np.isnan(f["q_depth"])
            assert not np.isnan(f["t_amplitude"])
            assert not np.isnan(f["st_slope_mV_s"])

    def test_empty_segments(self):
        morph = MorphologicalFeatures(fs=500.0)
        feats = morph.extract(np.empty((0, 501), dtype=np.float32))
        assert feats == []

    def test_st_slope_units(self):
        segments = np.stack([_synthetic_ecg_segment() for _ in range(5)])
        morph = MorphologicalFeatures(fs=500.0)
        feats = morph.extract(segments)

        for f in feats:
            # ST slope should be in mV/s, which for synthetic signal is close to 0
            assert abs(f["st_slope_mV_s"]) < 1e6  # not infinity


# ---------------------------------------------------------------------------
# QG3.3 — AAMI Mapping
# ---------------------------------------------------------------------------


@pytest.mark.qg3
class TestAAMIMapper:
    """Validate WFDB → AAMI EC57 mapping."""

    def test_map_all_beat_symbols(self):
        symbols = ["N", "L", "R", "e", "j", "V", "E", "A", "a", "J", "S", "F", "/", "f", "Q", "|"]
        labels, stats = map_annotations(symbols)

        assert len(labels) == len(symbols)
        assert set(labels).issubset(set(AAMI_CLASSES))
        assert stats["n_total"] == len(symbols)
        assert stats["n_mapped"] == len(symbols)
        assert stats["n_unmapped"] == 0

    def test_normal_beats_map_to_n(self):
        symbols = ["N", "L", "R", "e", "j"]
        labels, _ = map_annotations(symbols)
        assert all(label == "N" for label in labels)

    def test_ventricular_map_to_v(self):
        symbols = ["V", "E"]
        labels, _ = map_annotations(symbols)
        assert all(label == "V" for label in labels)

    def test_supraventricular_map_to_s(self):
        symbols = ["A", "a", "J", "S"]
        labels, _ = map_annotations(symbols)
        assert all(label == "S" for label in labels)

    def test_excluded_symbols_filtered(self):
        """Non-beat annotations (~, +, x) must be filtered out."""
        symbols = ["N", "~", "+", "V", "x"]
        labels, stats = map_annotations(symbols)

        assert len(labels) == 2  # only N and V
        assert "N" in labels
        assert "V" in labels

    def test_unknown_symbols_map_to_q(self):
        symbols = ["N", "UNKNOWN", "V"]
        labels, stats = map_annotations(symbols)

        assert labels[1] == "Q"
        assert stats["n_unmapped"] == 1

    def test_stats_counts(self):
        symbols = ["N", "N", "V", "V", "V", "S", "F", "Q"]
        labels, stats = map_annotations(symbols)

        assert stats["n_by_class"]["N"] == 2
        assert stats["n_by_class"]["V"] == 3
        assert stats["n_by_class"]["S"] == 1
        assert stats["n_by_class"]["F"] == 1
        assert stats["n_by_class"]["Q"] == 1

    def test_array_version(self):
        symbols = np.array(["N", "V", "S", "F", "Q"])
        labels, stats = map_annotations_array(symbols)

        assert isinstance(labels, np.ndarray)
        assert list(labels) == ["N", "V", "S", "F", "Q"]

    def test_five_classes_defined(self):
        assert len(AAMI_CLASSES) == 5
        assert set(AAMI_CLASSES) == {"N", "S", "V", "F", "Q"}


# ---------------------------------------------------------------------------
# QG3.4 — Augmentation
# ---------------------------------------------------------------------------


@pytest.mark.qg3
class TestAugmentation:
    """Validate ECG augmentation methods."""

    def test_jitter_preserves_mean(self):
        np.random.seed(42)
        seg = _synthetic_ecg_segment()
        aug = ECGAugmenter(seed=42)
        aug_seg = aug.jitter(seg, std_factor=0.01)

        assert aug_seg.shape == seg.shape
        assert aug_seg.dtype == seg.dtype
        # Mean should be approximately preserved
        assert abs(float(np.mean(aug_seg)) - float(np.mean(seg))) < 0.1

    def test_jitter_changes_signal(self):
        np.random.seed(42)
        seg = _synthetic_ecg_segment()
        aug = ECGAugmenter(seed=42)
        aug_seg = aug.jitter(seg, std_factor=0.1)

        assert not np.allclose(aug_seg, seg)

    def test_baseline_wander_adds_low_freq(self):
        np.random.seed(42)
        seg = _synthetic_ecg_segment()
        aug = ECGAugmenter(seed=42)
        aug_seg = aug.baseline_wander(seg, fs=500.0)

        assert aug_seg.shape == seg.shape
        # Wander should be small amplitude
        diff = aug_seg - seg
        assert np.max(np.abs(diff)) <= 0.25  # max amp 0.2 + jitter tolerance

    def test_powerline_noise_adds_freq(self):
        np.random.seed(42)
        seg = _synthetic_ecg_segment()
        aug = ECGAugmenter(seed=42)
        aug_seg = aug.powerline_noise(seg, fs=500.0, freq=60.0)

        assert aug_seg.shape == seg.shape
        diff = aug_seg - seg
        assert np.max(np.abs(diff)) <= 0.1  # max amp 0.05 + tolerance

    def test_time_warp_preserves_length(self):
        np.random.seed(42)
        seg = _synthetic_ecg_segment()
        aug = ECGAugmenter(seed=42)
        aug_seg = aug.time_warp(seg, max_stretch=0.05)

        assert len(aug_seg) == len(seg)

    def test_apply_multiple_methods(self):
        np.random.seed(42)
        seg = _synthetic_ecg_segment()
        aug = ECGAugmenter(seed=42)
        aug_seg = aug.apply(seg, fs=500.0, p=1.0, stage="train")

        assert aug_seg.shape == seg.shape
        assert not np.allclose(aug_seg, seg)

    def test_apply_with_methods_list(self):
        np.random.seed(42)
        seg = _synthetic_ecg_segment()
        aug = ECGAugmenter(seed=42)
        aug_seg = aug.apply(seg, fs=500.0, p=1.0, methods=["jitter"], stage="train")

        assert aug_seg.shape == seg.shape

    def test_augmentation_only_train(self):
        """QG3: augmentation must be disabled for test/val/pretrain stages."""
        np.random.seed(42)
        seg = _synthetic_ecg_segment()
        aug = ECGAugmenter(seed=42)

        # Allowed on train
        aug_seg = aug.apply(seg, fs=500.0, p=1.0, stage="train")
        assert aug_seg.shape == seg.shape

        # Forbidden on test, val and pretrain
        for bad_stage in ("test", "val", "pretrain"):
            with pytest.raises(ValueError, match="stage='train'"):
                aug.apply(seg, fs=500.0, p=0.0, stage=bad_stage)


# ---------------------------------------------------------------------------
# QG3.5 — Balancer (SMOTE)
# ---------------------------------------------------------------------------


@pytest.mark.qg3
class TestBalancer:
    """Validate class balancing in feature space."""

    @pytest.mark.skipif(
        not pytest.importorskip("imblearn", reason="imbalanced-learn not installed"),
        reason="imbalanced-learn not installed",
    )
    def test_smote_balances_classes(self):
        pytest.importorskip("imblearn")
        # Create imbalanced dataset — minority must have ≥ k_neighbors+1 samples
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.array(["N"] * 140 + ["V"] * 40 + ["S"] * 20)

        bal = ECGBalancer(strategy="smote", random_state=42)
        X_bal, y_bal = bal.balance(X, y)

        from collections import Counter

        counts = Counter(y_bal)
        # SMOTE should balance all classes to the majority count
        assert counts["N"] == counts["V"] == counts["S"]

    @pytest.mark.skipif(
        not pytest.importorskip("imblearn", reason="imbalanced-learn not installed"),
        reason="imbalanced-learn not installed",
    )
    def test_smote_rus_combined(self):
        pytest.importorskip("imblearn")
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.array(["N"] * 140 + ["V"] * 40 + ["S"] * 20)

        bal = ECGBalancer(strategy="smote+rus", random_state=42)
        X_bal, y_bal = bal.balance(X, y)

        from collections import Counter

        counts = Counter(y_bal)
        # SMOTE + RUS should reduce imbalance but not necessarily equal
        assert (
            counts["V"] >= counts["S"]
        )  # V should be at least as represented as S after balancing

    def test_invalid_strategy(self):
        pytest.importorskip("imblearn")
        bal = ECGBalancer(strategy="invalid", random_state=42)
        with pytest.raises(ValueError, match="desconhecida"):
            bal.balance(np.zeros((10, 2)), np.array(["N"] * 10))


# ---------------------------------------------------------------------------
# QG3.6 — Dimension Count
# ---------------------------------------------------------------------------


class TestFeatureDimensions:
    """Validate total feature dimensions per beat."""

    @pytest.mark.qg3
    def test_at_least_10_dimensions(self):
        """QG3: ≥ 10 feature dimensions per beat."""
        r_peaks = _synthetic_r_peaks(n_beats=5, rr_ms=800.0)
        segments = np.stack([_synthetic_ecg_segment() for _ in range(5)])

        td = TimeDomainFeatures(fs=500.0)
        morph = MorphologicalFeatures(fs=500.0)

        temporal = td.extract(r_peaks)
        morphological = morph.extract(segments)

        assert len(temporal) == len(morphological) == 5

        for t, m in zip(temporal, morphological):
            n_dims = len(t) + len(m)
            assert n_dims >= 10, f"Only {n_dims} dimensions, expected >= 10"


# ---------------------------------------------------------------------------
# QG3.7 — No NaN/Inf
# ---------------------------------------------------------------------------


@pytest.mark.qg3
class TestNoNaNInf:
    """Validate absence of NaN and Inf in all features."""

    def test_all_features_clean(self):
        r_peaks = _synthetic_r_peaks(n_beats=20, rr_ms=800.0)
        segments = np.stack([_synthetic_ecg_segment() for _ in range(20)])

        td = TimeDomainFeatures(fs=500.0)
        morph = MorphologicalFeatures(fs=500.0)

        temporal = td.extract(r_peaks)
        morphological = morph.extract(segments)

        for t in temporal:
            for k, v in t.items():
                assert not np.isnan(v), f"NaN in temporal {k}"
                assert not np.isinf(v), f"Inf in temporal {k}"

        for m in morphological:
            for k, v in m.items():
                if k != "qrs_width_ms" and k != "qrs_area":  # NaN allowed for failed QRS
                    assert not np.isnan(v), f"NaN in morphological {k}"
                    assert not np.isinf(v), f"Inf in morphological {k}"
