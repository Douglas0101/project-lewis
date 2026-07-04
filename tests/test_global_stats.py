import numpy as np
import pytest
from src.data.global_stats import GlobalStatsHelper


def test_compute_from_training_only():
    np.random.seed(42)
    X_train = np.random.randn(100, 500, 1).astype(np.float32) * 2.0 + 1.0
    X_val = np.random.randn(20, 500, 1).astype(np.float32) * 2.0 + 1.0
    helper = GlobalStatsHelper()
    mean, std = helper.fit(X_train)
    assert isinstance(mean, (float, np.floating))
    assert isinstance(std, (float, np.floating))
    assert std > 0

    X_train_norm = helper.transform(X_train)
    assert abs(float(X_train_norm.mean())) < 0.05
    assert abs(float(X_train_norm.std()) - 1.0) < 0.05

    X_val_norm = helper.transform(X_val)
    assert not np.any(np.isnan(X_val_norm))
    assert not np.any(np.isinf(X_val_norm))


def test_clipping_reduces_std():
    np.random.seed(42)
    X = np.random.randn(100, 500, 1).astype(np.float32)
    X[0, 0, 0] = 20.0
    helper = GlobalStatsHelper(clip_limits=(-5.0, 5.0))
    _, std_clipped = helper.fit(X)
    std_unclipped = float(np.std(X))
    assert std_clipped < std_unclipped


def test_save_load(tmp_path):
    np.random.seed(42)
    helper = GlobalStatsHelper()
    helper.fit(np.random.randn(50, 500, 1).astype(np.float32))
    path = tmp_path / "stats.json"
    helper.save(path)
    loaded = GlobalStatsHelper.load(path)
    assert float(helper.mean) == float(loaded.mean)
    assert float(helper.std) == float(loaded.std)


def test_transform_without_fit_raises():
    np.random.seed(42)
    helper = GlobalStatsHelper()
    with pytest.raises(ValueError, match="fit"):
        helper.transform(np.random.randn(10, 500, 1))


def test_transform_1d_input():
    np.random.seed(42)
    helper = GlobalStatsHelper()
    helper.fit(np.random.randn(100, 500, 1).astype(np.float32))
    x = np.random.randn(500).astype(np.float32)
    y = helper.transform(x)
    assert y.shape == (500,)


def test_transform_2d_input():
    np.random.seed(42)
    helper = GlobalStatsHelper()
    helper.fit(np.random.randn(100, 500, 1).astype(np.float32))
    X = np.random.randn(10, 500).astype(np.float32)
    Y = helper.transform(X)
    assert Y.shape == (10, 500)


def test_load_without_fit(tmp_path):
    np.random.seed(42)
    helper = GlobalStatsHelper()
    path = tmp_path / "stats.json"
    helper.save(path)
    loaded = GlobalStatsHelper.load(path)
    assert loaded.mean is None
    assert loaded.std is None


def test_chunk_size_roundtrip(tmp_path):
    np.random.seed(42)
    helper = GlobalStatsHelper(chunk_size=1024)
    helper.fit(np.random.randn(50, 500, 1).astype(np.float32))
    path = tmp_path / "stats.json"
    helper.save(path)
    loaded = GlobalStatsHelper.load(path)
    assert loaded.chunk_size == 1024
