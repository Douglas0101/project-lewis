import numpy as np
import pytest

from src.data.smote_sequence import SmoteSequence


def _build_3d(class_counts, timesteps=500, channels=1, seed=0):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for label, count in class_counts.items():
        X_list.append(rng.normal(size=(count, timesteps, channels)).astype(np.float32))
        y_list.append(np.full(count, label, dtype=np.int64))
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def test_smote_preserves_shape():
    class_counts = {0: 80, 1: 20}
    X, y = _build_3d(class_counts)

    sampler = SmoteSequence(ratio={1: 2.0}, random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)

    assert X_res.shape[1:] == (500, 1), "Shape (timesteps, channels) deve ser preservado"
    assert y_res.shape[0] == X_res.shape[0]
    assert np.sum(y_res == 1) == 40, "Classe minoritária deve ser dobrada"
    assert np.sum(y_res == 0) == 80, "Classe majoritária não deve mudar"


def test_smote_no_samples_keeps_shape():
    class_counts = {0: 80, 1: 40}
    X, y = _build_3d(class_counts)

    sampler = SmoteSequence(ratio={1: 1.0}, random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)

    assert X_res.shape[1:] == (500, 1)
    assert np.sum(y_res == 1) == 40
    assert np.sum(y_res == 0) == 80


def test_smote_invalid_shape_raises():
    X_2d = np.zeros((100, 500))
    y = np.zeros(100, dtype=np.int64)

    sampler = SmoteSequence(ratio={1: 2.0})
    with pytest.raises(ValueError):
        sampler.fit_resample(X_2d, y)
