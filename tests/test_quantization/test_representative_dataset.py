"""Testes para representative dataset estratificado."""

from __future__ import annotations

import numpy as np
import pytest

from src.quantization.representative_dataset import StratifiedRepresentativeDataset


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.random((200, 500, 1)).astype(np.float32)
    y = np.array([0] * 120 + [1] * 80)
    return X, y


def test_generator_yields_expected_batches(sample_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = sample_data
    dataset = StratifiedRepresentativeDataset(
        X, y, min_samples_per_class=50, batch_size=1, random_state=42
    )
    gen = dataset.generator()
    batches = list(gen())

    assert len(batches) == 100
    for batch in batches:
        assert len(batch) == 1
        assert batch[0].dtype == np.float32
        assert batch[0].shape == (1, 500, 1)


def test_generator_respects_min_samples_per_class(
    sample_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = sample_data
    dataset = StratifiedRepresentativeDataset(
        X, y, min_samples_per_class=500, batch_size=1, random_state=42
    )
    gen = dataset.generator()
    batches = list(gen())

    assert len(batches) == len(y)


def test_generator_deterministic(sample_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = sample_data
    dataset_a = StratifiedRepresentativeDataset(
        X, y, min_samples_per_class=50, batch_size=1, random_state=123
    )
    dataset_b = StratifiedRepresentativeDataset(
        X, y, min_samples_per_class=50, batch_size=1, random_state=123
    )

    batches_a = [batch[0] for batch in dataset_a.generator()()]
    batches_b = [batch[0] for batch in dataset_b.generator()()]

    assert len(batches_a) == len(batches_b)
    for a, b in zip(batches_a, batches_b):
        np.testing.assert_array_equal(a, b)
