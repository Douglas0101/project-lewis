"""Representative dataset estratificado para calibração PTQ/QAT."""

from __future__ import annotations

from typing import Callable, Iterator, List

import numpy as np


class StratifiedRepresentativeDataset:
    """Gera batches representativos estratificados por classe.

    Garante pelo menos ``min_samples_per_class`` amostras por classe quando
    disponíveis; caso contrário, utiliza todas as amostras da classe sem
    reposição. Os índices são embaralhados deterministicamente para formar o
    generator consumido pelo ``tf.lite.TFLiteConverter``.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        min_samples_per_class: int = 500,
        batch_size: int = 1,
        random_state: int = 42,
    ) -> None:
        self.X = X
        self.y = y
        self.min_samples_per_class = min_samples_per_class
        self.batch_size = batch_size
        self.random_state = random_state

    def generator(self) -> Callable[[], Iterator[List[np.ndarray]]]:
        """Retorna generator que yields [x] com shape (batch_size, *feature_shape)."""
        rng = np.random.default_rng(self.random_state)
        indices: List[int] = []
        for cls in np.unique(self.y):
            cls_idx = np.where(self.y == cls)[0]
            n = min(self.min_samples_per_class, len(cls_idx))
            chosen = rng.choice(cls_idx, size=n, replace=False)
            indices.extend(chosen.tolist())
        rng.shuffle(indices)

        def gen() -> Iterator[List[np.ndarray]]:
            for i in indices:
                x = self.X[i : i + self.batch_size]
                yield [x.astype(np.float32)]

        return gen
