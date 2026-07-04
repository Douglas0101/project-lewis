from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.data.preprocessor import ECGPreprocessor


class GlobalStatsHelper:
    """Compute and apply global z-score statistics fitted only on training data."""

    def __init__(
        self,
        clip_limits: Tuple[float, float] = (-5.0, 5.0),
        chunk_size: int = 8192,
        eps: float = 1.0e-12,
    ):
        self.clip_limits = clip_limits
        self.chunk_size = chunk_size
        self.eps = eps
        self.mean: Optional[np.float32] = None
        self.std: Optional[np.float32] = None

    def fit(self, X: np.ndarray) -> Tuple[np.float32, np.float32]:
        """Compute scalar mean/std on training data after clipping."""
        mean, std = ECGPreprocessor.compute_global_stats(
            X,
            clip_limits=self.clip_limits,
            chunk_size=self.chunk_size,
            eps=self.eps,
        )
        self.mean = np.float32(mean)
        self.std = np.float32(std)
        return self.mean, self.std

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted z-score normalization."""
        if self.mean is None or self.std is None:
            raise ValueError("Call fit() before transform()")
        X = np.asarray(X)
        original_shape = X.shape
        if X.ndim == 1:
            X = X.reshape(1, -1, 1)
        elif X.ndim == 2:
            X = X[..., np.newaxis]
        mean = np.float32(self.mean)
        std = np.float32(self.std)
        return ((X.astype(np.float32, copy=False) - mean) / std).reshape(original_shape)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean": float(self.mean) if self.mean is not None else None,
            "std": float(self.std) if self.std is not None else None,
            "clip_limits": list(self.clip_limits),
            "eps": self.eps,
            "chunk_size": self.chunk_size,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GlobalStatsHelper":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        inst = cls(
            clip_limits=tuple(data["clip_limits"]),
            eps=data["eps"],
            chunk_size=data.get("chunk_size", 8192),
        )
        if data["mean"] is not None:
            inst.mean = np.float32(data["mean"])
            inst.std = np.float32(data["std"])
        return inst
