from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def fit_feature_scaler_on_train(
    X: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[StandardScaler, np.ndarray, np.ndarray]:
    """Fit StandardScaler on the first training fold and return train/val indices."""
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, groups=groups))
    train_idx, val_idx = splits[0]

    scaler = StandardScaler()
    scaler.fit(X[train_idx])

    return scaler, train_idx, val_idx


def scale_features(
    X: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    return scaler.transform(X).astype(np.float32, copy=False)
