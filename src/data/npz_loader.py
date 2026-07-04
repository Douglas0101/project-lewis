from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_feature_npz_and_parquet(
    feature_npz: Path,
    feature_parquet: Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load raw X, y, and metadata DataFrame from NPZ + Parquet."""
    data = np.load(feature_npz, mmap_mode="r")
    x_raw = data["X"]
    if str(x_raw.dtype) != "float32":
        x_raw = x_raw.astype(np.float32)
    y = data["y"].astype(np.int64)
    if x_raw.ndim == 2:
        x_raw = x_raw[..., np.newaxis]

    df = pd.read_parquet(feature_parquet)
    if len(x_raw) != len(df) or len(y) != len(df):
        raise ValueError(f"Mismatch: X={len(x_raw)}, y={len(y)}, df={len(df)}")

    return x_raw, y, df
