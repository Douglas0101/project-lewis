"""Testes para scripts/prepare_two_stage_datasets.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.prepare_two_stage_datasets import (
    AAMI_CLASSES,
    _prepare_stage1,
    _prepare_stage2,
)


def _make_source(n: int = 20, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 500, 1)).astype(np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 2, dtype=np.int64)
    y = y[:n]
    df = pd.DataFrame(
        {
            "record_id": [f"rec_{i % 4}" for i in range(n)],
            "aami_label": [AAMI_CLASSES[int(v)] for v in y],
            "rr_prev": rng.uniform(0.6, 1.2, size=n),
            "qrs_width_ms": rng.uniform(60.0, 120.0, size=n),
        }
    )
    return X, y, df


def test_prepare_stage1_default_includes_q_as_abnormal(tmp_path):
    X, y, df = _make_source()
    npz_path = tmp_path / "stage1.npz"
    parquet_path = tmp_path / "stage1.parquet"

    _prepare_stage1(X, y, df, npz_path=npz_path, parquet_path=parquet_path)

    data = np.load(npz_path)
    assert data["y"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    assert len(data["X"]) == len(data["y"])


def test_prepare_stage1_exclude_q_drops_q_samples(tmp_path):
    X, y, df = _make_source()
    npz_path = tmp_path / "stage1_no_q.npz"
    parquet_path = tmp_path / "stage1_no_q.parquet"

    _prepare_stage1(
        X,
        y,
        df,
        npz_path=npz_path,
        parquet_path=parquet_path,
        exclude_q=True,
    )

    data = np.load(npz_path)
    df_out = pd.read_parquet(parquet_path)

    # Q samples (label 4) must be removed
    assert 4 not in df_out["aami_label"].values
    assert len(data["X"]) == len(df_out)
    # Labels must still be 0/1
    assert set(np.unique(data["y"])) == {0, 1}


def test_prepare_stage1_includes_morphological_features(tmp_path):
    X, y, df = _make_source()
    npz_path = tmp_path / "stage1.npz"
    parquet_path = tmp_path / "stage1.parquet"

    _prepare_stage1(X, y, df, npz_path=npz_path, parquet_path=parquet_path)

    data = np.load(npz_path)
    assert "features" in data
    assert data["features"].shape == (len(y), 2)
    # X deve conter apenas o sinal raw (1 canal); features ficam em array separado
    assert data["X"].shape == (len(y), 500, 1)
    assert list(data["feature_columns"]) == ["rr_prev", "qrs_width_ms"]


def test_prepare_stage2_excludes_n_and_q(tmp_path):
    X, y, df = _make_source()
    npz_path = tmp_path / "stage2.npz"
    parquet_path = tmp_path / "stage2.parquet"

    _prepare_stage2(X, y, df, npz_path=npz_path, parquet_path=parquet_path)

    data = np.load(npz_path)
    df_out = pd.read_parquet(parquet_path)

    assert set(np.unique(data["y"])) == {0, 1, 2}
    assert not set(df_out["aami_label"].unique()).intersection({"N", "Q"})
    assert "stage" in df_out.columns
