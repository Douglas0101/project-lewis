"""E07R regression tests for the full-template loader data contract.

The v3.1 r4/r5 generations canonically store beats as ``(n, 500, 1)``
(golden rule: input shape ``(500, 1)``). The PD full-template loader was
written against the legacy 2-D layout and blocked the authorized r4
source before any E06.5-PD cell could run. These tests pin the accepted
contract: 2-D ``(n, 500)`` or single-channel 3-D ``(n, 500, 1)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.stage2_research.config import load_research_config
from src.stage2_research.contracts import DatasetConfig, HashedPath, ResearchError
from src.stage2_research.data import load_full_template_dataset
from src.stage2_research.integrity import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_LABELS = ("N", "S", "V", "F", "Q")


def _write_template_pair(tmp_path: Path, signals: np.ndarray) -> DatasetConfig:
    n_samples = signals.shape[0]
    frame = pd.DataFrame(
        {
            "dataset": ["mitdb"] * n_samples,
            "record_id": ["201"] * n_samples,
            "beat_idx": list(range(n_samples)),
            "label_aami": [_LABELS[i % len(_LABELS)] for i in range(n_samples)],
            "r_peak_sample": [250] * n_samples,
        }
    )
    npz_path = tmp_path / "full.npz"
    parquet_path = tmp_path / "full.parquet"
    np.savez(npz_path, X=signals.astype(np.float32))
    frame.to_parquet(parquet_path, index=False)
    return DatasetConfig(
        stage2_npz=HashedPath(path=npz_path, sha256=sha256_file(npz_path)),
        stage2_parquet=HashedPath(path=parquet_path, sha256=sha256_file(parquet_path)),
        full_npz=HashedPath(path=npz_path, sha256=sha256_file(npz_path)),
        full_parquet=HashedPath(path=parquet_path, sha256=sha256_file(parquet_path)),
    )


def _config_with(datasets: DatasetConfig):
    config = load_research_config(PROJECT_ROOT / "config" / "stage2_research.yaml")
    return config.model_copy(update={"datasets": datasets})


def test_full_template_loader_accepts_single_channel_3d_npz(tmp_path: Path) -> None:
    """The authorized r4 source layout ``(n, 500, 1)`` must load unchanged."""
    rng = np.random.default_rng(7)
    signals = rng.normal(size=(12, 500, 1)).astype(np.float32)
    config = _config_with(_write_template_pair(tmp_path, signals))

    full = load_full_template_dataset(config)

    assert full.signals.shape == (12, 500, 1)
    assert full.labels.shape == (12,)
    assert set(np.unique(full.labels)) <= {"N", "S", "V", "F", "Q"}
    assert full.manifest["signal_shape"] == [12, 500, 1]


def test_full_template_loader_keeps_legacy_2d_contract(tmp_path: Path) -> None:
    """The legacy 2-D layout ``(n, 500)`` remains accepted."""
    rng = np.random.default_rng(11)
    signals = rng.normal(size=(9, 500)).astype(np.float32)
    config = _config_with(_write_template_pair(tmp_path, signals))

    full = load_full_template_dataset(config)

    assert full.signals.shape == (9, 500)


def test_full_template_loader_rejects_wrong_layout(tmp_path: Path) -> None:
    """Layouts outside the two documented contracts stay fail-closed."""
    rng = np.random.default_rng(13)
    signals = rng.normal(size=(5, 250)).astype(np.float32)
    config = _config_with(_write_template_pair(tmp_path, signals))

    with pytest.raises(ResearchError, match="full template source shape/length mismatch"):
        load_full_template_dataset(config)
