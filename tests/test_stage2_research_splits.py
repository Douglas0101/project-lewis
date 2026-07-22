"""Frozen Stage 2 split and leakage tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.stage2_research.config import load_research_config
from src.stage2_research.data import Stage2Dataset
from src.stage2_research.integrity import hash_canonical
from src.stage2_research.splits import (
    freeze_or_validate_splits,
    generate_split_manifests,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "stage2_research.yaml"
EXPECTED_FOLDS = (1, 2, 3, 4, 5)


def _synthetic_dataset() -> Stage2Dataset:
    groups = np.repeat(np.asarray([f"g{index:02d}" for index in range(30)]), 6)
    labels = np.tile(np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64), 30)
    frame = pd.DataFrame(
        {
            "dataset": "synthetic",
            "record_id": groups,
            "beat_idx": np.arange(labels.size),
            "label_aami": np.asarray(["S", "V", "F"])[labels],
        }
    )
    manifest = {"schema_version": "synthetic", "n_samples": labels.size}
    manifest_hash = hash_canonical(manifest)
    return Stage2Dataset(
        frame=frame,
        signals=np.zeros((labels.size, 500), dtype=np.float32),
        labels=labels,
        base_features=np.zeros((labels.size, 16), dtype=np.float32),
        groups=groups,
        manifest=manifest,
        manifest_hash=manifest_hash,
    )


def test_split_manifests_are_reproducible_and_nested() -> None:
    config = load_research_config(CONFIG_PATH)
    dataset = _synthetic_dataset()

    outer_a, inner_a = generate_split_manifests(config, dataset)
    outer_b, inner_b = generate_split_manifests(config, dataset)

    assert outer_a.manifest_hash == outer_b.manifest_hash
    assert inner_a.manifest_hash == inner_b.manifest_hash
    assert tuple(item.fold for item in outer_a.outer_folds) == EXPECTED_FOLDS
    for outer_fold, inner_fold in zip(
        outer_a.outer_folds,
        inner_a.inner_folds,
        strict=True,
    ):
        outer_train = set(outer_fold.train.groups)
        outer_test = set(outer_fold.test.groups)
        inner_train = set(inner_fold.train.groups)
        inner_validation = set(inner_fold.validation.groups)
        assert outer_train.isdisjoint(outer_test)
        assert inner_train.isdisjoint(inner_validation)
        assert inner_validation.isdisjoint(outer_test)
        assert inner_train <= outer_train
        assert inner_validation <= outer_train


def test_frozen_split_manifest_is_reused_without_regeneration(tmp_path: Path) -> None:
    config = load_research_config(CONFIG_PATH).model_copy(
        update={"output_root": tmp_path / "experiments"}
    )
    dataset = _synthetic_dataset()

    outer_first, inner_first, created_first = freeze_or_validate_splits(config, dataset)
    outer_second, inner_second, created_second = freeze_or_validate_splits(config, dataset)

    assert created_first
    assert not created_second
    assert outer_first == outer_second
    assert inner_first == inner_second
    assert (config.output_root / "splits" / "outer_splits_v2.4.json").is_file()
    assert (config.output_root / "splits" / "inner_splits_v2.4.json").is_file()
    assert (config.output_root / "splits" / "split_diagnostics.csv").is_file()
