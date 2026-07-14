"""Executable proof of the Stage 1 positive-class mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_two_stage_datasets import _prepare_stage1
from src.models.keras_artifact_inspector import inspect_keras_archive

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "stage1_float32_v2.0.keras"
CONTRACT_PATH = PROJECT_ROOT / "docs" / "positive_class_contract.md"


def test_stage1_integer_target_and_output_index_contract(tmp_path: Path) -> None:
    """N maps to 0 and every non-N AAMI source class maps to positive index 1."""
    source_labels = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    signals = np.zeros((len(source_labels), 500, 1), dtype=np.float32)
    frame = pd.DataFrame(
        {
            "rr_prev": np.ones(len(source_labels), dtype=np.float32),
            "qrs_width_ms": np.ones(len(source_labels), dtype=np.float32),
        }
    )
    output_npz = tmp_path / "stage1_binary.npz"
    output_parquet = tmp_path / "stage1_binary.parquet"

    _prepare_stage1(signals, source_labels, frame, output_npz, output_parquet)

    with np.load(output_npz, allow_pickle=False) as prepared:
        assert prepared["y"].tolist() == [0, 1, 1, 1, 1]

    inspection = inspect_keras_archive(MODEL_PATH)
    assert inspection.output_units == 2
    assert inspection.output_activation == "softmax"
    assert inspection.compile_loss == "sparse_categorical_crossentropy"


def test_positive_class_contract_is_explicitly_documented() -> None:
    """The semantic mapping must not depend on pipeline column usage alone."""
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    normalized = " ".join(contract.split())

    assert "output index `0 = N/Normal`" in normalized
    assert "output index `1 = Anormal`" in normalized
    assert "does not rely only on the inference pipeline selecting column 1" in normalized
    assert "commit `27ad38b`" in normalized
