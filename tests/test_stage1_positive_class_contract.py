"""Executable proof of the Stage 1 positive-class mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.prepare_two_stage_datasets import _prepare_stage1
from src.models.keras_artifact_inspector import inspect_keras_archive
from src.training_integrity.contracts import (
    DatasetRole,
    FoldAssignment,
    IdentityStatus,
    PatientIdentityManifest,
    PatientIdentityRecord,
    PatientSplitManifest,
)
from src.training_integrity.integrity import hash_canonical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "stage1_float32_v2.0.keras"
CONTRACT_PATH = PROJECT_ROOT / "docs" / "positive_class_contract.md"


def _identity_and_split() -> tuple[PatientIdentityManifest, PatientSplitManifest]:
    records = tuple(
        PatientIdentityRecord(
            dataset_id="mitdb",
            record_id=f"r{fold}",
            patient_id=f"mitdb:subject:{fold}",
            patient_group_id=f"mitdb:subject:{fold}",
            role=DatasetRole.CONFIRMATORY_CORE,
            identity_status=IdentityStatus.IDENTITY_VERIFIED,
            evidence_ref="fixture",
        )
        for fold in range(5)
    )
    identity = PatientIdentityManifest(
        schema_version="patient-identity-v3.1.0",
        source_data_hash="e" * 64,
        records=records,
        confirmatory_patient_count=5,
        confirmatory_record_count=5,
        quarantined_record_count=0,
    )
    split = PatientSplitManifest(
        schema_version="patient-split-v3.1.0",
        split_version="3.1.0",
        n_splits=5,
        random_state=42,
        source_data_hash=identity.source_data_hash,
        patient_identity_hash=hash_canonical("patient-identity", identity),
        core_dataset_ids=("mitdb",),
        quarantine_dataset_ids=(),
        folds=tuple(
            FoldAssignment(
                fold=fold,
                outer_test_patient_ids=(f"mitdb:subject:{fold}",),
                outer_test_record_keys=(f"mitdb/r{fold}",),
                n_samples=1,
                class_counts={"N": 1},
                dataset_counts={"mitdb": 1},
            )
            for fold in range(5)
        ),
        quarantined_records=(),
    )
    return identity, split


@pytest.mark.requires_artifacts
def test_stage1_integer_target_and_output_index_contract(tmp_path: Path) -> None:
    """N maps to 0 and every non-N AAMI source class maps to positive index 1."""
    source_labels = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    signals = np.zeros((len(source_labels), 500, 1), dtype=np.float32)
    labels = ["N", "S", "V", "F", "Q"]
    original = ["N", "A", "V", "F", "/"]
    canonical = ["N", "S", "V", "FUSION", "Q_OR_UNKNOWN"]
    indices = np.arange(1, 6, dtype=np.int64) * 100
    frame = pd.DataFrame(
        {
            "dataset": ["mitdb"] * 5,
            "record_id": [f"r{fold}" for fold in range(5)],
            "beat_idx": np.arange(5, dtype=np.int64),
            "r_peak_sample": indices,
            "label_aami": labels,
            "source_sampling_rate": [500.0] * 5,
            "target_sampling_rate": [500.0] * 5,
            "annotation_index_native": indices,
            "annotation_time_seconds": indices / 500.0,
            "annotation_index_target": indices,
            "class_original": original,
            "class_canonical": canonical,
            "qf_flatline": [False] * 5,
            "qf_clip": [False] * 5,
            "qf_off_center": [False] * 5,
            "rr_prev": np.ones(len(source_labels), dtype=np.float32),
            "qrs_width_ms": np.ones(len(source_labels), dtype=np.float32),
        }
    )
    output_npz = tmp_path / "stage1_binary.npz"
    output_parquet = tmp_path / "stage1_binary.parquet"

    identity, split = _identity_and_split()
    _prepare_stage1(
        signals,
        source_labels,
        frame,
        output_npz,
        output_parquet,
        identity=identity,
        split=split,
    )

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
