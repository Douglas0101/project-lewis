"""Testes para scripts/prepare_two_stage_datasets.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.prepare_two_stage_datasets import (
    AAMI_CLASSES,
    _prepare_stage1,
    _prepare_stage2,
)
from src.training_integrity.contracts import (
    DatasetRole,
    FoldAssignment,
    IdentityStatus,
    PatientIdentityManifest,
    PatientIdentityRecord,
    PatientSplitManifest,
)
from src.training_integrity.integrity import hash_canonical


def _contracts() -> tuple[PatientIdentityManifest, PatientSplitManifest]:
    records = tuple(
        PatientIdentityRecord(
            dataset_id="mitdb",
            record_id=f"rec_{fold}",
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
        source_data_hash="d" * 64,
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
                outer_test_record_keys=(f"mitdb/rec_{fold}",),
                n_samples=4,
                class_counts={"N": 1},
                dataset_counts={"mitdb": 4},
            )
            for fold in range(5)
        ),
        quarantined_records=(),
    )
    return identity, split


def _make_source(n: int = 20, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 500, 1)).astype(np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 2, dtype=np.int64)
    y = y[:n]
    labels = [AAMI_CLASSES[int(v)] for v in y]
    original_by_class = {"N": "N", "S": "A", "V": "V", "F": "F", "Q": "/"}
    canonical_by_class = {"N": "N", "S": "S", "V": "V", "F": "FUSION", "Q": "Q_OR_UNKNOWN"}
    target_indices = np.arange(1, n + 1, dtype=np.int64) * 100
    df = pd.DataFrame(
        {
            "dataset": ["mitdb"] * n,
            "record_id": [f"rec_{i % 5}" for i in range(n)],
            "beat_idx": np.arange(n, dtype=np.int64),
            "r_peak_sample": target_indices,
            "aami_label": labels,
            "label_aami": labels,
            "source_sampling_rate": [500.0] * n,
            "target_sampling_rate": [500.0] * n,
            "annotation_index_native": target_indices,
            "annotation_time_seconds": target_indices / 500.0,
            "annotation_index_target": target_indices,
            "class_original": [original_by_class[label] for label in labels],
            "class_canonical": [canonical_by_class[label] for label in labels],
            "qf_flatline": [False] * n,
            "qf_clip": [False] * n,
            "qf_off_center": [False] * n,
            "rr_prev": rng.uniform(0.6, 1.2, size=n),
            "qrs_width_ms": rng.uniform(60.0, 120.0, size=n),
        }
    )
    return X, y, df


def test_prepare_stage1_default_includes_q_as_abnormal(tmp_path):
    X, y, df = _make_source()
    npz_path = tmp_path / "stage1.npz"
    parquet_path = tmp_path / "stage1.parquet"

    identity, split = _contracts()
    _prepare_stage1(
        X,
        y,
        df,
        npz_path=npz_path,
        parquet_path=parquet_path,
        identity=identity,
        split=split,
    )

    data = np.load(npz_path)
    assert data["y"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    assert len(data["X"]) == len(data["y"])


def test_prepare_stage1_exclude_q_drops_q_samples(tmp_path):
    X, y, df = _make_source()
    npz_path = tmp_path / "stage1_no_q.npz"
    parquet_path = tmp_path / "stage1_no_q.parquet"

    identity, split = _contracts()
    _prepare_stage1(
        X,
        y,
        df,
        npz_path=npz_path,
        parquet_path=parquet_path,
        identity=identity,
        split=split,
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

    identity, split = _contracts()
    _prepare_stage1(
        X,
        y,
        df,
        npz_path=npz_path,
        parquet_path=parquet_path,
        identity=identity,
        split=split,
    )

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
