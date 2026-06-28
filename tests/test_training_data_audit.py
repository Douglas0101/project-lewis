"""Data-quality tests for the pre-training dataset and audit pipeline.

These tests validate that:
- the training catalog is loadable and non-empty;
- beat records produced by the feature pipeline satisfy the pydantic schema;
- the fine-tuning manifest (if already built) is valid;
- the audit script reports PASS on a small stratified sample;
- the generated feature parquet has no NaN/Inf values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.training_schemas import BeatRecord, TrainingDataAuditReport, TrainingDatasetManifest
from src.features.pipeline import (
    CATALOG_PATH,
    _build_beat_records,
    _find_processed_npy,
    _load_catalog,
    _load_raw_annotations,
)
from scripts.audit_training_data import AuditConfig, DataQualityAuditor

DATA_AVAILABLE = CATALOG_PATH.exists() and CATALOG_PATH.stat().st_size > 0
MITDB_100_AVAILABLE = (
    _find_processed_npy("100", "mitdb") is not None
    and _load_raw_annotations("100", "mitdb") is not None
)
FEATURES_AVAILABLE = Path("data/features/training_manifest.json").exists()


@pytest.mark.skipif(not DATA_AVAILABLE, reason="catalog not built yet")
def test_catalog_loadable() -> None:
    assert CATALOG_PATH.exists(), "catalog not found"
    records = _load_catalog(CATALOG_PATH)
    assert len(records) > 0
    datasets = {r["dataset"] for r in records}
    assert "mitdb" in datasets


@pytest.mark.skipif(not MITDB_100_AVAILABLE, reason="MITDB record 100 not available")
def test_build_beat_records_for_mitdb_100() -> None:
    npy_path = _find_processed_npy("100", "mitdb")
    assert npy_path is not None
    sig = np.load(npy_path).astype(np.float32)
    r_peaks, labels = _load_raw_annotations("100", "mitdb")
    beats, X_rec, y_rec = _build_beat_records(
        sig,
        r_peaks,
        labels,
        "100",
        "mitdb",
        str(Path("data/lineage/mitdb/100_lineage.json")),
    )
    assert len(beats) > 0
    assert len(beats) == len(X_rec) == len(y_rec)
    for b in beats[:10]:
        assert isinstance(b, BeatRecord)
        assert b.label_aami in {"N", "S", "V", "F", "Q"}
        assert b.segment_shape[0] in (300, 500)
        assert not np.isnan(b.morph.qrs_width_ms)
        assert not np.isnan(b.morph.qrs_area)


@pytest.mark.skipif(not FEATURES_AVAILABLE, reason="features not built yet")
def test_finetuning_manifest_valid() -> None:
    manifest_path = Path("data/features/training_manifest.json")
    manifest = TrainingDatasetManifest.model_validate_json(manifest_path.read_text())
    assert manifest.n_beats > 0
    assert "mitdb" in manifest.datasets_included
    assert all(c in manifest.global_class_distribution for c in ("N", "S", "V", "F", "Q"))
    assert manifest.quality_flags.no_nan_inf


@pytest.mark.skipif(not FEATURES_AVAILABLE, reason="features not built yet")
def test_finetuning_artifacts_sane() -> None:
    parquet_path = Path("data/features/finetuning_mitbih_family.parquet")
    npz_path = Path("data/features/finetuning_mitbih_family.npz")
    if not parquet_path.exists() or not npz_path.exists():
        pytest.skip("features not built yet")
    df = pd.read_parquet(parquet_path)
    assert not df.isnull().any().any(), f"NaN columns: {list(df.columns[df.isnull().any()])}"
    numeric = df.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()

    npz = np.load(npz_path)
    X = npz["X"]
    y = npz["y"]
    assert X.dtype == np.float32
    assert X.shape == (len(df), 500)
    assert y.dtype == np.int8
    assert y.shape == (len(df),)
    assert set(np.unique(y).tolist()).issubset({0, 1, 2, 3, 4})


@pytest.mark.skipif(not DATA_AVAILABLE, reason="catalog not available")
def test_audit_script_passes_small_sample(tmp_path: Path) -> None:
    cfg = AuditConfig(sample_size=20)
    auditor = DataQualityAuditor(cfg=cfg, dlq_path=tmp_path / "dlq.jsonl")
    report = auditor.run(
        catalog_path=CATALOG_PATH,
        output_dir=tmp_path,
    )
    assert isinstance(report, TrainingDataAuditReport)
    assert report.overall_status == "PASS"
    assert report.n_records_inspected > 0
    assert (tmp_path / "training_data_audit.json").exists()
    assert (tmp_path / "training_data_audit.md").exists()
