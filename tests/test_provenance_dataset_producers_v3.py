"""Regression tests for provenance-bound v3.1 dataset producers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.features import afdb_rhythm, pipeline
from src.training_integrity.contracts import (
    DatasetRole,
    IdentityStatus,
    PatientIdentityManifest,
    PatientIdentityRecord,
)
from src.training_integrity.preflight import family_source_reconstruction_check


def test_family_producer_emits_channel_axis_and_pickle_free_npz(
    tmp_path: Path,
    monkeypatch,
) -> None:
    processed = tmp_path / "record.npy"
    signal = np.sin(np.linspace(0.0, 400.0, 61_000, dtype=np.float32))
    np.save(processed, signal)
    target_samples = np.concatenate(
        [
            np.asarray([-29, 420, 640], dtype=np.int64),
            1_082 + np.arange(98, dtype=np.int64) * 600,
        ]
    )
    symbols = np.asarray(["N"] * len(target_samples), dtype=str)
    custody = pipeline.AnnotationCustody(
        native_samples=target_samples,
        target_samples=target_samples,
        original_symbols=symbols,
        canonical_labels=symbols,
        source_sampling_rate=500.0,
    )
    monkeypatch.setattr(
        pipeline,
        "_load_catalog",
        lambda _path: [{"dataset": "mitdb", "record_name": "fixture"}],
    )
    monkeypatch.setattr(pipeline, "_find_processed_npy", lambda *_args: processed)
    monkeypatch.setattr(pipeline, "_load_raw_annotation_custody", lambda *_args: custody)

    parquet_path = tmp_path / "family.parquet"
    pipeline.build_finetuning_dataset(parquet_path, datasets=["mitdb"])

    with np.load(parquet_path.with_suffix(".npz"), allow_pickle=False) as archive:
        assert archive["X"].shape[1:] == (500, 1)
        for member in archive.files:
            assert archive[member].dtype.kind != "O"

    identity = PatientIdentityManifest(
        schema_version="patient-identity-v3.1.0",
        source_data_hash="f" * 64,
        records=(
            PatientIdentityRecord(
                dataset_id="mitdb",
                record_id="fixture",
                patient_id="mitdb:subject:fixture",
                patient_group_id="mitdb:subject:fixture",
                role=DatasetRole.CONFIRMATORY_CORE,
                identity_status=IdentityStatus.IDENTITY_VERIFIED,
                evidence_ref="fixture",
            ),
        ),
        confirmatory_patient_count=1,
        confirmatory_record_count=1,
        quarantined_record_count=0,
    )
    reconstruction = family_source_reconstruction_check(parquet_path, identity)
    assert reconstruction.code == "FAMILY_SOURCE_RECONSTRUCTION_VALIDATED"


def test_afdb_producer_emits_pickle_free_identity_arrays(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    output_dir = tmp_path / "output"
    raw_dir.mkdir()
    processed_dir.mkdir()
    (raw_dir / "00001.hea").touch()
    (raw_dir / "00001.atr").touch()
    signal = np.linspace(-1.0, 1.0, 10_000, dtype=np.float32)
    np.save(processed_dir / "00001_ECG1.npy", signal)
    monkeypatch.setattr(
        afdb_rhythm,
        "_load_rhythm_intervals_500",
        lambda _base, _length: [(0, 5_000, 0, 10_000, "N", "SINUS")],
    )

    stats = afdb_rhythm.build_afdb_rhythm_episodes(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        output_dir=output_dir,
    )

    assert stats["n_episodes"] == 2
    with np.load(output_dir / "afdb_rhythm_episodes.npz", allow_pickle=False) as archive:
        assert archive["X"].shape == (2, 5_000, 1)
        for member in archive.files:
            assert archive[member].dtype.kind != "O"
