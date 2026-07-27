"""Ordered Stage 2 custody generation tests for E07R."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.stage2_research.stage2_custody import (
    build_stage2_custody_bundle,
    publish_stage2_custody_generation,
)
from src.training_integrity.contracts import CheckStatus
from src.training_integrity.integrity import sha256_file, waveform_row_sha256
from src.training_integrity.preflight import ordered_row_binding_check


def _parent(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(42)
    waveforms = rng.normal(size=(15, 500, 1)).astype(np.float32)
    datasets = np.asarray(
        ["mitdb"] * 6 + ["incart"] * 6 + ["svdb"] * 3,
        dtype=str,
    )
    labels = np.asarray(
        ["N", "S", "V", "F", "Q", "S"] * 2 + ["S", "V", "F"],
        dtype=str,
    )
    record_ids = np.asarray(
        ["100", "100", "101", "201", "202", "202"]
        + ["I01", "I01", "I02", "I02", "I03", "I03"]
        + ["800", "801", "802"],
        dtype=str,
    )
    sample_ids = np.asarray(
        [
            f"{dataset}:{record}:beat:{index}"
            for index, (dataset, record) in enumerate(zip(datasets, record_ids, strict=True))
        ],
        dtype=str,
    )
    waveform_hashes = np.asarray(
        [waveform_row_sha256(row) for row in waveforms],
        dtype=str,
    )
    parent_y = np.asarray(
        [{"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}[label] for label in labels],
        dtype=np.int8,
    )
    frame = pd.DataFrame(
        {
            "dataset": datasets,
            "record_id": record_ids,
            "beat_idx": np.arange(len(labels), dtype=np.int64),
            "label_aami": labels,
            "sample_id": sample_ids,
            "segment_id": sample_ids,
            "waveform_sha256": waveform_hashes,
            "source_sampling_rate": 360.0,
            "target_sampling_rate": 500.0,
            "annotation_index_native": np.arange(len(labels), dtype=np.int64),
            "annotation_index_target": np.arange(len(labels), dtype=np.int64),
            "class_original": labels,
            "class_canonical": labels,
            "y": parent_y,
        }
    )
    npz_path = tmp_path / "parent.npz"
    parquet_path = tmp_path / "parent.parquet"
    np.savez_compressed(
        npz_path,
        X=waveforms,
        y=parent_y,
        sample_id=sample_ids,
        waveform_sha256=waveform_hashes,
    )
    frame.to_parquet(parquet_path, index=False)
    return npz_path, parquet_path


def test_stage2_custody_filters_ordered_confirmatory_rows(tmp_path: Path) -> None:
    parent_npz, parent_parquet = _parent(tmp_path)
    bundle = build_stage2_custody_bundle(
        parent_npz,
        parent_parquet,
        expected_parent_npz_sha256=sha256_file(parent_npz),
        expected_parent_parquet_sha256=sha256_file(parent_parquet),
        source_commit="1" * 40,
        source_manifest_hash="2" * 64,
    )

    assert set(bundle.frame["dataset"].astype(str)) == {"incart", "mitdb"}
    assert set(bundle.frame["label_aami"].astype(str)) == {"S", "V", "F"}
    assert bundle.waveforms.shape == (8, 500, 1)
    assert bundle.labels.tolist() == [0, 1, 2, 0, 0, 1, 2, 0]
    assert np.array_equal(bundle.sample_ids, bundle.frame["sample_id"].astype(str))
    assert np.array_equal(
        bundle.waveform_hashes,
        bundle.frame["waveform_sha256"].astype(str),
    )


def test_stage2_custody_publication_is_ordered_and_write_once(tmp_path: Path) -> None:
    parent_npz, parent_parquet = _parent(tmp_path)
    bundle = build_stage2_custody_bundle(
        parent_npz,
        parent_parquet,
        expected_parent_npz_sha256=sha256_file(parent_npz),
        expected_parent_parquet_sha256=sha256_file(parent_parquet),
        source_commit="1" * 40,
        source_manifest_hash="2" * 64,
    )
    target = tmp_path / "v3.1.0-r5-stage2-pd"

    manifest, complete = publish_stage2_custody_generation(target, bundle)

    assert manifest.status == "AUTHORIZED_FOR_E07R_INTERNAL_TRAINING"
    assert complete.status == "COMPLETE"
    check = ordered_row_binding_check(
        target / "stage2_multiclass.npz",
        target / "stage2_multiclass.parquet",
        scope="STAGE2_R5",
    )
    assert check.status is CheckStatus.PASS
    with np.load(target / "stage2_multiclass.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"X", "y", "sample_id", "waveform_sha256"}

    with pytest.raises(FileExistsError, match="immutable publication"):
        publish_stage2_custody_generation(target, bundle)


def test_stage2_custody_rejects_parent_hash_mismatch(tmp_path: Path) -> None:
    parent_npz, parent_parquet = _parent(tmp_path)

    with pytest.raises(ValueError, match="parent NPZ SHA-256 mismatch"):
        build_stage2_custody_bundle(
            parent_npz,
            parent_parquet,
            expected_parent_npz_sha256="0" * 64,
            expected_parent_parquet_sha256=sha256_file(parent_parquet),
            source_commit="1" * 40,
            source_manifest_hash="2" * 64,
        )
