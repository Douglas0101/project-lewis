from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.prepare_two_stage_datasets import _prepare_stage1
from src.training_integrity.contracts import (
    CheckStatus,
    DatasetRole,
    EpistemicCategory,
    FoldAssignment,
    IdentityStatus,
    PatientIdentityManifest,
    PatientIdentityRecord,
    PatientSplitManifest,
    PreflightCheck,
    PreflightEvidenceBundle,
)
from src.training_integrity.integrity import (
    afdb_episode_sample_id,
    beat_sample_id,
    hash_canonical,
)
from src.training_integrity.preflight import (
    REQUIRED_AFDB_LINEAGE_COLUMNS,
    REQUIRED_SAMPLE_LINEAGE_COLUMNS,
    afdb_lineage_schema_check,
    afdb_lineage_value_check,
    afdb_source_reconstruction_check,
    family_source_reconstruction_check,
    finalize_preflight_report,
    ordered_row_binding_check,
    publish_pretraining_gate,
    sample_lineage_schema_check,
    sample_lineage_value_check,
    stage1_parent_binding_check,
    waveform_row_sha256,
)


def _write_parquet(path: Path, columns: dict[str, list[object]]) -> None:
    pq.write_table(pa.table(columns), path)


def test_default_beat_family_excludes_afdb_rhythm_dataset() -> None:
    from src.features.pipeline import FINETUNE_DATASETS

    assert FINETUNE_DATASETS == ("mitdb", "svdb", "incart")
    assert "afdb" not in FINETUNE_DATASETS


def test_sample_lineage_schema_blocks_missing_custody_fields(tmp_path: Path) -> None:
    path = tmp_path / "stage1.parquet"
    _write_parquet(
        path,
        {
            "dataset_id": ["mitdb"],
            "record_id": ["100"],
            "annotation_index_target": [100],
        },
    )

    check = sample_lineage_schema_check(path)
    assert check.status is CheckStatus.BLOCK
    assert "patient_id" in check.details["missing_columns"]
    assert "annotation_index_native" in check.details["missing_columns"]
    assert set(check.details["missing_columns"]) == set(REQUIRED_SAMPLE_LINEAGE_COLUMNS) - {
        "dataset_id",
        "record_id",
        "annotation_index_target",
    }


def test_sample_lineage_schema_passes_complete_contract(tmp_path: Path) -> None:
    path = tmp_path / "stage1.parquet"
    _write_parquet(path, {column: ["x"] for column in REQUIRED_SAMPLE_LINEAGE_COLUMNS})
    check = sample_lineage_schema_check(path)
    assert check.status is CheckStatus.PASS


def _single_patient_identity() -> PatientIdentityManifest:
    return PatientIdentityManifest(
        schema_version="patient-identity-v3.1.0",
        source_data_hash="a" * 64,
        records=(
            PatientIdentityRecord(
                dataset_id="mitdb",
                record_id="100",
                patient_id="mitdb:subject:100",
                patient_group_id="mitdb:subject:100",
                role=DatasetRole.CONFIRMATORY_CORE,
                identity_status=IdentityStatus.IDENTITY_VERIFIED,
                evidence_ref="fixture",
            ),
        ),
        confirmatory_patient_count=1,
        confirmatory_record_count=1,
        quarantined_record_count=0,
    )


def _single_patient_split(identity: PatientIdentityManifest) -> PatientSplitManifest:
    folds = []
    for fold in range(5):
        patient_id = "mitdb:subject:100" if fold == 0 else f"mitdb:subject:dummy-{fold}"
        record_key = "mitdb/100" if fold == 0 else f"mitdb/dummy-{fold}"
        folds.append(
            FoldAssignment(
                fold=fold,
                outer_test_patient_ids=(patient_id,),
                outer_test_record_keys=(record_key,),
                n_samples=1,
                class_counts={"N": 1},
                dataset_counts={"mitdb": 1},
            )
        )
    return PatientSplitManifest(
        schema_version="patient-split-v3.1.0",
        split_version="3.1.0",
        n_splits=5,
        random_state=42,
        source_data_hash=identity.source_data_hash,
        patient_identity_hash=hash_canonical("patient-identity", identity),
        core_dataset_ids=("mitdb",),
        quarantine_dataset_ids=(),
        folds=tuple(folds),
        quarantined_records=(),
    )


def _producer_identity() -> PatientIdentityManifest:
    records = [
        PatientIdentityRecord(
            dataset_id="mitdb",
            record_id="100" if fold == 0 else f"dummy-{fold}",
            patient_id=("mitdb:subject:100" if fold == 0 else f"mitdb:subject:dummy-{fold}"),
            patient_group_id=("mitdb:subject:100" if fold == 0 else f"mitdb:subject:dummy-{fold}"),
            role=DatasetRole.CONFIRMATORY_CORE,
            identity_status=IdentityStatus.IDENTITY_VERIFIED,
            evidence_ref="fixture",
        )
        for fold in range(5)
    ]
    return PatientIdentityManifest(
        schema_version="patient-identity-v3.1.0",
        source_data_hash="a" * 64,
        records=tuple(records),
        confirmatory_patient_count=5,
        confirmatory_record_count=5,
        quarantined_record_count=0,
    )


def _lineage_columns(
    *, target_index: int, fold: int = 0, binary_label: int = 0
) -> dict[str, list[object]]:
    return {
        "dataset_id": ["mitdb"],
        "patient_id": ["mitdb:subject:100"],
        "record_id": ["100"],
        "beat_index": [0],
        "segment_id": [beat_sample_id("mitdb", "100", 0, target_index)],
        "sample_id": [beat_sample_id("mitdb", "100", 0, target_index)],
        "waveform_sha256": ["a" * 64],
        "source_sampling_rate": [360.0],
        "target_sampling_rate": [500.0],
        "annotation_index_native": [72],
        "annotation_time_seconds": [0.2],
        "annotation_index_target": [target_index],
        "class_original": ["N"],
        "class_canonical": ["N"],
        "y": [binary_label],
        "quality_label": ["VALID"],
        "split": ["outer_test"],
        "fold": [fold],
    }


def test_sample_lineage_values_validate_dual_clock_and_identity(tmp_path: Path) -> None:
    path = tmp_path / "lineage.parquet"
    _write_parquet(path, _lineage_columns(target_index=100))
    identity = _single_patient_identity()
    check = sample_lineage_value_check(path, identity, _single_patient_split(identity))
    assert check.status is CheckStatus.PASS


def test_sample_lineage_values_reject_clock_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "lineage.parquet"
    _write_parquet(path, _lineage_columns(target_index=101))
    identity = _single_patient_identity()
    check = sample_lineage_value_check(path, identity, _single_patient_split(identity))
    assert check.status is CheckStatus.BLOCK
    assert check.code == "SAMPLE_LINEAGE_VALUES_INVALID"


def test_sample_lineage_values_reject_wrong_patient_fold(tmp_path: Path) -> None:
    path = tmp_path / "lineage.parquet"
    _write_parquet(path, _lineage_columns(target_index=100, fold=1))
    identity = _single_patient_identity()
    check = sample_lineage_value_check(path, identity, _single_patient_split(identity))
    assert check.status is CheckStatus.BLOCK
    assert check.details["issue_counts"]["confirmatory_split_mismatch"] == 1


def test_sample_lineage_values_reject_binary_label_semantic_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "lineage.parquet"
    _write_parquet(path, _lineage_columns(target_index=100, binary_label=1))
    identity = _single_patient_identity()
    check = sample_lineage_value_check(path, identity, _single_patient_split(identity))
    assert check.status is CheckStatus.BLOCK
    assert check.details["issue_counts"]["binary_label_semantic_mismatch"] == 1


def test_stage1_parent_binding_rejects_incomplete_child_population(tmp_path: Path) -> None:
    parent_path = tmp_path / "family.parquet"
    valid_stage_path = tmp_path / "stage-valid.parquet"
    subset_stage_path = tmp_path / "stage-subset.parquet"
    parent: dict[str, list[object]] = {
        "sample_id": ["n", "q", "v"],
        "waveform_sha256": ["a" * 64, "b" * 64, "c" * 64],
        "label_aami": ["N", "Q", "V"],
        "class_canonical": ["N", "Q_OR_UNKNOWN", "V"],
        "y": [0, 4, 2],
    }
    _write_parquet(parent_path, parent)
    valid_stage: dict[str, list[object]] = {
        "sample_id": ["n", "v"],
        "waveform_sha256": ["a" * 64, "c" * 64],
        "class_canonical": ["N", "V"],
        "y": [0, 1],
    }
    _write_parquet(valid_stage_path, valid_stage)
    assert stage1_parent_binding_check(parent_path, valid_stage_path).status is CheckStatus.PASS
    _write_parquet(
        subset_stage_path,
        {column: values[:1] for column, values in valid_stage.items()},
    )
    check = stage1_parent_binding_check(parent_path, subset_stage_path)
    assert check.status is CheckStatus.BLOCK
    assert check.code == "STAGE1_PARENT_BINDING_INVALID"


def test_family_source_reconstruction_rejects_missing_eligible_beats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.features.pipeline as pipeline
    from src.data.segmenter import ECGSegmenter

    signal_path = tmp_path / "100_MLII.npy"
    signal = np.linspace(-1.0, 1.0, 2000, dtype=np.float32)
    np.save(signal_path, signal)
    custody = pipeline.AnnotationCustody(
        native_samples=np.array([360, 720, 1080]),
        target_samples=np.array([500, 1000, 1500]),
        original_symbols=np.array(["N", "V", "A"]),
        canonical_labels=np.array(["N", "V", "S"]),
        source_sampling_rate=360.0,
    )
    monkeypatch.setattr(pipeline, "_load_raw_annotation_custody", lambda record, dataset: custody)
    monkeypatch.setattr(pipeline, "_find_processed_npy", lambda record, dataset: signal_path)
    segmenter = ECGSegmenter(fs=500.0, window_ms=1000.0, min_window_ms=600.0)
    waveforms, labels, metadata = segmenter.segment_with_labels(
        signal,
        custody.target_samples,
        custody.canonical_labels,
        rr_intervals_ms=None,
    )
    kept = np.asarray(metadata["kept_indices"], dtype=np.int64)
    label_to_int = {"N": 0, "S": 1, "V": 2}
    sample_ids = [
        beat_sample_id("mitdb", "100", int(index), int(custody.target_samples[index]))
        for index in kept
    ]
    frame = {
        "dataset": ["mitdb"] * len(kept),
        "record_id": ["100"] * len(kept),
        "beat_idx": kept.tolist(),
        "sample_id": sample_ids,
        "segment_id": sample_ids,
        "waveform_sha256": [waveform_row_sha256(row[..., np.newaxis]) for row in waveforms],
        "source_sampling_rate": [360.0] * len(kept),
        "target_sampling_rate": [500.0] * len(kept),
        "annotation_index_native": custody.native_samples[kept].tolist(),
        "annotation_time_seconds": (custody.native_samples[kept] / 360.0).tolist(),
        "annotation_index_target": custody.target_samples[kept].tolist(),
        "class_original": custody.original_symbols[kept].tolist(),
        "class_canonical": labels.tolist(),
        "label_aami": labels.tolist(),
        "y": [label_to_int[str(label)] for label in labels],
    }
    complete_path = tmp_path / "family-complete.parquet"
    incomplete_path = tmp_path / "family-incomplete.parquet"
    _write_parquet(complete_path, frame)
    assert (
        family_source_reconstruction_check(complete_path, _single_patient_identity()).status
        is CheckStatus.PASS
    )
    _write_parquet(
        incomplete_path,
        {column: values[:-1] for column, values in frame.items()},
    )
    check = family_source_reconstruction_check(incomplete_path, _single_patient_identity())
    assert check.status is CheckStatus.BLOCK
    assert check.details["issue_counts"]["source_beat_count_mismatch"] == 1


def _afdb_identity() -> PatientIdentityManifest:
    return PatientIdentityManifest(
        schema_version="patient-identity-v3.1.0",
        source_data_hash="b" * 64,
        records=(
            PatientIdentityRecord(
                dataset_id="afdb",
                record_id="04015",
                patient_id=None,
                patient_group_id=None,
                role=DatasetRole.RHYTHM_EXPLORATORY,
                identity_status=IdentityStatus.IDENTITY_UNVERIFIED,
                evidence_ref="fixture",
            ),
        ),
        confirmatory_patient_count=0,
        confirmatory_record_count=0,
        quarantined_record_count=1,
    )


def _afdb_lineage_columns(*, end_target: int = 5000) -> dict[str, list[object]]:
    sample_id = afdb_episode_sample_id("04015", 0, 0, end_target)
    return {
        "dataset_id": ["afdb"],
        "record_id": ["04015"],
        "patient_id": ["UNKNOWN_OR_UNVERIFIED"],
        "episode_idx": [0],
        "segment_id": [sample_id],
        "sample_id": [sample_id],
        "waveform_sha256": ["c" * 64],
        "source_sampling_rate": [250.0],
        "target_sampling_rate": [500.0],
        "start_sample_native": [0],
        "end_sample_native": [2500],
        "start_time_seconds": [0.0],
        "end_time_seconds": [10.0],
        "start_sample_target": [0],
        "end_sample_target": [end_target],
        "interval_start_native": [0],
        "interval_end_native": [5000],
        "interval_start_target": [0],
        "interval_end_target": [10000],
        "rhythm_original": ["AFIB"],
        "rhythm_canonical": ["AFIB"],
        "split": ["rhythm_exploratory"],
        "fold": [-1],
    }


def test_afdb_lineage_validates_source_interval_and_clocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.features.afdb_rhythm as afdb_rhythm

    path = tmp_path / "afdb.parquet"
    _write_parquet(path, _afdb_lineage_columns())
    monkeypatch.setattr(
        afdb_rhythm,
        "_load_rhythm_intervals_500",
        lambda base: [(0, 5000, 0, 10000, "AFIB", "AFIB")],
    )
    assert afdb_lineage_schema_check(path).status is CheckStatus.PASS
    check = afdb_lineage_value_check(path, _afdb_identity(), raw_dir=tmp_path)
    assert check.status is CheckStatus.PASS


def test_afdb_lineage_rejects_episode_boundary_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.features.afdb_rhythm as afdb_rhythm

    path = tmp_path / "afdb.parquet"
    _write_parquet(path, _afdb_lineage_columns(end_target=4999))
    monkeypatch.setattr(
        afdb_rhythm,
        "_load_rhythm_intervals_500",
        lambda base: [(0, 5000, 0, 10000, "AFIB", "AFIB")],
    )
    check = afdb_lineage_value_check(path, _afdb_identity(), raw_dir=tmp_path)
    assert check.status is CheckStatus.BLOCK
    assert check.details["issue_counts"]["episode_boundary_mismatch"] == 1


def test_afdb_source_reconstruction_rejects_episode_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.features.afdb_rhythm as afdb_rhythm

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    (raw_dir / "04015.hea").write_text("fixture\n", encoding="utf-8")
    (raw_dir / "04015.atr").write_bytes(b"fixture")
    signal = np.linspace(-1.0, 1.0, 10000, dtype=np.float32)
    np.save(processed_dir / "04015_ECG1.npy", signal)
    interval = (0, 5000, 0, 10000, "AFIB", "AFIB")
    monkeypatch.setattr(
        afdb_rhythm,
        "_load_rhythm_intervals_500",
        lambda base, length: [interval],
    )
    rows: list[tuple[object, ...]] = []
    for episode_index, start in enumerate((0, 5000)):
        end = start + 5000
        sample_id = afdb_episode_sample_id("04015", episode_index, start, end)
        rows.append(
            (
                "04015",
                sample_id,
                waveform_row_sha256(signal[start:end].reshape(5000, 1)),
                *interval,
            )
        )
    columns = [
        "record_id",
        "sample_id",
        "waveform_sha256",
        "interval_start_native",
        "interval_end_native",
        "interval_start_target",
        "interval_end_target",
        "rhythm_original",
        "rhythm_canonical",
    ]
    complete = tmp_path / "afdb-complete.parquet"
    subset = tmp_path / "afdb-subset.parquet"
    _write_parquet(
        complete,
        {column: [row[index] for row in rows] for index, column in enumerate(columns)},
    )
    assert (
        afdb_source_reconstruction_check(
            complete, raw_dir=raw_dir, processed_dir=processed_dir
        ).status
        is CheckStatus.PASS
    )
    _write_parquet(
        subset,
        {column: [rows[0][index]] for index, column in enumerate(columns)},
    )
    check = afdb_source_reconstruction_check(subset, raw_dir=raw_dir, processed_dir=processed_dir)
    assert check.status is CheckStatus.BLOCK
    assert check.code == "AFDB_SOURCE_RECONSTRUCTION_INVALID"


def test_afdb_lineage_required_columns_are_canonical() -> None:
    assert "interval_start_native" in REQUIRED_AFDB_LINEAGE_COLUMNS
    assert "rhythm_canonical" in REQUIRED_AFDB_LINEAGE_COLUMNS


def test_ordered_row_binding_requires_sample_and_waveform_identity(tmp_path: Path) -> None:
    npz_path = tmp_path / "data.npz"
    parquet_path = tmp_path / "data.parquet"
    np.savez(npz_path, X=np.zeros((2, 500, 1), dtype=np.float32), y=np.array([0, 0]))
    _write_parquet(parquet_path, {"y": [0, 0]})

    check = ordered_row_binding_check(npz_path, parquet_path)
    assert check.status is CheckStatus.BLOCK
    assert check.code == "ORDERED_ROW_BINDING_INCOMPLETE"


def test_ordered_row_binding_detects_same_class_permutation(tmp_path: Path) -> None:
    npz_path = tmp_path / "data.npz"
    parquet_path = tmp_path / "data.parquet"
    sample_ids = np.array(["a", "b"])
    waveforms = np.stack(
        [
            np.zeros((500, 1), dtype=np.float32),
            np.ones((500, 1), dtype=np.float32),
        ]
    )
    waveform_hashes = np.array([waveform_row_sha256(row) for row in waveforms])
    np.savez(
        npz_path,
        X=waveforms,
        y=np.array([0, 0]),
        sample_id=sample_ids,
        waveform_sha256=waveform_hashes,
    )
    _write_parquet(
        parquet_path,
        {
            "sample_id": ["b", "a"],
            "waveform_sha256": waveform_hashes[::-1].tolist(),
            "y": [0, 0],
        },
    )

    check = ordered_row_binding_check(npz_path, parquet_path)
    assert check.status is CheckStatus.BLOCK
    assert check.code == "ORDERED_ROW_BINDING_MISMATCH"


def test_ordered_row_binding_passes_exact_order(tmp_path: Path) -> None:
    npz_path = tmp_path / "data.npz"
    parquet_path = tmp_path / "data.parquet"
    sample_ids = np.array(["a", "b"])
    waveforms = np.stack(
        [
            np.zeros((500, 1), dtype=np.float32),
            np.ones((500, 1), dtype=np.float32),
        ]
    )
    waveform_hashes = np.array([waveform_row_sha256(row) for row in waveforms])
    np.savez(
        npz_path,
        X=waveforms,
        y=np.array([0, 1]),
        sample_id=sample_ids,
        waveform_sha256=waveform_hashes,
    )
    _write_parquet(
        parquet_path,
        {
            "sample_id": sample_ids.tolist(),
            "waveform_sha256": waveform_hashes.tolist(),
            "y": [0, 1],
        },
    )

    check = ordered_row_binding_check(npz_path, parquet_path)
    assert check.status is CheckStatus.PASS


def test_stage1_producer_emits_verifiable_ordered_binding(tmp_path: Path) -> None:
    import pandas as pd

    npz_path = tmp_path / "stage1.npz"
    parquet_path = tmp_path / "stage1.parquet"
    waveforms = np.stack(
        [
            np.zeros((500, 1), dtype=np.float32),
            np.ones((500, 1), dtype=np.float32),
        ]
    )
    frame = pd.DataFrame(
        {
            "dataset": ["mitdb", "mitdb"],
            "record_id": ["100", "100"],
            "beat_idx": [0, 1],
            "r_peak_sample": [100, 200],
            "label_aami": ["N", "V"],
            "source_sampling_rate": [360.0, 360.0],
            "target_sampling_rate": [500.0, 500.0],
            "annotation_index_native": [72, 144],
            "annotation_time_seconds": [0.2, 0.4],
            "annotation_index_target": [100, 200],
            "class_original": ["N", "V"],
            "class_canonical": ["N", "V"],
            "rr_prev": [800.0, 810.0],
            "qrs_width_ms": [90.0, 110.0],
            "qf_flatline": [False, False],
            "qf_clip": [False, False],
            "qf_off_center": [False, False],
        }
    )
    identity = _producer_identity()
    _prepare_stage1(
        waveforms,
        np.array([0, 2], dtype=np.int64),
        frame,
        npz_path,
        parquet_path,
        identity=identity,
        split=_single_patient_split(identity),
        exclude_q=True,
    )
    check = ordered_row_binding_check(npz_path, parquet_path)
    assert check.status is CheckStatus.PASS


def test_blocked_preflight_cannot_publish_gate(tmp_path: Path) -> None:
    report = finalize_preflight_report(
        generation_id="advanced-training-v3.1.0-r1",
        checks=(
            PreflightCheck(
                code="SAMPLE_LINEAGE_INCOMPLETE",
                status=CheckStatus.BLOCK,
                epistemic_category=EpistemicCategory.OBSERVED,
                evidence="required columns absent",
                denominator="1 parquet schema",
                limitation="native annotation evidence is unavailable",
                details={},
            ),
        ),
    )
    assert report.final_state == "REVIEW_REQUIRED"
    assert report.training_allowed is False

    with pytest.raises(ValueError, match="blocked preflight"):
        publish_pretraining_gate(
            tmp_path / "PRETRAINING_GATE_PASS",
            cast(PreflightEvidenceBundle, SimpleNamespace(preflight_report=report)),
            config_path=tmp_path / "missing.yaml",
            project_root=tmp_path,
        )
    assert not (tmp_path / "PRETRAINING_GATE_PASS").exists()
