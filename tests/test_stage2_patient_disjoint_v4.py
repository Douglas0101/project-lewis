"""Patient-disjoint Stage 2 v4.0 identity and split contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.e06_protocol import E06EvaluationContract
from src.stage2_research.e07r_contracts import (
    PatientSplitPartitionV4,
    Stage2PatientMappingV4,
)
from src.stage2_research.e07r_integrity import (
    E07RIntegrityError,
    _verify_partition_against_rows,
)
from src.stage2_research.patient_disjoint import (
    MITDB_201_202_PATIENT_ID,
    SVDB_CONSERVATIVE_PATIENT_ID,
    build_stage2_patient_mapping,
    generate_patient_disjoint_splits,
    publish_patient_disjoint_bundle,
)

SOURCE_HASH = "a" * 64
IDENTITY_HASH = "b" * 64
PHYSIONET_URL = "https://physionet.org/physiobank/database/html/mitdbdir/intro.htm"


def _fixture() -> tuple[pd.DataFrame, np.ndarray, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    upstream: list[dict[str, object]] = []

    mitdb_records = ["100", "101", "102", "103", "104", "105", "106", "201", "202"]
    for record in mitdb_records:
        patient = (
            MITDB_201_202_PATIENT_ID if record in {"201", "202"} else f"mitdb:subject:{record}"
        )
        upstream.append(
            {
                "dataset_id": "mitdb",
                "record_id": record,
                "patient_group_id": patient,
                "identity_status": "IDENTITY_VERIFIED",
                "evidence_ref": "official MIT-BIH subject mapping",
            }
        )
        for beat, label in enumerate((0, 0, 1, 1, 2, 2)):
            rows.append(
                {
                    "dataset": "mitdb",
                    "record_id": record,
                    "beat_idx": beat,
                    "label_aami": ("S", "V", "F")[label],
                }
            )

    for patient_number in range(1, 9):
        for suffix in (0, 1):
            record = f"I{patient_number:02d}{suffix}"
            upstream.append(
                {
                    "dataset_id": "incart",
                    "record_id": record,
                    "patient_group_id": f"incart:patient:{patient_number}",
                    "identity_status": "IDENTITY_VERIFIED",
                    "evidence_ref": "authenticated WFDB patient header",
                }
            )
            for beat, label in enumerate((0, 1, 1, 2)):
                rows.append(
                    {
                        "dataset": "incart",
                        "record_id": record,
                        "beat_idx": beat,
                        "label_aami": ("S", "V", "F")[label],
                    }
                )

    for record in ("800", "801", "802", "803"):
        upstream.append(
            {
                "dataset_id": "svdb",
                "record_id": record,
                "patient_group_id": None,
                "identity_status": "IDENTITY_UNVERIFIED",
                "evidence_ref": "patient linkage unavailable",
            }
        )
        for beat, label in enumerate((0, 0, 1, 1, 2)):
            rows.append(
                {
                    "dataset": "svdb",
                    "record_id": record,
                    "beat_idx": beat,
                    "label_aami": ("S", "V", "F")[label],
                }
            )

    frame = pd.DataFrame(rows)
    labels = np.asarray(
        [{"S": 0, "V": 1, "F": 2}[str(value)] for value in frame["label_aami"]],
        dtype=np.int64,
    )
    return frame, labels, upstream


def _mapping() -> tuple[pd.DataFrame, np.ndarray, Stage2PatientMappingV4]:
    frame, labels, upstream = _fixture()
    mapping = build_stage2_patient_mapping(
        frame,
        upstream,
        stage2_parquet_sha256=SOURCE_HASH,
        upstream_identity_sha256=IDENTITY_HASH,
        mitdb_source_url=PHYSIONET_URL,
    )
    return frame, labels, mapping


def _confirmatory_mapping() -> tuple[pd.DataFrame, np.ndarray, Stage2PatientMappingV4]:
    frame, labels, mapping = _mapping()
    mask = frame["dataset"].astype(str).to_numpy() != "svdb"
    return frame.loc[mask].reset_index(drop=True), labels[mask], mapping


def test_mapping_groups_mitdb_201_202_and_conservatively_groups_svdb() -> None:
    frame, _labels, mapping = _mapping()
    by_key = {(item.dataset, item.record_id): item for item in mapping.records}

    assert by_key[("mitdb", "201")].patient_id == MITDB_201_202_PATIENT_ID
    assert by_key[("mitdb", "202")].patient_id == MITDB_201_202_PATIENT_ID
    assert {by_key[("svdb", record)].patient_id for record in ("800", "801", "802", "803")} == {
        None
    }
    assert {
        by_key[("svdb", record)].partition_barrier_id for record in ("800", "801", "802", "803")
    } == {SVDB_CONSERVATIVE_PATIENT_ID}
    assert mapping.record_count == frame.loc[:, ["dataset", "record_id"]].drop_duplicates().shape[0]
    assert mapping.mapping_hash


def test_mapping_rejects_inconsistent_official_201_202_identity() -> None:
    frame, _labels, upstream = _fixture()
    for row in upstream:
        if row["dataset_id"] == "mitdb" and row["record_id"] == "202":
            row["patient_group_id"] = "mitdb:subject:202"

    with pytest.raises(ValueError, match="201/202"):
        build_stage2_patient_mapping(
            frame,
            upstream,
            stage2_parquet_sha256=SOURCE_HASH,
            upstream_identity_sha256=IDENTITY_HASH,
            mitdb_source_url=PHYSIONET_URL,
        )


def test_outer_and_all_inner_folds_are_patient_and_record_disjoint() -> None:
    frame, labels, mapping = _confirmatory_mapping()
    contract = E06EvaluationContract(n_splits=5, inner_splits=4, random_seed=42)

    generated_a = generate_patient_disjoint_splits(
        frame, labels, mapping, contract, source_manifest_hash="c" * 64
    )
    generated_b = generate_patient_disjoint_splits(
        frame, labels, mapping, contract, source_manifest_hash="c" * 64
    )

    assert generated_a.split_manifest.manifest_hash == generated_b.split_manifest.manifest_hash
    assert generated_a.leakage_report.status == "PASS"
    assert generated_a.leakage_report.outer_folds_checked == 5
    assert generated_a.leakage_report.inner_folds_checked == 20
    assert generated_a.leakage_report.known_group_201_202_respected
    assert len(generated_a.outer_folds.folds) == 5
    assert len(generated_a.inner_folds.folds) == 20

    for outer_fold in generated_a.outer_folds.folds:
        train_patients = set(outer_fold.train.patient_ids)
        test_patients = set(outer_fold.test.patient_ids)
        assert train_patients.isdisjoint(test_patients)
        assert set(outer_fold.train.record_ids).isdisjoint(outer_fold.test.record_ids)
        assert ("201" in outer_fold.train.record_ids) == ("202" in outer_fold.train.record_ids)
        assert ("201" in outer_fold.test.record_ids) == ("202" in outer_fold.test.record_ids)

    for inner_fold in generated_a.inner_folds.folds:
        train_patients = set(inner_fold.train.patient_ids)
        val_patients = set(inner_fold.validation.patient_ids)
        test_patients = set(inner_fold.outer_test_patient_ids)
        assert train_patients.isdisjoint(val_patients)
        assert train_patients.isdisjoint(test_patients)
        assert val_patients.isdisjoint(test_patients)
        assert set(inner_fold.train.record_ids).isdisjoint(inner_fold.validation.record_ids)


def test_preflight_rejects_indices_that_disagree_with_clean_record_lists() -> None:
    frame, labels, mapping = _confirmatory_mapping()
    generated = generate_patient_disjoint_splits(
        frame,
        labels,
        mapping,
        E06EvaluationContract(n_splits=5, inner_splits=4, random_seed=42),
        source_manifest_hash="c" * 64,
    )
    fold = generated.outer_folds.folds[0]
    corrupted_indices = sorted(
        (set(fold.train.indices) - {fold.train.indices[0]}) | {fold.test.indices[0]}
    )
    corrupted = PatientSplitPartitionV4.model_validate(
        {
            **fold.train.model_dump(mode="json"),
            "indices": corrupted_indices,
        }
    )

    with pytest.raises(E07RIntegrityError, match="row binding mismatch"):
        _verify_partition_against_rows(
            corrupted,
            frame,
            labels,
            mapping,
            context="corrupted outer train",
        )


def test_patient_disjoint_bundle_publication_is_write_once(tmp_path: Path) -> None:
    frame, labels, mapping = _confirmatory_mapping()
    contract = E06EvaluationContract(n_splits=5, inner_splits=4, random_seed=42)
    generated = generate_patient_disjoint_splits(
        frame, labels, mapping, contract, source_manifest_hash="c" * 64
    )
    target = tmp_path / "v4.0"

    hashes = publish_patient_disjoint_bundle(target, generated)

    assert hashes
    assert (target / "outer_folds.json").is_file()
    assert (target / "inner_folds.json").is_file()
    assert (target / "split_manifest.json").is_file()
    assert (target / "patient_groups.json").is_file()
    assert (target / "leakage_checks.json").is_file()
    assert (target / "fold_statistics.json").is_file()
    assert (target / "outer_splits_stage2.json").is_file()
    assert (target / "inner_splits_stage2.json").is_file()

    with pytest.raises(FileExistsError, match="immutable publication"):
        publish_patient_disjoint_bundle(target, generated)
