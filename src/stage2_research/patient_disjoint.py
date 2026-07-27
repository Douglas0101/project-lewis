"""E07R patient identity, patient-disjoint splits, and immutable publication."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.models.e06_protocol import E06EvaluationContract, build_outer_splits
from src.stage2_research.contracts import (
    InnerFoldManifest,
    InnerSplitManifest,
    OuterFoldManifest,
    SplitManifest,
    SplitPartition,
)
from src.stage2_research.e07r_contracts import (
    FoldStatisticV4,
    FoldStatisticsV4,
    MitdbPatientMappingDocument,
    PatientDisjointSplitManifestV4,
    PatientInnerFoldV4,
    PatientInnerFoldsV4,
    PatientOuterFoldV4,
    PatientOuterFoldsV4,
    PatientSplitPartitionV4,
    SplitLeakageReportV4,
    Stage2PatientMappingV4,
    Stage2PatientRecordV4,
)
from src.stage2_research.integrity import hash_canonical, sha256_file
from src.training_integrity.integrity import exclusive_publication, write_json_exclusive

MITDB_201_202_PATIENT_ID = "mitdb:subject:201_202"
SVDB_CONSERVATIVE_PATIENT_ID = "svdb:conservative-unverified-cohort"
SPLIT_VERSION = "v4.0-patient-disjoint"
LABEL_TO_INDEX = {"S": 0, "V": 1, "F": 2}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}


@dataclass(frozen=True)
class GeneratedPatientDisjointSplits:
    """All persisted and Stage 2 adapter artifacts for one split generation."""

    patient_mapping: Stage2PatientMappingV4
    outer_folds: PatientOuterFoldsV4
    inner_folds: PatientInnerFoldsV4
    split_manifest: PatientDisjointSplitManifestV4
    leakage_report: SplitLeakageReportV4
    fold_statistics: FoldStatisticsV4
    stage2_outer: SplitManifest
    stage2_inner: InnerSplitManifest


def _safe_int(value: Any, context: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{context} is not integral") from error
    if integer != value:
        raise ValueError(f"{context} is not integral")
    return integer


def _distinct_record_keys(frame: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    missing = {"dataset", "record_id"} - set(frame.columns)
    if missing:
        raise ValueError(f"Stage 2 identity frame missing columns: {sorted(missing)}")
    distinct = (
        frame.loc[:, ["dataset", "record_id"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["dataset", "record_id"], kind="stable")
    )
    return tuple((row.dataset, row.record_id) for row in distinct.itertuples(index=False))


def _upstream_by_key(
    upstream_records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in upstream_records:
        key = (str(record.get("dataset_id", "")), str(record.get("record_id", "")))
        if not all(key):
            raise ValueError("upstream identity record lacks dataset_id or record_id")
        if key in result:
            raise ValueError(f"duplicate upstream identity record: {key}")
        result[key] = record
    return result


def build_stage2_patient_mapping(
    frame: pd.DataFrame,
    upstream_records: Sequence[Mapping[str, Any]],
    *,
    stage2_parquet_sha256: str,
    upstream_identity_sha256: str,
    mitdb_source_url: str,
) -> Stage2PatientMappingV4:
    """Build complete Stage 2 identity without trusting record_id as patient_id."""
    keys = _distinct_record_keys(frame)
    datasets = {dataset for dataset, _ in keys}
    allowed_datasets = {"incart", "mitdb", "svdb"}
    if not {"incart", "mitdb"}.issubset(datasets) or not datasets.issubset(allowed_datasets):
        raise ValueError(f"unexpected Stage 2 datasets: {sorted(datasets)}")
    upstream = _upstream_by_key(upstream_records)
    missing_upstream = [key for key in keys if key not in upstream]
    if missing_upstream:
        raise ValueError(f"records absent from upstream identity: {missing_upstream[:5]}")

    special_keys = (("mitdb", "201"), ("mitdb", "202"))
    if all(key in keys for key in special_keys):
        upstream_special = [upstream[key].get("patient_group_id") for key in special_keys]
        if not upstream_special[0] or upstream_special[0] != upstream_special[1]:
            raise ValueError("official MIT-BIH 201/202 identity is inconsistent")

    records: list[Stage2PatientRecordV4] = []
    for dataset, record_id in keys:
        source = upstream[(dataset, record_id)]
        status = str(source.get("identity_status", ""))
        patient_group = source.get("patient_group_id")
        evidence = str(source.get("evidence_ref", "")).strip()
        if dataset in {"incart", "mitdb"}:
            if status != "IDENTITY_VERIFIED" or not patient_group:
                raise ValueError(f"{dataset}/{record_id} lacks verified patient identity")
            if dataset == "mitdb" and record_id in {"201", "202"}:
                patient_id: str | None = MITDB_201_202_PATIENT_ID
                partition_barrier_id = MITDB_201_202_PATIENT_ID
                identity_status = "OFFICIAL"
                evidence_ref = f"{mitdb_source_url} — records 201 and 202 are one subject"
            else:
                patient_id = str(patient_group)
                partition_barrier_id = patient_id
                identity_status = "IDENTITY_VERIFIED"
                evidence_ref = evidence
        else:
            if status != "IDENTITY_UNVERIFIED":
                raise ValueError("SVDB identity must remain explicitly unverified")
            patient_id = None
            partition_barrier_id = SVDB_CONSERVATIVE_PATIENT_ID
            identity_status = "IDENTITY_UNVERIFIED_GROUPED_CONSERVATIVELY"
            evidence_ref = (
                "No authenticated SVDB record-to-patient mapping; all records remain "
                "in one conservative split group to prevent possible leakage."
            )
        records.append(
            Stage2PatientRecordV4.model_validate(
                {
                    "dataset": dataset,
                    "record_id": record_id,
                    "patient_id": patient_id,
                    "partition_barrier_id": partition_barrier_id,
                    "identity_status": identity_status,
                    "evidence_ref": evidence_ref,
                }
            )
        )

    dataset_record_counts = {
        dataset: sum(record.dataset == dataset for record in records)
        for dataset in sorted(datasets)
    }
    dataset_verified_patient_counts = {
        dataset: len(
            {
                record.patient_id
                for record in records
                if record.dataset == dataset and record.patient_id is not None
            }
        )
        for dataset in sorted(datasets)
    }
    dataset_partition_barrier_counts = {
        dataset: len(
            {record.partition_barrier_id for record in records if record.dataset == dataset}
        )
        for dataset in sorted(datasets)
    }
    payload = {
        "schema_version": "stage2-patient-identity-v4.0",
        "stage2_parquet_sha256": stage2_parquet_sha256,
        "upstream_identity_sha256": upstream_identity_sha256,
        "mitdb_source_url": mitdb_source_url,
        "mapping_policy": ("official_verified_or_conservative_single_group_for_unverified_dataset"),
        "records": [record.model_dump(mode="json") for record in records],
        "record_count": len(records),
        "verified_patient_count": len(
            {record.patient_id for record in records if record.patient_id is not None}
        ),
        "partition_barrier_count": len({record.partition_barrier_id for record in records}),
        "dataset_record_counts": dataset_record_counts,
        "dataset_verified_patient_counts": dataset_verified_patient_counts,
        "dataset_partition_barrier_counts": dataset_partition_barrier_counts,
    }
    return Stage2PatientMappingV4.model_validate(
        {**payload, "mapping_hash": hash_canonical(payload)}
    )


def build_mitdb_mapping_document(
    mapping: Stage2PatientMappingV4,
) -> MitdbPatientMappingDocument:
    """Build the required official MIT-BIH evidence mapping document."""
    mitdb_records = [record for record in mapping.records if record.dataset == "mitdb"]
    special = sorted(
        record.record_id
        for record in mitdb_records
        if record.patient_id == MITDB_201_202_PATIENT_ID
    )
    if special != ["201", "202"]:
        raise ValueError("MIT-BIH 201/202 official group is incomplete")
    payload = {
        "dataset": "MIT-BIH Arrhythmia Database",
        "source": "PhysioNet official documentation",
        "source_url": mapping.mitdb_source_url,
        "access_date": "2026-07-26",
        "mapping_policy": "official_evidence_required",
        "default_rule": "record_id_equals_patient_id_unless_official_evidence",
        "patient_groups": [
            {
                "patient_id": MITDB_201_202_PATIENT_ID,
                "record_ids": ["201", "202"],
                "evidence": "PhysioNet states records 201 and 202 belong to the same individual",
                "confidence": "OFFICIAL",
            }
        ],
        "offline_provenance": False,
        "cached_source_path": "docs/physionet_mitdb_patient_statement.md",
    }
    return MitdbPatientMappingDocument.model_validate(
        {**payload, "mapping_hash": hash_canonical(payload)}
    )


def patient_groups_for_frame(
    frame: pd.DataFrame,
    mapping: Stage2PatientMappingV4,
) -> np.ndarray:
    """Resolve every Stage 2 row to one complete patient group."""
    by_key = {(record.dataset, record.record_id): record.patient_id for record in mapping.records}
    groups: list[str] = []
    for row in frame.loc[:, ["dataset", "record_id"]].astype(str).itertuples(index=False):
        key = (row.dataset, row.record_id)
        try:
            patient_id = by_key[key]
        except KeyError as error:
            raise ValueError(f"Stage 2 row lacks patient mapping: {key}") from error
        if patient_id is None:
            raise ValueError(
                f"Stage 2 confirmatory split cannot use unverified patient identity: {key}"
            )
        groups.append(patient_id)
    return np.asarray(groups, dtype=str)


def _partition(
    indices: np.ndarray,
    frame: pd.DataFrame,
    labels: np.ndarray,
    patient_groups: np.ndarray,
) -> PatientSplitPartitionV4:
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0) or np.any(values >= len(frame)):
        raise ValueError("split indices are invalid")
    patients = tuple(sorted(set(patient_groups[values].astype(str).tolist())))
    records_array = frame.iloc[values]["record_id"].astype(str).to_numpy()
    records = tuple(sorted(set(records_array.tolist())))
    class_counts: dict[str, int] = dict.fromkeys(("S", "V", "F"), 0)
    unique, counts = np.unique(labels[values], return_counts=True)
    for label, count in zip(unique, counts, strict=True):
        class_counts[INDEX_TO_LABEL[_safe_int(label, "class label")]] = _safe_int(
            count, "class count"
        )
    f_mask = labels[values] == LABEL_TO_INDEX["F"]
    return PatientSplitPartitionV4(
        indices=tuple(_safe_int(item, "split index") for item in values),
        patient_ids=patients,
        record_ids=records,
        class_counts=class_counts,
        f_208=_safe_int(np.sum(f_mask & (records_array == "208")), "F 208 count"),
        f_213=_safe_int(np.sum(f_mask & (records_array == "213")), "F 213 count"),
        f_outside_208_213=_safe_int(
            np.sum(f_mask & ~np.isin(records_array, ["208", "213"])),
            "F outside 208/213 count",
        ),
        n_samples=len(values),
        n_patients=len(patients),
        n_records=len(records),
        indices_hash=hash_canonical(values.tolist()),
        patient_ids_hash=hash_canonical(patients),
        record_ids_hash=hash_canonical(records),
    )


def _stage2_partition(partition: PatientSplitPartitionV4) -> SplitPartition:
    return SplitPartition(
        indices=partition.indices,
        groups=partition.patient_ids,
        class_counts=partition.class_counts,
        f_208=partition.f_208,
        f_213=partition.f_213,
        f_outside_208_213=partition.f_outside_208_213,
        indices_hash=partition.indices_hash,
        groups_hash=partition.patient_ids_hash,
    )


def _outer_models(
    frame: pd.DataFrame,
    labels: np.ndarray,
    patient_groups: np.ndarray,
    mapping: Stage2PatientMappingV4,
    contract: E06EvaluationContract,
) -> tuple[PatientOuterFoldsV4, SplitManifest, list[tuple[np.ndarray, np.ndarray]]]:
    dataset_binding_hash = hash_canonical(
        {
            "stage2_parquet_sha256": mapping.stage2_parquet_sha256,
            "patient_mapping_hash": mapping.mapping_hash,
        }
    )
    splits = build_outer_splits(labels, patient_groups, contract)
    rows: list[PatientOuterFoldV4] = []
    adapters: list[OuterFoldManifest] = []
    for index, (train_indices, test_indices) in enumerate(splits):
        train = _partition(train_indices, frame, labels, patient_groups)
        test = _partition(test_indices, frame, labels, patient_groups)
        patient_overlap = tuple(sorted(set(train.patient_ids) & set(test.patient_ids)))
        record_overlap = tuple(sorted(set(train.record_ids) & set(test.record_ids)))
        if patient_overlap or record_overlap:
            raise ValueError(f"outer fold {index + 1} leaks groups")
        rows.append(
            PatientOuterFoldV4(
                fold=index + 1,
                random_state=contract.random_seed,
                train=train,
                test=test,
                patient_overlap=patient_overlap,
                record_overlap=record_overlap,
            )
        )
        adapters.append(
            OuterFoldManifest(
                fold=index + 1,
                train=_stage2_partition(train),
                test=_stage2_partition(test),
                overlap_groups=patient_overlap,
            )
        )
    outer_payload = {
        "schema_version": "stage2-outer-patient-disjoint-v4.0",
        "split_version": SPLIT_VERSION,
        "dataset_binding_hash": dataset_binding_hash,
        "patient_mapping_hash": mapping.mapping_hash,
        "folds": [row.model_dump(mode="json") for row in rows],
    }
    outer = PatientOuterFoldsV4.model_validate(
        {**outer_payload, "manifest_hash": hash_canonical(outer_payload)}
    )
    adapter_payload = {
        "schema_version": "stage2-splits-v2.4",
        "dataset_manifest_hash": dataset_binding_hash,
        "splitter": "StratifiedGroupKFold",
        "split_random_state": contract.random_seed,
        "outer_folds": [row.model_dump(mode="json") for row in adapters],
    }
    adapter = SplitManifest.model_validate(
        {**adapter_payload, "manifest_hash": hash_canonical(adapter_payload)}
    )
    return outer, adapter, splits


def _inner_models(
    frame: pd.DataFrame,
    labels: np.ndarray,
    patient_groups: np.ndarray,
    mapping: Stage2PatientMappingV4,
    contract: E06EvaluationContract,
    outer: PatientOuterFoldsV4,
    stage2_outer: SplitManifest,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[PatientInnerFoldsV4, InnerSplitManifest]:
    rows: list[PatientInnerFoldV4] = []
    selected_adapters: list[InnerFoldManifest] = []
    placeholder = np.zeros(labels.shape[0], dtype=np.uint8)
    for outer_index, (outer_train, outer_test) in enumerate(outer_splits):
        seed = contract.random_seed + outer_index + 1
        splitter = StratifiedGroupKFold(
            n_splits=contract.inner_splits,
            shuffle=True,
            random_state=seed,
        )
        local_splits = list(
            splitter.split(
                placeholder[outer_train],
                labels[outer_train],
                patient_groups[outer_train],
            )
        )
        if len(local_splits) != contract.inner_splits:
            raise ValueError("inner splitter did not return the required folds")
        outer_test_patients = tuple(sorted(set(patient_groups[outer_test].astype(str).tolist())))
        outer_test_records = tuple(
            sorted(set(frame.iloc[outer_test]["record_id"].astype(str).tolist()))
        )
        for inner_index, (local_train, local_validation) in enumerate(local_splits):
            train_indices = outer_train[np.asarray(local_train, dtype=np.int64)]
            validation_indices = outer_train[np.asarray(local_validation, dtype=np.int64)]
            train = _partition(train_indices, frame, labels, patient_groups)
            validation = _partition(validation_indices, frame, labels, patient_groups)
            train_patients = set(train.patient_ids)
            validation_patients = set(validation.patient_ids)
            test_patients = set(outer_test_patients)
            train_records = set(train.record_ids)
            validation_records = set(validation.record_ids)
            row = PatientInnerFoldV4(
                outer_fold=outer_index + 1,
                inner_fold=inner_index + 1,
                random_state=seed,
                selected_for_training=inner_index == 0,
                train=train,
                validation=validation,
                outer_test_patient_ids=outer_test_patients,
                outer_test_record_ids=outer_test_records,
                train_validation_patient_overlap=tuple(
                    sorted(train_patients & validation_patients)
                ),
                train_outer_test_patient_overlap=tuple(sorted(train_patients & test_patients)),
                validation_outer_test_patient_overlap=tuple(
                    sorted(validation_patients & test_patients)
                ),
                train_validation_record_overlap=tuple(sorted(train_records & validation_records)),
            )
            if any(
                (
                    row.train_validation_patient_overlap,
                    row.train_outer_test_patient_overlap,
                    row.validation_outer_test_patient_overlap,
                    row.train_validation_record_overlap,
                )
            ):
                raise ValueError(f"inner fold {outer_index + 1}/{inner_index + 1} leaks groups")
            rows.append(row)
            if inner_index == 0:
                selected_adapters.append(
                    InnerFoldManifest(
                        fold=outer_index + 1,
                        train=_stage2_partition(train),
                        validation=_stage2_partition(validation),
                        outer_test_groups=outer_test_patients,
                    )
                )
    payload = {
        "schema_version": "stage2-inner-patient-disjoint-v4.0",
        "split_version": SPLIT_VERSION,
        "dataset_binding_hash": outer.dataset_binding_hash,
        "patient_mapping_hash": mapping.mapping_hash,
        "outer_manifest_hash": outer.manifest_hash,
        "folds": [row.model_dump(mode="json") for row in rows],
    }
    inner = PatientInnerFoldsV4.model_validate(
        {**payload, "manifest_hash": hash_canonical(payload)}
    )
    adapter_payload = {
        "schema_version": "stage2-inner-splits-v2.4",
        "dataset_manifest_hash": outer.dataset_binding_hash,
        "outer_split_manifest_hash": stage2_outer.manifest_hash,
        "split_random_state": contract.random_seed,
        "inner_folds": [row.model_dump(mode="json") for row in selected_adapters],
    }
    adapter = InnerSplitManifest.model_validate(
        {**adapter_payload, "manifest_hash": hash_canonical(adapter_payload)}
    )
    return inner, adapter


def _leakage_report(
    outer: PatientOuterFoldsV4,
    inner: PatientInnerFoldsV4,
) -> SplitLeakageReportV4:
    patient_overlap = any(outer_fold.patient_overlap for outer_fold in outer.folds) or any(
        inner_fold.train_validation_patient_overlap
        or inner_fold.train_outer_test_patient_overlap
        or inner_fold.validation_outer_test_patient_overlap
        for inner_fold in inner.folds
    )
    record_overlap = any(outer_fold.record_overlap for outer_fold in outer.folds) or any(
        inner_fold.train_validation_record_overlap for inner_fold in inner.folds
    )
    known_group_respected = True
    structural_zeros: list[str] = []
    low_support: list[str] = []
    for outer_fold in outer.folds:
        for partition_name, partition in (
            ("train", outer_fold.train),
            ("test", outer_fold.test),
        ):
            if any(count == 0 for count in partition.class_counts.values()):
                structural_zeros.append(f"outer_{outer_fold.fold}:{partition_name}")
        if outer_fold.test.class_counts["F"] < 20:
            low_support.append(
                f"outer_{outer_fold.fold}:test:F={outer_fold.test.class_counts['F']}"
            )
        for partition in (outer_fold.train, outer_fold.test):
            if ("201" in partition.record_ids) != ("202" in partition.record_ids):
                known_group_respected = False
    for inner_fold in inner.folds:
        for partition in (inner_fold.train, inner_fold.validation):
            if ("201" in partition.record_ids) != ("202" in partition.record_ids):
                known_group_respected = False
        if ("201" in inner_fold.outer_test_record_ids) != (
            "202" in inner_fold.outer_test_record_ids
        ):
            known_group_respected = False
    payload = {
        "split_version": SPLIT_VERSION,
        "patient_disjoint": not patient_overlap,
        "record_disjoint": not record_overlap,
        "outer_folds_checked": len(outer.folds),
        "inner_folds_checked": len(inner.folds),
        "patient_overlap_found": bool(patient_overlap),
        "record_overlap_found": bool(record_overlap),
        "known_group_201_202_respected": known_group_respected,
        "structural_zero_folds": tuple(structural_zeros),
        "low_support_folds": tuple(low_support),
        "status": (
            "PASS"
            if not patient_overlap and not record_overlap and known_group_respected
            else "FAIL"
        ),
    }
    return SplitLeakageReportV4.model_validate({**payload, "report_hash": hash_canonical(payload)})


def _fold_statistics(
    frame: pd.DataFrame,
    labels: np.ndarray,
    patient_groups: np.ndarray,
    outer: PatientOuterFoldsV4,
) -> FoldStatisticsV4:
    rows: list[FoldStatisticV4] = []
    for fold in outer.folds:
        test_indices = np.asarray(fold.test.indices, dtype=np.int64)
        test_f_groups = set(patient_groups[test_indices][labels[test_indices] == 2])
        rows.append(
            FoldStatisticV4(
                fold=fold.fold,
                train_samples=fold.train.n_samples,
                test_samples=fold.test.n_samples,
                train_patients=fold.train.n_patients,
                test_patients=fold.test.n_patients,
                train_records=fold.train.n_records,
                test_records=fold.test.n_records,
                train_class_counts=fold.train.class_counts,
                test_class_counts=fold.test.class_counts,
                test_f_patients=len(test_f_groups),
                contains_svdb_conservative_group_in_test=(
                    SVDB_CONSERVATIVE_PATIENT_ID in fold.test.patient_ids
                ),
            )
        )
    payload = {
        "schema_version": "stage2-fold-statistics-v4.0",
        "rows": [row.model_dump(mode="json") for row in rows],
    }
    return FoldStatisticsV4.model_validate({**payload, "statistics_hash": hash_canonical(payload)})


def generate_patient_disjoint_splits(
    frame: pd.DataFrame,
    labels: np.ndarray,
    mapping: Stage2PatientMappingV4,
    contract: E06EvaluationContract,
    *,
    source_manifest_hash: str,
) -> GeneratedPatientDisjointSplits:
    """Generate all outer/inner splits and prove patient disjointness."""
    label_values = np.asarray(labels, dtype=np.int64)
    if label_values.ndim != 1 or label_values.size != len(frame):
        raise ValueError("labels must align with the Stage 2 frame")
    if set(np.unique(label_values).tolist()) != {0, 1, 2}:
        raise ValueError("Stage 2 labels must be exactly S/V/F")
    patient_groups = patient_groups_for_frame(frame, mapping)
    outer, stage2_outer, outer_splits = _outer_models(
        frame,
        label_values,
        patient_groups,
        mapping,
        contract,
    )
    inner, stage2_inner = _inner_models(
        frame,
        label_values,
        patient_groups,
        mapping,
        contract,
        outer,
        stage2_outer,
        outer_splits,
    )
    leakage = _leakage_report(outer, inner)
    if leakage.status != "PASS":
        raise ValueError("patient-disjoint leakage report failed")
    statistics = _fold_statistics(
        frame,
        label_values,
        patient_groups,
        outer,
    )
    manifest_payload = {
        "schema_version": "stage2-patient-disjoint-split-bundle-v4.0",
        "split_version": SPLIT_VERSION,
        "algorithm": "StratifiedGroupKFold",
        "outer_folds": contract.n_splits,
        "inner_folds_per_outer": contract.inner_splits,
        "random_state": contract.random_seed,
        "source_manifest_hash": source_manifest_hash,
        "dataset_binding_hash": outer.dataset_binding_hash,
        "patient_mapping_hash": mapping.mapping_hash,
        "outer_manifest_hash": outer.manifest_hash,
        "inner_manifest_hash": inner.manifest_hash,
        "leakage_report_hash": leakage.report_hash,
        "fold_statistics_hash": statistics.statistics_hash,
        "stage2_outer_adapter_hash": stage2_outer.manifest_hash,
        "stage2_inner_adapter_hash": stage2_inner.manifest_hash,
    }
    split_manifest = PatientDisjointSplitManifestV4.model_validate(
        {**manifest_payload, "manifest_hash": hash_canonical(manifest_payload)}
    )
    return GeneratedPatientDisjointSplits(
        patient_mapping=mapping,
        outer_folds=outer,
        inner_folds=inner,
        split_manifest=split_manifest,
        leakage_report=leakage,
        fold_statistics=statistics,
        stage2_outer=stage2_outer,
        stage2_inner=stage2_inner,
    )


def publish_patient_disjoint_bundle(
    target: Path,
    generated: GeneratedPatientDisjointSplits,
) -> dict[str, str]:
    """Publish the complete split bundle once through an atomic directory rename."""
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.publish.lock"
    staging = target.parent / f".{target.name}.{os.getpid()}.staging"
    documents = {
        "outer_folds.json": generated.outer_folds.model_dump(mode="json"),
        "inner_folds.json": generated.inner_folds.model_dump(mode="json"),
        "split_manifest.json": generated.split_manifest.model_dump(mode="json"),
        "patient_groups.json": generated.patient_mapping.model_dump(mode="json"),
        "leakage_checks.json": generated.leakage_report.model_dump(mode="json"),
        "fold_statistics.json": generated.fold_statistics.model_dump(mode="json"),
        "outer_splits_stage2.json": generated.stage2_outer.model_dump(mode="json"),
        "inner_splits_stage2.json": generated.stage2_inner.model_dump(mode="json"),
    }
    try:
        with exclusive_publication(lock_path, [target]):
            staging.mkdir(mode=0o700)
            for name, value in documents.items():
                write_json_exclusive(staging / name, value)
            staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {name: sha256_file(target / name) for name in sorted(documents)}
