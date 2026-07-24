"""Deterministic patient-aware outer split construction and publication."""

from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .contracts import (
    DatasetRole,
    FoldAssignment,
    PatientIdentityManifest,
    PatientSplitManifest,
    SplitBundlePublication,
)
from .integrity import hash_canonical, write_detached_sha256, write_json_exclusive


def _safe_count(value: Any, *, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid count for {context}: {value!r}") from error


def validate_split_identity_consistency(
    identity: PatientIdentityManifest,
    split: PatientSplitManifest,
) -> None:
    """Reject any split that is not a complete partition of its identity manifest."""
    if split.source_data_hash != identity.source_data_hash:
        raise ValueError("split and identity source hashes differ")
    if split.patient_identity_hash != hash_canonical("patient-identity", identity):
        raise ValueError("split does not bind the supplied patient identity manifest")
    core_records = {
        record.record_key: record
        for record in identity.records
        if record.role is DatasetRole.CONFIRMATORY_CORE
    }
    quarantined = {
        record.record_key
        for record in identity.records
        if record.role is not DatasetRole.CONFIRMATORY_CORE
    }
    fold_by_patient: dict[str, int] = {}
    fold_by_record: dict[str, int] = {}
    for fold in split.folds:
        for patient_id in fold.outer_test_patient_ids:
            fold_by_patient[patient_id] = fold.fold
        for record_key in fold.outer_test_record_keys:
            fold_by_record[record_key] = fold.fold
    expected_patients = {
        record.patient_group_id for record in core_records.values() if record.patient_group_id
    }
    if set(fold_by_patient) != expected_patients:
        raise ValueError("split patient population differs from confirmatory identity")
    if set(fold_by_record) != set(core_records):
        raise ValueError("split record population differs from confirmatory identity")
    for record_key, record in core_records.items():
        if record.patient_group_id is None or (
            fold_by_record[record_key] != fold_by_patient[record.patient_group_id]
        ):
            raise ValueError(f"patient/record fold mismatch: {record_key}")
    if set(split.quarantined_records) != quarantined:
        raise ValueError("split quarantine differs from non-confirmatory identity records")
    expected_core_datasets = {
        record.dataset_id
        for record in identity.records
        if record.role is DatasetRole.CONFIRMATORY_CORE
    }
    expected_quarantine_datasets = {
        record.dataset_id
        for record in identity.records
        if record.role is not DatasetRole.CONFIRMATORY_CORE
    }
    if set(split.core_dataset_ids) != expected_core_datasets:
        raise ValueError("split core datasets differ from identity roles")
    if set(split.quarantine_dataset_ids) != expected_quarantine_datasets:
        raise ValueError("split quarantine datasets differ from identity roles")


def build_patient_split_manifest(
    data: pd.DataFrame,
    identity: PatientIdentityManifest,
    *,
    split_version: str,
    n_splits: int,
    random_state: int,
) -> PatientSplitManifest:
    """Assign confirmatory patients once and quarantine unresolved datasets."""
    if split_version != "3.1.0":
        raise ValueError("split_version must be 3.1.0")
    if n_splits != 5:
        raise ValueError("the advanced protocol requires exactly five outer folds")
    required = {"dataset", "record_id", "label_aami"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"split input missing columns: {sorted(missing)}")

    by_key = {record.record_key: record for record in identity.records}
    frame = data.loc[:, ["dataset", "record_id", "label_aami"]].astype(str).copy()
    frame["record_key"] = frame["dataset"] + "/" + frame["record_id"]
    missing_identity = sorted(set(frame["record_key"]) - set(by_key))
    if missing_identity:
        raise ValueError(f"missing patient identity for records: {missing_identity}")

    frame["role"] = frame["record_key"].map(lambda key: by_key[key].role.value)
    frame["patient_group_id"] = frame["record_key"].map(lambda key: by_key[key].patient_group_id)
    core = frame[frame["role"] == DatasetRole.CONFIRMATORY_CORE.value].copy()
    if core.empty:
        raise ValueError("confirmatory core is empty")
    if core["patient_group_id"].isna().any():
        raise ValueError("confirmatory row has unresolved patient identity")

    labels = core["label_aami"].to_numpy()
    groups = core["patient_group_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    folds: list[FoldAssignment] = []
    for fold, (_, test_index) in enumerate(
        splitter.split(np.zeros(len(core), dtype=np.int8), labels, groups)
    ):
        test = core.iloc[test_index]
        patient_ids = tuple(sorted(test["patient_group_id"].astype(str).unique().tolist()))
        record_keys = tuple(sorted(test["record_key"].unique().tolist()))
        class_counts = {
            str(label): _safe_count(count, context=f"class {label}")
            for label, count in sorted(Counter(test["label_aami"]).items())
        }
        dataset_counts = {
            str(dataset): _safe_count(count, context=f"dataset {dataset}")
            for dataset, count in sorted(Counter(test["dataset"]).items())
        }
        folds.append(
            FoldAssignment(
                fold=fold,
                outer_test_patient_ids=patient_ids,
                outer_test_record_keys=record_keys,
                n_samples=len(test),
                class_counts=class_counts,
                dataset_counts=dataset_counts,
            )
        )

    core_dataset_ids = tuple(
        sorted(
            {
                record.dataset_id
                for record in identity.records
                if record.role is DatasetRole.CONFIRMATORY_CORE
            }
        )
    )
    quarantine_dataset_ids = tuple(
        sorted(
            {
                record.dataset_id
                for record in identity.records
                if record.role is not DatasetRole.CONFIRMATORY_CORE
            }
        )
    )
    quarantined_records = tuple(
        sorted(
            record.record_key
            for record in identity.records
            if record.role is not DatasetRole.CONFIRMATORY_CORE
        )
    )
    split = PatientSplitManifest(
        schema_version="patient-split-v3.1.0",
        split_version="3.1.0",
        n_splits=5,
        random_state=random_state,
        source_data_hash=identity.source_data_hash,
        patient_identity_hash=hash_canonical("patient-identity", identity),
        core_dataset_ids=core_dataset_ids,
        quarantine_dataset_ids=quarantine_dataset_ids,
        folds=tuple(folds),
        quarantined_records=quarantined_records,
    )
    validate_split_identity_consistency(identity, split)
    return split


def publish_split_bundle(
    target: Path,
    *,
    identity: PatientIdentityManifest,
    split: PatientSplitManifest,
) -> None:
    """Publish a validated write-once split bundle with a final commit marker."""
    validate_split_identity_consistency(identity, split)
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / f".{target.name}.publish.lock"
    lock_fd: int | None = None
    lock_owned = False
    created_target = False
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        lock_owned = True
        os.close(lock_fd)
        lock_fd = None
        target.mkdir(parents=False, exist_ok=False)
        created_target = True
        identity_path = target / "patient_identity_manifest.json"
        split_path = target / "index.json"
        write_json_exclusive(identity_path, identity)
        write_detached_sha256(identity_path)
        write_json_exclusive(split_path, split)
        write_detached_sha256(split_path)
        for fold in split.folds:
            fold_path = target / f"fold_{fold.fold}.json"
            write_json_exclusive(fold_path, fold)
            write_detached_sha256(fold_path)
        completion_path = target / "SPLIT_BUNDLE_COMPLETE.json"
        completion = SplitBundlePublication(
            schema_version="patient-split-publication-v3.1.0",
            patient_identity_hash=hash_canonical("patient-identity", identity),
            patient_split_hash=hash_canonical("patient-split", split),
            status="SPLIT_BUNDLE_COMPLETE",
        )
        write_json_exclusive(completion_path, completion)
        write_detached_sha256(completion_path)
    except Exception:
        if created_target:
            shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_owned:
            lock.unlink(missing_ok=True)
