"""Evidence-based patient identity mapping without demographic persistence."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .contracts import (
    DatasetIdentityPolicy,
    DatasetRole,
    IdentityMethod,
    IdentityStatus,
    LegacyLeakageAudit,
    PatientIdentityManifest,
    PatientIdentityPolicy,
    PatientIdentityRecord,
)
from .integrity import resolve_project_path

_INCART_PATIENT_RE = re.compile(r"patient\s+([1-9]|[12][0-9]|3[0-2])", re.IGNORECASE)


def _header_comments(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return tuple(
        line[1:].strip()
        for line in path.read_text(encoding="utf-8", errors="strict").splitlines()
        if line.startswith("#")
    )


def _incart_patient_id(header_path: Path) -> str:
    matches = [
        match.group(1)
        for comment in _header_comments(header_path)
        if (match := _INCART_PATIENT_RE.fullmatch(comment)) is not None
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{header_path}: expected exactly one INCART patient line, found {len(matches)}"
        )
    return f"incart:patient:{matches[0]}"


def _mitdb_patient_id(record_id: str, policy: DatasetIdentityPolicy) -> str:
    for group in policy.same_patient_record_groups:
        if record_id in group:
            canonical = "-".join(sorted(group))
            return f"mitdb:subject:{canonical}"
    return f"mitdb:subject:{record_id}"


def _record_from_policy(
    *,
    project_root: Path,
    dataset_policy: DatasetIdentityPolicy,
    record_id: str,
) -> PatientIdentityRecord:
    raw_dir = resolve_project_path(project_root, dataset_policy.raw_dir)
    header = raw_dir / f"{record_id}.hea"
    if not header.is_file():
        raise FileNotFoundError(header)

    if dataset_policy.method is IdentityMethod.INCART_HEADER_PATIENT:
        patient_id = _incart_patient_id(header)
        return PatientIdentityRecord(
            dataset_id=dataset_policy.dataset_id,
            record_id=record_id,
            patient_id=patient_id,
            patient_group_id=patient_id,
            role=dataset_policy.role,
            identity_status=IdentityStatus.IDENTITY_VERIFIED,
            evidence_ref=dataset_policy.evidence_ref,
        )
    if dataset_policy.method is IdentityMethod.MITDB_DOCUMENTED_SUBJECT:
        patient_id = _mitdb_patient_id(record_id, dataset_policy)
        return PatientIdentityRecord(
            dataset_id=dataset_policy.dataset_id,
            record_id=record_id,
            patient_id=patient_id,
            patient_group_id=patient_id,
            role=dataset_policy.role,
            identity_status=IdentityStatus.IDENTITY_VERIFIED,
            evidence_ref=dataset_policy.evidence_ref,
        )
    if dataset_policy.method is IdentityMethod.UNRESOLVED:
        return PatientIdentityRecord(
            dataset_id=dataset_policy.dataset_id,
            record_id=record_id,
            patient_id=None,
            patient_group_id=None,
            role=dataset_policy.role,
            identity_status=IdentityStatus.IDENTITY_UNVERIFIED,
            evidence_ref=dataset_policy.evidence_ref,
        )
    raise ValueError(f"unsupported identity method: {dataset_policy.method}")


def build_patient_identity_manifest(
    records_frame: pd.DataFrame,
    *,
    project_root: Path,
    policy: PatientIdentityPolicy,
    source_data_hash: str,
) -> PatientIdentityManifest:
    """Build a complete record mapping and enforce policy counts."""
    required = {"dataset", "record_id"}
    missing = required - set(records_frame.columns)
    if missing:
        raise ValueError(f"identity input missing columns: {sorted(missing)}")

    distinct = (
        records_frame.loc[:, ["dataset", "record_id"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["dataset", "record_id"], kind="stable")
    )
    policies = {dataset.dataset_id: dataset for dataset in policy.datasets}
    observed_datasets = set(distinct["dataset"])
    unknown = observed_datasets - set(policies)
    if unknown:
        raise ValueError(f"datasets without identity policy: {sorted(unknown)}")
    absent = set(policies) - observed_datasets
    if absent:
        raise ValueError(f"identity policy datasets absent from inputs: {sorted(absent)}")

    records: list[PatientIdentityRecord] = []
    for dataset_id, dataset_rows in distinct.groupby("dataset", sort=True):
        dataset_policy = policies[dataset_id]
        record_ids = tuple(sorted(dataset_rows["record_id"].tolist()))
        if len(record_ids) != dataset_policy.expected_records:
            raise ValueError(
                f"{dataset_id}: expected {dataset_policy.expected_records} records, "
                f"observed {len(record_ids)}"
            )
        dataset_records = [
            _record_from_policy(
                project_root=project_root,
                dataset_policy=dataset_policy,
                record_id=record_id,
            )
            for record_id in record_ids
        ]
        verified_patients = {
            record.patient_group_id
            for record in dataset_records
            if record.identity_status is IdentityStatus.IDENTITY_VERIFIED
        }
        if (
            dataset_policy.expected_patients is not None
            and len(verified_patients) != dataset_policy.expected_patients
        ):
            raise ValueError(
                f"{dataset_id}: expected {dataset_policy.expected_patients} patients, "
                f"observed {len(verified_patients)}"
            )
        records.extend(dataset_records)

    core = [record for record in records if record.role is DatasetRole.CONFIRMATORY_CORE]
    quarantine = [record for record in records if record.role is not DatasetRole.CONFIRMATORY_CORE]
    return PatientIdentityManifest(
        schema_version="patient-identity-v3.1.0",
        source_data_hash=source_data_hash,
        records=tuple(records),
        confirmatory_patient_count=len({record.patient_group_id for record in core}),
        confirmatory_record_count=len(core),
        quarantined_record_count=len(quarantine),
    )


def audit_legacy_outer_fold_leakage(
    identity: PatientIdentityManifest,
    *,
    split_dir: Path,
) -> LegacyLeakageAudit:
    """Measure known-patient overlap in legacy record-grouped outer folds."""
    by_record: dict[str, PatientIdentityRecord] = {}
    for record in identity.records:
        if record.identity_status is not IdentityStatus.IDENTITY_VERIFIED:
            continue
        if record.record_id in by_record:
            raise ValueError(f"legacy record ID is ambiguous across datasets: {record.record_id}")
        by_record[record.record_id] = record

    patient_folds: dict[str, set[int]] = defaultdict(set)
    paths = sorted(split_dir.glob("fold_*.json"))
    if not paths:
        raise FileNotFoundError(f"no legacy fold manifests in {split_dir}")
    for fallback_fold, path in enumerate(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            fold = int(payload.get("fold", fallback_fold))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid legacy fold manifest: {path}") from error
        record_ids = payload.get("val_record_ids", payload.get("val_patient_ids"))
        if not isinstance(record_ids, list):
            raise ValueError(f"{path}: missing val_record_ids")
        for record_id in map(str, record_ids):
            identity_record = by_record.get(record_id)
            if identity_record is not None and identity_record.patient_group_id is not None:
                patient_folds[identity_record.patient_group_id].add(fold)

    crossing = {
        patient_id: tuple(sorted(folds))
        for patient_id, folds in sorted(patient_folds.items())
        if len(folds) > 1
    }
    return LegacyLeakageAudit(
        checked_patient_count=len(patient_folds),
        cross_fold_patient_count=len(crossing),
        cross_fold_patients=crossing,
    )
