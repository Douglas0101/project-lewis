from pathlib import Path

import pandas as pd
import pytest

from src.training_integrity.contracts import (
    DatasetRole,
    IdentityStatus,
    PatientIdentityManifest,
    PatientIdentityRecord,
)
from src.training_integrity.integrity import hash_canonical
from src.training_integrity.splits import (
    build_patient_split_manifest,
    publish_split_bundle,
)

SOURCE_HASH = "2" * 64


def _identity_manifest() -> PatientIdentityManifest:
    records: list[PatientIdentityRecord] = []
    for patient in range(10):
        dataset = "mitdb" if patient < 5 else "incart"
        for record_suffix in range(2):
            records.append(
                PatientIdentityRecord(
                    dataset_id=dataset,
                    record_id=f"r{patient:02d}_{record_suffix}",
                    patient_id=f"{dataset}:patient:{patient:02d}",
                    patient_group_id=f"{dataset}:patient:{patient:02d}",
                    role=DatasetRole.CONFIRMATORY_CORE,
                    identity_status=IdentityStatus.IDENTITY_VERIFIED,
                    evidence_ref="synthetic evidence",
                )
            )
    records.append(
        PatientIdentityRecord(
            dataset_id="svdb",
            record_id="800",
            patient_id=None,
            patient_group_id=None,
            role=DatasetRole.DOMAIN_SENSITIVITY,
            identity_status=IdentityStatus.IDENTITY_UNVERIFIED,
            evidence_ref="synthetic unresolved evidence",
        )
    )
    return PatientIdentityManifest(
        schema_version="patient-identity-v3.1.0",
        source_data_hash=SOURCE_HASH,
        records=tuple(records),
        confirmatory_patient_count=10,
        confirmatory_record_count=20,
        quarantined_record_count=1,
    )


def _data() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for patient in range(10):
        dataset = "mitdb" if patient < 5 else "incart"
        for record_suffix in range(2):
            record = f"r{patient:02d}_{record_suffix}"
            for label in ("N", "S", "V", "F", "Q"):
                rows.extend(
                    {"dataset": dataset, "record_id": record, "label_aami": label}
                    for _ in range(patient + 1)
                )
    rows.extend({"dataset": "svdb", "record_id": "800", "label_aami": "S"} for _ in range(20))
    return pd.DataFrame(rows)


def test_patient_split_is_deterministic_disjoint_and_quarantines_svdb() -> None:
    identity = _identity_manifest()
    first = build_patient_split_manifest(
        _data(),
        identity,
        split_version="3.1.0",
        n_splits=5,
        random_state=42,
    )
    second = build_patient_split_manifest(
        _data(),
        identity,
        split_version="3.1.0",
        n_splits=5,
        random_state=42,
    )

    assert hash_canonical("patient-split", first) == hash_canonical("patient-split", second)
    assert {fold.fold for fold in first.folds} == set(range(5))
    assert first.quarantined_records == ("svdb/800",)

    seen_patients: set[str] = set()
    seen_records: set[str] = set()
    for fold in first.folds:
        assert not (seen_patients & set(fold.outer_test_patient_ids))
        assert not (seen_records & set(fold.outer_test_record_keys))
        seen_patients.update(fold.outer_test_patient_ids)
        seen_records.update(fold.outer_test_record_keys)
        assert "svdb/800" not in fold.outer_test_record_keys
    assert len(seen_patients) == 10
    assert len(seen_records) == 20


def test_split_rejects_unmapped_core_record() -> None:
    data = _data()
    data.loc[len(data)] = {
        "dataset": "mitdb",
        "record_id": "unmapped",
        "label_aami": "N",
    }
    with pytest.raises(ValueError, match="missing patient identity"):
        build_patient_split_manifest(
            data,
            _identity_manifest(),
            split_version="3.1.0",
            n_splits=5,
            random_state=42,
        )


def test_losing_split_publisher_never_removes_foreign_lock(tmp_path: Path) -> None:
    identity = _identity_manifest()
    split = build_patient_split_manifest(
        _data(),
        identity,
        split_version="3.1.0",
        n_splits=5,
        random_state=42,
    )
    target = tmp_path / "v3.1.0"
    lock = tmp_path / ".v3.1.0.publish.lock"
    lock.write_text("owned by another publisher\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        publish_split_bundle(target, identity=identity, split=split)

    assert lock.read_text(encoding="utf-8") == "owned by another publisher\n"
    assert not target.exists()


def test_split_publisher_rejects_identity_inconsistent_manifest(tmp_path: Path) -> None:
    identity = _identity_manifest()
    split = build_patient_split_manifest(
        _data(),
        identity,
        split_version="3.1.0",
        n_splits=5,
        random_state=42,
    )
    first_fold = split.folds[0]
    inconsistent_fold = first_fold.model_copy(
        update={"outer_test_record_keys": first_fold.outer_test_record_keys[1:]}
    )
    inconsistent = split.model_copy(update={"folds": (inconsistent_fold, *split.folds[1:])})
    target = tmp_path / "inconsistent"
    with pytest.raises(ValueError, match="record population"):
        publish_split_bundle(target, identity=identity, split=inconsistent)
    assert not target.exists()


def test_split_bundle_publish_is_exclusive(tmp_path: Path) -> None:
    identity = _identity_manifest()
    split = build_patient_split_manifest(
        _data(),
        identity,
        split_version="3.1.0",
        n_splits=5,
        random_state=42,
    )
    target = tmp_path / "v3.1.0"

    publish_split_bundle(target, identity=identity, split=split)
    assert (target / "SPLIT_BUNDLE_COMPLETE.json").is_file()
    assert (target / "SPLIT_BUNDLE_COMPLETE.json.sha256").is_file()
    original = {
        path.relative_to(target): path.read_bytes() for path in target.rglob("*") if path.is_file()
    }

    with pytest.raises(FileExistsError):
        publish_split_bundle(target, identity=identity, split=split)

    assert {
        path.relative_to(target): path.read_bytes() for path in target.rglob("*") if path.is_file()
    } == original
