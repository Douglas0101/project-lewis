import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from src.training_integrity.config import (
    load_advanced_training_config,
    load_patient_identity_policy,
)
from src.training_integrity.contracts import (
    DatasetIdentityPolicy,
    DatasetRole,
    IdentityMethod,
    IdentityStatus,
    PatientIdentityPolicy,
)
from src.training_integrity.patient_identity import (
    audit_legacy_outer_fold_leakage,
    build_patient_identity_manifest,
)

SOURCE_HASH = "1" * 64


def _write_header(path: Path, *comments: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"{path.stem} 1 257 1000", f"{path.stem}.dat 16 200/mV 16 0 0 0 0"]
    body.extend(f"# {comment}" for comment in comments)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _policy() -> PatientIdentityPolicy:
    return PatientIdentityPolicy(
        schema_version="patient-identity-policy-v3.1.0",
        datasets=(
            DatasetIdentityPolicy(
                dataset_id="incart",
                role=DatasetRole.CONFIRMATORY_CORE,
                method=IdentityMethod.INCART_HEADER_PATIENT,
                raw_dir="raw_incart",
                expected_records=2,
                expected_patients=1,
                evidence_ref="local WFDB header patient number",
            ),
            DatasetIdentityPolicy(
                dataset_id="mitdb",
                role=DatasetRole.CONFIRMATORY_CORE,
                method=IdentityMethod.MITDB_DOCUMENTED_SUBJECT,
                raw_dir="raw_mitbih",
                expected_records=3,
                expected_patients=2,
                same_patient_record_groups=(("201", "202"),),
                evidence_ref="PhysioNet records directory",
            ),
            DatasetIdentityPolicy(
                dataset_id="svdb",
                role=DatasetRole.DOMAIN_SENSITIVITY,
                method=IdentityMethod.UNRESOLVED,
                raw_dir="raw_svdb",
                expected_records=1,
                expected_patients=None,
                evidence_ref="patient linkage unavailable",
            ),
        ),
    )


def _records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["incart", "incart", "mitdb", "mitdb", "mitdb", "svdb"],
            "record_id": ["I01", "I02", "100", "201", "202", "800"],
        }
    )


def test_identity_manifest_groups_evidence_backed_patients(tmp_path: Path) -> None:
    _write_header(
        tmp_path / "raw_incart" / "I01.hea",
        "<age>: 65 <sex>: F <diagnoses> hidden",
        "patient 1",
    )
    _write_header(tmp_path / "raw_incart" / "I02.hea", "patient 1")
    for record in ("100", "201", "202"):
        _write_header(tmp_path / "raw_mitbih" / f"{record}.hea")
    _write_header(tmp_path / "raw_svdb" / "800.hea")

    manifest = build_patient_identity_manifest(
        _records(),
        project_root=tmp_path,
        policy=_policy(),
        source_data_hash=SOURCE_HASH,
    )
    by_key = {(record.dataset_id, record.record_id): record for record in manifest.records}

    assert by_key[("incart", "I01")].patient_id == "incart:patient:1"
    assert by_key[("incart", "I02")].patient_id == "incart:patient:1"
    assert by_key[("mitdb", "201")].patient_id == "mitdb:subject:201-202"
    assert by_key[("mitdb", "202")].patient_id == "mitdb:subject:201-202"
    assert by_key[("mitdb", "100")].patient_id == "mitdb:subject:100"
    assert by_key[("svdb", "800")].identity_status is IdentityStatus.IDENTITY_UNVERIFIED
    assert by_key[("svdb", "800")].patient_id is None
    assert manifest.confirmatory_patient_count == 3
    assert manifest.quarantined_record_count == 1

    serialized = json.dumps(manifest.model_dump(mode="json"), sort_keys=True)
    assert "<age>" not in serialized
    assert "diagnoses" not in serialized


def test_identity_policy_rejects_raw_path_escape() -> None:
    with pytest.raises(ValidationError):
        DatasetIdentityPolicy(
            dataset_id="incart",
            role=DatasetRole.CONFIRMATORY_CORE,
            method=IdentityMethod.INCART_HEADER_PATIENT,
            raw_dir="../outside",
            expected_records=1,
            expected_patients=1,
            evidence_ref="fixture",
        )


def test_identity_manifest_rejects_policy_dataset_missing_from_inputs(tmp_path: Path) -> None:
    _write_header(tmp_path / "raw_incart" / "I01.hea", "patient 1")
    _write_header(tmp_path / "raw_incart" / "I02.hea", "patient 1")
    for record in ("100", "201", "202"):
        _write_header(tmp_path / "raw_mitbih" / f"{record}.hea")
    with pytest.raises(ValueError, match="datasets absent from inputs"):
        build_patient_identity_manifest(
            _records().query("dataset != 'svdb'"),
            project_root=tmp_path,
            policy=_policy(),
            source_data_hash=SOURCE_HASH,
        )


def test_incart_header_requires_exactly_one_patient_line(tmp_path: Path) -> None:
    _write_header(tmp_path / "raw_incart" / "I01.hea", "patient 1", "patient 2")
    _write_header(tmp_path / "raw_incart" / "I02.hea", "patient 1")
    for record in ("100", "201", "202"):
        _write_header(tmp_path / "raw_mitbih" / f"{record}.hea")
    _write_header(tmp_path / "raw_svdb" / "800.hea")

    with pytest.raises(ValueError, match="exactly one INCART patient"):
        build_patient_identity_manifest(
            _records(),
            project_root=tmp_path,
            policy=_policy(),
            source_data_hash=SOURCE_HASH,
        )


def test_legacy_leakage_audit_detects_same_patient_across_folds(tmp_path: Path) -> None:
    _write_header(tmp_path / "raw_incart" / "I01.hea", "patient 1")
    _write_header(tmp_path / "raw_incart" / "I02.hea", "patient 1")
    for record in ("100", "201", "202"):
        _write_header(tmp_path / "raw_mitbih" / f"{record}.hea")
    _write_header(tmp_path / "raw_svdb" / "800.hea")
    manifest = build_patient_identity_manifest(
        _records(),
        project_root=tmp_path,
        policy=_policy(),
        source_data_hash=SOURCE_HASH,
    )

    split_dir = tmp_path / "legacy"
    split_dir.mkdir()
    (split_dir / "fold_0.json").write_text(
        json.dumps({"val_record_ids": ["I01", "100", "201"]}), encoding="utf-8"
    )
    (split_dir / "fold_1.json").write_text(
        json.dumps({"val_record_ids": ["I02", "202"]}), encoding="utf-8"
    )

    result = audit_legacy_outer_fold_leakage(manifest, split_dir=split_dir)
    assert result.cross_fold_patient_count == 2
    assert result.cross_fold_patients["incart:patient:1"] == (0, 1)
    assert result.cross_fold_patients["mitdb:subject:201-202"] == (0, 1)


def test_project_identity_audit_reproduces_known_counts() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / "advanced_training_v3.1.yaml"
    family_path = project_root / "data" / "features" / "finetuning_mitbih_family.parquet"
    rhythm_path = project_root / "data" / "features" / "afdb_rhythm_episodes.parquet"
    if not family_path.is_file() or not rhythm_path.is_file():
        pytest.skip("DVC training artifacts are unavailable")

    config, resolved_root = load_advanced_training_config(config_path)
    policy = load_patient_identity_policy(resolved_root / config.identity_policy)
    family = pd.read_parquet(family_path, columns=["dataset", "record_id"])
    rhythm = pd.read_parquet(rhythm_path, columns=["record_id"]).assign(dataset="afdb")
    manifest = build_patient_identity_manifest(
        pd.concat([family, rhythm], ignore_index=True),
        project_root=resolved_root,
        policy=policy,
        source_data_hash=SOURCE_HASH,
    )
    leakage = audit_legacy_outer_fold_leakage(
        manifest,
        split_dir=resolved_root / config.legacy_split_dir,
    )

    assert manifest.confirmatory_record_count == 123
    assert manifest.confirmatory_patient_count == 79
    assert manifest.quarantined_record_count == 101
    assert leakage.cross_fold_patient_count == 29
