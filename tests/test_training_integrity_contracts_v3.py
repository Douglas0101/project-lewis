import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.training_integrity.contracts import (
    CheckStatus,
    DatasetIdentityPolicy,
    DatasetRole,
    EpistemicCategory,
    FoldAssignment,
    IdentityMethod,
    IdentityStatus,
    LegacyLeakageAudit,
    PatientIdentityManifest,
    PatientIdentityRecord,
    PatientSplitManifest,
    PreflightCheck,
    PreflightEvidenceBundle,
    TrainingGenerationManifest,
)
from src.training_integrity.integrity import (
    build_file_manifest,
    exclusive_publication,
    hash_canonical,
    publish_staged_file_exclusive,
    verify_detached_sha256,
    write_detached_sha256,
    write_json_exclusive,
)
from src.training_integrity.preflight import (
    REQUIRED_GATE_CHECK_CODES,
    REQUIRED_GATE_PASS_CHECK_CODES,
    finalize_preflight_report,
    publish_pretraining_gate,
    publish_report_bundle,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _generation() -> TrainingGenerationManifest:
    return TrainingGenerationManifest(
        schema_version="training-generation-v3.1.0",
        generation_id="advanced-training-v3.1.0-r1",
        raw_data_hash=SHA_A,
        annotation_hash=SHA_A,
        processed_data_hash=SHA_A,
        ontology_hash=SHA_A,
        preprocessing_hash=SHA_A,
        feature_schema_hash=SHA_A,
        patient_split_hash=SHA_A,
        training_config_hash=SHA_A,
        source_revision=SHA_A,
        environment_hash=SHA_A,
        research_execution_authorized=True,
        promotion_authorized=False,
    )


def test_contracts_reject_unknown_fields_and_bad_hashes() -> None:
    payload = _generation().model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        TrainingGenerationManifest.model_validate(payload)

    payload.pop("unexpected")
    payload["raw_data_hash"] = "ABC"
    with pytest.raises(ValidationError):
        TrainingGenerationManifest.model_validate(payload)

    payload["raw_data_hash"] = SHA_A
    payload["research_execution_authorized"] = 1
    with pytest.raises(ValidationError):
        TrainingGenerationManifest.model_validate(payload)


def test_dataset_identity_methods_cannot_be_reused_for_afdb() -> None:
    with pytest.raises(ValidationError, match="not valid for afdb"):
        DatasetIdentityPolicy(
            dataset_id="afdb",
            role=DatasetRole.RHYTHM_EXPLORATORY,
            method=IdentityMethod.MITDB_DOCUMENTED_SUBJECT,
            raw_dir="data/raw_afdb",
            expected_records=23,
            expected_patients=23,
            same_patient_record_groups=(),
            evidence_ref="invalid fixture",
        )


def test_unresolved_identity_cannot_enter_confirmatory_core() -> None:
    with pytest.raises(ValidationError):
        PatientIdentityRecord(
            dataset_id="svdb",
            record_id="800",
            patient_id=None,
            patient_group_id=None,
            role=DatasetRole.CONFIRMATORY_CORE,
            identity_status=IdentityStatus.IDENTITY_UNVERIFIED,
            evidence_ref="local headers contain no patient mapping",
        )


def test_unresolved_identity_is_allowed_only_in_quarantine() -> None:
    record = PatientIdentityRecord(
        dataset_id="svdb",
        record_id="800",
        patient_id=None,
        patient_group_id=None,
        role=DatasetRole.DOMAIN_SENSITIVITY,
        identity_status=IdentityStatus.IDENTITY_UNVERIFIED,
        evidence_ref="local headers contain no patient mapping",
    )
    assert record.role is DatasetRole.DOMAIN_SENSITIVITY
    assert record.patient_id is None


def test_hash_canonical_is_domain_separated() -> None:
    value = {"a": 1, "b": [2, 3]}
    assert hash_canonical("patient-identity", value) != hash_canonical("patient-split", value)
    assert hash_canonical("patient-identity", value) == hash_canonical(
        "patient-identity", {"b": [2, 3], "a": 1}
    )


def test_nested_persisted_mappings_are_immutable() -> None:
    check = PreflightCheck(
        code="IMMUTABLE",
        status=CheckStatus.PASS,
        epistemic_category=EpistemicCategory.OBSERVED,
        evidence="fixture",
        denominator="1 fixture",
        limitation="none",
        details={"nested": [1, 2]},
    )
    with pytest.raises(TypeError):
        check.details["new"] = 3  # type: ignore[index]
    assert check.details["nested"] == (1, 2)


def test_exclusive_publication_preserves_foreign_lock_and_immutable_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.bin"
    lock = tmp_path / ".publish.lock"
    lock.write_text("foreign\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        with exclusive_publication(lock, (target,)):
            raise AssertionError("foreign lock must prevent entry")
    assert lock.read_text(encoding="utf-8") == "foreign\n"

    lock.unlink()
    staged = tmp_path / "staged.bin"
    staged.write_bytes(b"immutable")
    with exclusive_publication(lock, (target,)):
        publish_staged_file_exclusive(staged, target)
    assert target.read_bytes() == b"immutable"
    assert target.stat().st_mode & 0o222 == 0
    assert not lock.exists()


def test_detached_digest_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_json_exclusive(path, _generation())
    write_detached_sha256(path)
    assert verify_detached_sha256(path)

    path.chmod(0o644)
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        verify_detached_sha256(path)


def test_caller_authored_pass_claims_cannot_publish_trusted_artifacts(tmp_path: Path) -> None:
    records = tuple(
        PatientIdentityRecord(
            dataset_id="mitdb",
            record_id=f"r{fold}",
            patient_id=f"mitdb:subject:{fold}",
            patient_group_id=f"mitdb:subject:{fold}",
            role=DatasetRole.CONFIRMATORY_CORE,
            identity_status=IdentityStatus.IDENTITY_VERIFIED,
            evidence_ref="fixture",
        )
        for fold in range(5)
    )
    identity = PatientIdentityManifest(
        schema_version="patient-identity-v3.1.0",
        source_data_hash=SHA_A,
        records=records,
        confirmatory_patient_count=5,
        confirmatory_record_count=5,
        quarantined_record_count=0,
    )
    split = PatientSplitManifest(
        schema_version="patient-split-v3.1.0",
        split_version="3.1.0",
        n_splits=5,
        random_state=42,
        source_data_hash=SHA_A,
        patient_identity_hash=hash_canonical("patient-identity", identity),
        core_dataset_ids=("mitdb",),
        quarantine_dataset_ids=(),
        folds=tuple(
            FoldAssignment(
                fold=fold,
                outer_test_patient_ids=(f"mitdb:subject:{fold}",),
                outer_test_record_keys=(f"mitdb/r{fold}",),
                n_samples=1,
                class_counts={"N": 1},
                dataset_counts={"mitdb": 1},
            )
            for fold in range(5)
        ),
        quarantined_records=(),
    )
    partial_report = finalize_preflight_report(
        generation_id="advanced-training-v3.1.0-r1",
        checks=(
            PreflightCheck(
                code="ALL_VALID",
                status=CheckStatus.PASS,
                epistemic_category=EpistemicCategory.OBSERVED,
                evidence="fixture",
                denominator="1 fixture",
                limitation="none",
                details={},
            ),
        ),
    )
    assert not partial_report.training_allowed
    assert "PREFLIGHT_CHECK_SET_INCOMPLETE" in partial_report.blocking_codes

    manifest_specs = {
        "raw_data_manifest": "raw-data",
        "annotation_manifest": "annotations",
        "processed_data_manifest": "processed-data",
        "preprocessing_manifest": "preprocessing",
        "source_manifest": "research-source",
        "ontology_manifest": "ontology-source",
        "training_config_manifest": "training-config",
    }
    manifests = {}
    for key, category in manifest_specs.items():
        file_path = tmp_path / f"{key}.txt"
        file_path.write_text(f"{key}\n", encoding="utf-8")
        manifests[key] = build_file_manifest(tmp_path, (file_path,), category=category)
    git_evidence = {"git_head": "fixture", "git_tree": "fixture", "git_dirty": False}
    environment_evidence = {"python_version": "3.12", "uv_lock_sha256": SHA_A}
    feature_schema = {"input_shape": [500, 1], "target_sampling_rate": 500}
    components = {
        **{key: manifest.model_dump(mode="json") for key, manifest in manifests.items()},
        "git": git_evidence,
        "environment": environment_evidence,
        "feature_schema": feature_schema,
    }
    checks = tuple(
        PreflightCheck(
            code=code,
            status=(
                CheckStatus.PASS if code in REQUIRED_GATE_PASS_CHECK_CODES else CheckStatus.WARN
            ),
            epistemic_category=EpistemicCategory.OBSERVED,
            evidence="fixture",
            denominator="1 fixture",
            limitation="none",
            details={},
        )
        for code in sorted(REQUIRED_GATE_CHECK_CODES)
    )
    report = finalize_preflight_report(
        generation_id="advanced-training-v3.1.0-r1",
        checks=checks,
    )
    generation = _generation().model_copy(
        update={
            "raw_data_hash": manifests["raw_data_manifest"].payload_hash,
            "annotation_hash": manifests["annotation_manifest"].payload_hash,
            "processed_data_hash": manifests["processed_data_manifest"].payload_hash,
            "preprocessing_hash": manifests["preprocessing_manifest"].payload_hash,
            "ontology_hash": manifests["ontology_manifest"].files[0].sha256,
            "training_config_hash": manifests["training_config_manifest"].files[0].sha256,
            "feature_schema_hash": hash_canonical("feature-schema-v3.1.0", feature_schema),
            "patient_split_hash": hash_canonical("patient-split", split),
            "source_revision": hash_canonical(
                "research-source-snapshot",
                {
                    "git": git_evidence,
                    "files": manifests["source_manifest"].model_dump(mode="json"),
                },
            ),
            "environment_hash": hash_canonical("research-environment", environment_evidence),
        }
    )
    bundle = PreflightEvidenceBundle(
        schema_version="training-preflight-evidence-v3.1.0",
        generated_at_utc="2026-07-21T00:00:00+00:00",
        generation_manifest=generation,
        patient_identity_manifest=identity,
        patient_split_manifest=split,
        legacy_leakage_audit=LegacyLeakageAudit(
            checked_patient_count=5,
            cross_fold_patient_count=0,
            cross_fold_patients={},
        ),
        preflight_report=report,
        component_evidence=components,
    )
    roundtrip = PreflightEvidenceBundle.model_validate_json(bundle.model_dump_json())
    assert roundtrip == bundle
    inconsistent = bundle.model_dump(mode="json")
    inconsistent["patient_split_manifest"]["patient_identity_hash"] = SHA_B
    with pytest.raises(ValidationError, match="does not bind"):
        PreflightEvidenceBundle.model_validate_json(json.dumps(inconsistent))

    report_json = tmp_path / "report.json"
    report_markdown = tmp_path / "report.md"
    completion = tmp_path / "report.COMPLETE"
    missing_config = tmp_path / "missing-config.yaml"
    with pytest.raises(FileNotFoundError):
        publish_report_bundle(
            report_json,
            report_markdown,
            completion,
            bundle,
            config_path=missing_config,
            project_root=tmp_path,
        )
    assert not report_json.exists()
    assert not completion.exists()

    gate = tmp_path / "PRETRAINING_GATE_PASS"
    with pytest.raises(FileNotFoundError):
        publish_pretraining_gate(
            gate,
            bundle,
            config_path=missing_config,
            project_root=tmp_path,
        )
    assert not gate.exists()


def test_exclusive_json_write_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_json_exclusive(path, _generation())
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        write_json_exclusive(path, _generation().model_copy(update={"raw_data_hash": SHA_B}))

    assert path.read_bytes() == original
