#!/usr/bin/env python3
"""Publish and validate the write-once E07R integrity freeze."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from src.stage2_research.e07r_contracts import (
    E07REvidenceCompleteV4,
    E07RPDProtocolManifestV4,
    E07RPreauthorizationManifestV4,
    E07RProducerAttestationV4,
    LegacySplitQuarantineManifestV4,
    PatientDisjointSplitManifestV4,
    SplitLeakageReportV4,
    Stage2CustodyManifestV4,
    Stage2PatientMappingV4,
)
from src.stage2_research.e07r_integrity import (
    E07RIntegrityPaths,
    build_e07r_freeze_manifest,
    protect_freeze_pins,
    run_e07r_preflight,
)
from src.stage2_research.integrity import hash_canonical
from src.stage2_research.pd_workflows import build_pd_protocol_manifest
from src.training_integrity.integrity import sha256_file, write_json_exclusive

TModel = TypeVar("TModel", bound=BaseModel)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCER_FILES = (
    "scripts/build_stage2_patient_disjoint_v4.py",
    "src/stage2_research/stage2_custody.py",
    "src/stage2_research/e07r_contracts.py",
    "src/training_integrity/integrity.py",
    "src/training_integrity/preflight.py",
)
EVIDENCE_FILES = (
    "data/features/v3.1.0-r5-stage2-pd/stage2_multiclass.npz",
    "data/features/v3.1.0-r5-stage2-pd/stage2_multiclass.parquet",
    "data/features/v3.1.0-r5-stage2-pd/stage2_custody_manifest.json",
    "data/features/v3.1.0-r5-stage2-pd/STAGE2_CUSTODY_COMPLETE.json",
    "data/metadata/physionet_mitdb_patient_mapping.json",
    "data/metadata/stage2_patient_identity_v4.0.json",
    "data/splits/stage2_multiclass_patient_disjoint_v4.0/patient_groups.json",
    "data/splits/stage2_multiclass_patient_disjoint_v4.0/outer_folds.json",
    "data/splits/stage2_multiclass_patient_disjoint_v4.0/inner_folds.json",
    "data/splits/stage2_multiclass_patient_disjoint_v4.0/leakage_checks.json",
    "data/splits/stage2_multiclass_patient_disjoint_v4.0/fold_statistics.json",
    "data/splits/stage2_multiclass_patient_disjoint_v4.0/outer_splits_stage2.json",
    "data/splits/stage2_multiclass_patient_disjoint_v4.0/inner_splits_stage2.json",
    "data/splits/stage2_multiclass_patient_disjoint_v4.0/split_manifest.json",
    "experiments/stage2_v2.4_research/quarantine/"
    "splits_record_disjoint_leakage_era_v2.3/quarantine_manifest.json",
    "experiments/stage2_v2.4_research/integrity/e07r_split_leakage_report.json",
    "experiments/stage2_v2.4_research/integrity/e07r_preauth_manifest.json",
    "experiments/stage2_v2.4_research/integrity/e07r_pd_protocol_manifest.json",
    "experiments/stage2_v2.4_research/integrity/e07r_r5_producer_attestation.json",
    "docs/e07r_governance_preauthorization.md",
    "docs/e07r_execution_plan.md",
    "docs/physionet_mitdb_patient_statement.md",
)


def _model(path: Path, model: type[TModel]) -> TModel:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid freeze input: {path}") from error


def _publish_producer_attestation(paths: E07RIntegrityPaths) -> E07RProducerAttestationV4:
    custody = _model(
        paths.r5_dir / "stage2_custody_manifest.json",
        Stage2CustodyManifestV4,
    )
    payload = {
        "schema_version": "e07r-r5-producer-attestation-v1",
        "generation_id": "v3.1.0-r5-stage2-pd",
        "custody_manifest_hash": custody.manifest_hash,
        "producer_file_sha256": {
            relative: sha256_file(PROJECT_ROOT / relative) for relative in PRODUCER_FILES
        },
        "date": "2026-07-26",
    }
    attestation = E07RProducerAttestationV4.model_validate(
        {**payload, "attestation_hash": hash_canonical(payload)}
    )
    write_json_exclusive(
        paths.producer_attestation,
        attestation.model_dump(mode="json"),
    )
    return attestation


def _publish_evidence_completion(
    paths: E07RIntegrityPaths,
    producer: E07RProducerAttestationV4,
    protocol: E07RPDProtocolManifestV4,
) -> E07REvidenceCompleteV4:
    custody = _model(
        paths.r5_dir / "stage2_custody_manifest.json",
        Stage2CustodyManifestV4,
    )
    mapping = _model(paths.mapping, Stage2PatientMappingV4)
    split = _model(
        paths.split_dir / "split_manifest.json",
        PatientDisjointSplitManifestV4,
    )
    leakage = _model(
        paths.split_dir / "leakage_checks.json",
        SplitLeakageReportV4,
    )
    quarantine = _model(
        paths.quarantine_manifest,
        LegacySplitQuarantineManifestV4,
    )
    preauthorization = _model(
        paths.preauthorization_manifest,
        E07RPreauthorizationManifestV4,
    )
    payload = {
        "schema_version": "e07r-evidence-complete-v4.0",
        "status": "COMPLETE",
        "date": "2026-07-26",
        "custody_manifest_hash": custody.manifest_hash,
        "patient_mapping_hash": mapping.mapping_hash,
        "split_manifest_hash": split.manifest_hash,
        "leakage_report_hash": leakage.report_hash,
        "quarantine_manifest_hash": quarantine.manifest_hash,
        "preauthorization_manifest_hash": preauthorization.manifest_hash,
        "pd_protocol_manifest_hash": protocol.manifest_hash,
        "producer_attestation_hash": producer.attestation_hash,
        "artifact_file_sha256": {
            relative: sha256_file(PROJECT_ROOT / relative) for relative in EVIDENCE_FILES
        },
    }
    completion = E07REvidenceCompleteV4.model_validate(
        {**payload, "completion_hash": hash_canonical(payload)}
    )
    write_json_exclusive(
        paths.evidence_complete,
        completion.model_dump(mode="json"),
    )
    return completion


def main() -> int:
    paths = E07RIntegrityPaths(PROJECT_ROOT)
    protocol = build_pd_protocol_manifest(PROJECT_ROOT)
    write_json_exclusive(
        paths.pd_protocol_manifest,
        protocol.model_dump(mode="json"),
    )
    producer = _publish_producer_attestation(paths)
    completion = _publish_evidence_completion(paths, producer, protocol)
    freeze = build_e07r_freeze_manifest(PROJECT_ROOT)
    write_json_exclusive(paths.freeze_manifest, freeze.model_dump(mode="json"))
    protect_freeze_pins(
        PROJECT_ROOT,
        freeze,
        include_manifest=paths.freeze_manifest,
    )
    report = run_e07r_preflight(
        PROJECT_ROOT,
        workflow="FREEZE_VALIDATION",
        run_id="e07r-freeze-v4",
    )
    if report.status != "PASS":
        raise RuntimeError(
            "published E07R freeze failed its canonical validation: " + report.model_dump_json()
        )
    print(
        json.dumps(
            {
                "status": report.status,
                "pd_protocol_manifest_hash": protocol.manifest_hash,
                "producer_attestation_hash": producer.attestation_hash,
                "evidence_completion_hash": completion.completion_hash,
                "freeze_manifest_hash": freeze.manifest_hash,
                "freeze_pins": len(freeze.pins),
                "checks": [check.code for check in report.checks],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
