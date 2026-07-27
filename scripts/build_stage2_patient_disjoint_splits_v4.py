#!/usr/bin/env python3
"""Publish E07R identity, patient-disjoint splits, and legacy quarantine evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from src.models.e06_protocol import E06EvaluationContract
from src.stage2_research.e07r_contracts import (
    LegacySplitQuarantineManifestV4,
    MitdbPatientMappingDocument,
    Stage2CustodyCompleteV4,
    Stage2CustodyManifestV4,
    Stage2PatientMappingV4,
)
from src.stage2_research.integrity import hash_canonical
from src.stage2_research.patient_disjoint import (
    build_mitdb_mapping_document,
    build_stage2_patient_mapping,
    generate_patient_disjoint_splits,
    publish_patient_disjoint_bundle,
)
from src.training_integrity.integrity import (
    exclusive_publication,
    sha256_file,
    write_json_exclusive,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
R5_DIR = PROJECT_ROOT / "data/features/v3.1.0-r5-stage2-pd"
R5_NPZ = R5_DIR / "stage2_multiclass.npz"
R5_PARQUET = R5_DIR / "stage2_multiclass.parquet"
R5_MANIFEST = R5_DIR / "stage2_custody_manifest.json"
R5_COMPLETE = R5_DIR / "STAGE2_CUSTODY_COMPLETE.json"
UPSTREAM_IDENTITY = (
    PROJECT_ROOT / "data/splits/groupkfold_5_stratified/v3.1.0/patient_identity_manifest.json"
)
EXPECTED_UPSTREAM_IDENTITY_SHA256 = (
    "c8f31ddeda6825c2983e06fce31b344935c0a419d90386359c1d378279f62bab"
)
MITDB_MAPPING = PROJECT_ROOT / "data/metadata/physionet_mitdb_patient_mapping.json"
STAGE2_MAPPING = PROJECT_ROOT / "data/metadata/stage2_patient_identity_v4.0.json"
SPLIT_TARGET = PROJECT_ROOT / "data/splits/stage2_multiclass_patient_disjoint_v4.0"
QUARANTINE_DIR = (
    PROJECT_ROOT / "experiments/stage2_v2.4_research/quarantine/"
    "splits_record_disjoint_leakage_era_v2.3"
)
LEAKAGE_REPORT = (
    PROJECT_ROOT / "experiments/stage2_v2.4_research/integrity/e07r_split_leakage_report.json"
)
PHYSIONET_URL = "https://physionet.org/physiobank/database/html/mitdbdir/intro.htm"
SOURCE_FILES = (
    "src/models/e06_protocol.py",
    "src/stage2_research/e07r_contracts.py",
    "src/stage2_research/patient_disjoint.py",
    "scripts/build_stage2_patient_disjoint_splits_v4.py",
)
LEGACY_SPLIT_PATHS = (
    "experiments/stage2_v2.4_research/splits/outer_splits_v2.4.json",
    "experiments/stage2_v2.4_research/splits/inner_splits_v2.4.json",
    "experiments/stage2_v2.4_research/splits/split_diagnostics.csv",
    "experiments/stage2_v2.4_research/E03_split_protocol/" "split_manifest_GroupKFold.json",
    "experiments/stage2_v2.4_research/E03_split_protocol/"
    "split_manifest_StratifiedGroupKFold.json",
)


def _load_custody() -> Stage2CustodyManifestV4:
    manifest = Stage2CustodyManifestV4.model_validate_json(R5_MANIFEST.read_text())
    complete = Stage2CustodyCompleteV4.model_validate_json(R5_COMPLETE.read_text())
    if complete.manifest_hash != manifest.manifest_hash:
        raise ValueError("r5 completion marker does not bind the custody manifest")
    if complete.manifest_file_sha256 != sha256_file(R5_MANIFEST):
        raise ValueError("r5 custody manifest byte hash mismatch")
    for name, expected in manifest.output_file_sha256.items():
        if sha256_file(R5_DIR / name) != expected:
            raise ValueError(f"r5 custody artifact hash mismatch: {name}")
    return manifest


def _load_upstream_records() -> list[dict[str, Any]]:
    try:
        document = json.loads(UPSTREAM_IDENTITY.read_text(encoding="utf-8"))
        records = document["records"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("cannot load upstream patient identity records") from error
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("upstream patient identity records are not a JSON object list")
    return cast(list[dict[str, Any]], records)


def _source_manifest_hash(custody: Stage2CustodyManifestV4) -> str:
    return hash_canonical(
        {
            "custody_manifest_hash": custody.manifest_hash,
            "files": {path: sha256_file(PROJECT_ROOT / path) for path in SOURCE_FILES},
        }
    )


def _quarantine_manifest() -> LegacySplitQuarantineManifestV4:
    artifacts = []
    for relative in LEGACY_SPLIT_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise ValueError(f"legacy split evidence is missing: {relative}")
        artifacts.append(
            {
                "artifact_path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload = {
        "schema_version": "legacy-split-quarantine-v4.0",
        "status": "QUARANTINED_NOT_DELETED",
        "reason": "PATIENT_LEAKAGE_RECORD_DISJOINT_ONLY",
        "date": "2026-07-26",
        "active_for_e07r": False,
        "replacement_split_version": "v4.0-patient-disjoint",
        "artifacts": artifacts,
    }
    return LegacySplitQuarantineManifestV4.model_validate(
        {**payload, "manifest_hash": hash_canonical(payload)}
    )


def _publish_identity(
    mitdb_document: MitdbPatientMappingDocument,
    mapping_document: Stage2PatientMappingV4,
) -> None:
    lock = PROJECT_ROOT / "data/metadata/.e07r-identity-v4.publish.lock"
    with exclusive_publication(lock, [MITDB_MAPPING, STAGE2_MAPPING]):
        write_json_exclusive(
            MITDB_MAPPING,
            mitdb_document.model_dump(mode="json"),
        )
        write_json_exclusive(
            STAGE2_MAPPING,
            mapping_document.model_dump(mode="json"),
        )


def _publish_quarantine(manifest: LegacySplitQuarantineManifestV4) -> None:
    path = QUARANTINE_DIR / "quarantine_manifest.json"
    lock = QUARANTINE_DIR.parent / ".legacy-splits-quarantine.publish.lock"
    with exclusive_publication(lock, [path]):
        write_json_exclusive(path, manifest.model_dump(mode="json"))


def main() -> int:
    custody = _load_custody()
    if sha256_file(UPSTREAM_IDENTITY) != EXPECTED_UPSTREAM_IDENTITY_SHA256:
        raise ValueError("upstream patient identity SHA-256 mismatch")
    frame = pd.read_parquet(R5_PARQUET)
    with np.load(R5_NPZ, allow_pickle=False) as archive:
        labels = np.asarray(archive["y"], dtype=np.int64)
    upstream_records = _load_upstream_records()
    mapping = build_stage2_patient_mapping(
        frame,
        upstream_records,
        stage2_parquet_sha256=sha256_file(R5_PARQUET),
        upstream_identity_sha256=EXPECTED_UPSTREAM_IDENTITY_SHA256,
        mitdb_source_url=PHYSIONET_URL,
    )
    mitdb_document = build_mitdb_mapping_document(mapping)
    source_manifest_hash = _source_manifest_hash(custody)
    generated = generate_patient_disjoint_splits(
        frame,
        labels,
        mapping,
        E06EvaluationContract(n_splits=5, inner_splits=4, random_seed=42),
        source_manifest_hash=source_manifest_hash,
    )
    quarantine = _quarantine_manifest()

    _publish_identity(mitdb_document, mapping)
    split_hashes = publish_patient_disjoint_bundle(SPLIT_TARGET, generated)
    _publish_quarantine(quarantine)
    write_json_exclusive(
        LEAKAGE_REPORT,
        generated.leakage_report.model_dump(mode="json"),
    )
    print(
        json.dumps(
            {
                "status": generated.leakage_report.status,
                "mapping_hash": mapping.mapping_hash,
                "verified_patients": mapping.verified_patient_count,
                "records": mapping.record_count,
                "split_manifest_hash": generated.split_manifest.manifest_hash,
                "outer_folds_checked": generated.leakage_report.outer_folds_checked,
                "inner_folds_checked": generated.leakage_report.inner_folds_checked,
                "known_group_201_202_respected": (
                    generated.leakage_report.known_group_201_202_respected
                ),
                "split_file_sha256": split_hashes,
                "quarantine_manifest_hash": quarantine.manifest_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
