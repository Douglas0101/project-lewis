"""Fail-closed E07R freeze, preflight, and write-policy enforcement."""

from __future__ import annotations

import fcntl
import json
import os
import stat
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.stage2_research.e07r_contracts import (
    E07REvidenceCompleteV4,
    E07RFreezeManifestV4,
    E07RIntegrityCheckV4,
    E07RIntegrityViolationV4,
    E07RPDProtocolManifestV4,
    E07RPreauthorizationManifestV4,
    E07RPreflightReportV4,
    E07RProducerAttestationV4,
    FoldStatisticsV4,
    LegacySplitQuarantineManifestV4,
    PatientDisjointSplitManifestV4,
    PatientInnerFoldsV4,
    PatientOuterFoldsV4,
    PatientSplitPartitionV4,
    SplitLeakageReportV4,
    Stage2CustodyCompleteV4,
    Stage2CustodyManifestV4,
    Stage2PatientMappingV4,
)
from src.stage2_research.integrity import hash_canonical
from src.training_integrity.integrity import sha256_file, write_json_exclusive
from src.training_integrity.preflight import ordered_row_binding_check

TModel = TypeVar("TModel", bound=BaseModel)
PinRole = Literal[
    "CUSTODY",
    "IDENTITY",
    "SPLIT",
    "GOVERNANCE",
    "QUARANTINE",
    "SOURCE",
    "LEGACY_SENTINEL",
]
SOURCE_PIN_PATHS = (
    "src/models/e06_protocol.py",
    "src/stage2_research/config.py",
    "src/stage2_research/contracts.py",
    "src/stage2_research/data.py",
    "src/stage2_research/e07r_contracts.py",
    "src/stage2_research/e07r_integrity.py",
    "src/stage2_research/features.py",
    "src/stage2_research/integrity.py",
    "src/stage2_research/patient_disjoint.py",
    "src/stage2_research/pd_workflows.py",
    "src/stage2_research/splits.py",
    "src/stage2_research/stage2_custody.py",
    "src/stage2_research/training.py",
    "src/training_integrity/integrity.py",
    "src/training_integrity/preflight.py",
    "scripts/build_stage2_patient_disjoint_v4.py",
    "scripts/build_stage2_patient_disjoint_splits_v4.py",
    "scripts/freeze_e07r_integrity_v4.py",
    "scripts/run_stage2_e07r_pd.py",
    "config/stage2_research.yaml",
    "uv.lock",
)
LEGACY_TREE_ROOTS = (
    "models",
    "data/features/backup_v2.3",
    "data/features/quarantine_v31_working_20260718",
)


class E07RIntegrityError(RuntimeError):
    """A fail-closed E07R integrity or authorization failure."""


@dataclass(frozen=True)
class E07RIntegrityPaths:
    """Canonical E07R paths rooted at one project checkout."""

    project_root: Path

    @property
    def freeze_manifest(self) -> Path:
        return self.project_root / (
            "experiments/stage2_v2.4_research/integrity/e07r_freeze_manifest.json"
        )

    @property
    def preauthorization_manifest(self) -> Path:
        return self.project_root / (
            "experiments/stage2_v2.4_research/integrity/e07r_preauth_manifest.json"
        )

    @property
    def violation_log(self) -> Path:
        return self.project_root / (
            "experiments/stage2_v2.4_research/integrity/" "e07r_integrity_violations.jsonl"
        )

    @property
    def r5_dir(self) -> Path:
        return self.project_root / "data/features/v3.1.0-r5-stage2-pd"

    @property
    def mapping(self) -> Path:
        return self.project_root / "data/metadata/stage2_patient_identity_v4.0.json"

    @property
    def split_dir(self) -> Path:
        return self.project_root / ("data/splits/stage2_multiclass_patient_disjoint_v4.0")

    @property
    def quarantine_manifest(self) -> Path:
        return self.project_root / (
            "experiments/stage2_v2.4_research/quarantine/"
            "splits_record_disjoint_leakage_era_v2.3/quarantine_manifest.json"
        )

    @property
    def producer_attestation(self) -> Path:
        return self.project_root / (
            "experiments/stage2_v2.4_research/integrity/" "e07r_r5_producer_attestation.json"
        )

    @property
    def evidence_complete(self) -> Path:
        return self.project_root / (
            "experiments/stage2_v2.4_research/integrity/" "E07R_EVIDENCE_COMPLETE.json"
        )

    @property
    def pd_protocol_manifest(self) -> Path:
        return self.project_root / (
            "experiments/stage2_v2.4_research/integrity/" "e07r_pd_protocol_manifest.json"
        )


@dataclass(frozen=True)
class FreezePinSpec:
    """Input to deterministic freeze-manifest construction."""

    artifact_path: str
    role: PinRole
    enforce_read_only: bool


def _load_model(path: Path, model: type[TModel]) -> TModel:
    try:
        content = path.read_text(encoding="utf-8")
        return model.model_validate_json(content)
    except (OSError, ValueError) as error:
        raise E07RIntegrityError(f"invalid {model.__name__}: {path}") from error


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise E07RIntegrityError(f"invalid JSON artifact: {path}") from error


def _relative(project_root: Path, path: Path) -> str:
    root = project_root.resolve()
    candidate = path.resolve(strict=False)
    try:
        return str(candidate.relative_to(root))
    except ValueError:
        return str(candidate)


def production_freeze_pin_specs(project_root: Path) -> tuple[FreezePinSpec, ...]:
    """Return the exact production inventory required by the E07R freeze."""
    root = project_root.resolve()
    specs: dict[str, FreezePinSpec] = {}

    def add(paths: Iterable[str], role: PinRole, read_only: bool) -> None:
        for relative in paths:
            if relative in specs:
                raise E07RIntegrityError(f"duplicate production freeze pin: {relative}")
            specs[relative] = FreezePinSpec(relative, role, read_only)

    add(
        (
            "data/features/v3.1.0-r5-stage2-pd/stage2_multiclass.npz",
            "data/features/v3.1.0-r5-stage2-pd/stage2_multiclass.parquet",
            "data/features/v3.1.0-r5-stage2-pd/stage2_custody_manifest.json",
            "data/features/v3.1.0-r5-stage2-pd/STAGE2_CUSTODY_COMPLETE.json",
            "experiments/stage2_v2.4_research/integrity/" "e07r_r5_producer_attestation.json",
        ),
        "CUSTODY",
        True,
    )
    add(
        (
            "data/metadata/physionet_mitdb_patient_mapping.json",
            "data/metadata/stage2_patient_identity_v4.0.json",
        ),
        "IDENTITY",
        True,
    )
    split_root = "data/splits/stage2_multiclass_patient_disjoint_v4.0"
    add(
        tuple(
            f"{split_root}/{name}"
            for name in (
                "patient_groups.json",
                "outer_folds.json",
                "inner_folds.json",
                "leakage_checks.json",
                "fold_statistics.json",
                "outer_splits_stage2.json",
                "inner_splits_stage2.json",
                "split_manifest.json",
            )
        )
        + ("experiments/stage2_v2.4_research/integrity/" "e07r_split_leakage_report.json",),
        "SPLIT",
        True,
    )
    add(
        (
            "experiments/stage2_v2.4_research/integrity/" "e07r_preauth_manifest.json",
            "experiments/stage2_v2.4_research/integrity/" "E07R_EVIDENCE_COMPLETE.json",
            "experiments/stage2_v2.4_research/integrity/" "e07r_pd_protocol_manifest.json",
            "docs/e07r_governance_preauthorization.md",
            "docs/e07r_execution_plan.md",
            "docs/physionet_mitdb_patient_statement.md",
        ),
        "GOVERNANCE",
        True,
    )
    add(
        (
            "experiments/stage2_v2.4_research/quarantine/"
            "splits_record_disjoint_leakage_era_v2.3/quarantine_manifest.json",
        ),
        "QUARANTINE",
        True,
    )
    add(SOURCE_PIN_PATHS, "SOURCE", False)

    preauthorization = _load_model(
        E07RIntegrityPaths(root).preauthorization_manifest,
        E07RPreauthorizationManifestV4,
    )
    legacy_files = {
        path
        for path in preauthorization.legacy_hashes
        if path not in {"models_tree", "backup_v2_3_tree", "quarantine_v3_1_tree"}
    }
    quarantine = _load_model(
        E07RIntegrityPaths(root).quarantine_manifest,
        LegacySplitQuarantineManifestV4,
    )
    legacy_files.update(item.artifact_path for item in quarantine.artifacts)
    for relative_root in LEGACY_TREE_ROOTS:
        tree_root = root / relative_root
        if not tree_root.is_dir():
            raise E07RIntegrityError(f"legacy sentinel tree missing: {relative_root}")
        legacy_files.update(
            str(path.relative_to(root)) for path in sorted(tree_root.rglob("*")) if path.is_file()
        )
    add(tuple(sorted(legacy_files)), "LEGACY_SENTINEL", False)
    return tuple(specs[path] for path in sorted(specs))


def _assemble_e07r_freeze_manifest(
    project_root: Path,
    pin_specs: Iterable[FreezePinSpec],
    *,
    custody_manifest_hash: str,
    patient_mapping_hash: str,
    split_manifest_hash: str,
    preauthorization_manifest_hash: str,
) -> E07RFreezeManifestV4:
    root = project_root.resolve()
    pins: list[dict[str, object]] = []
    for spec in sorted(pin_specs, key=lambda item: item.artifact_path):
        path = root / spec.artifact_path
        if not path.is_file():
            raise E07RIntegrityError(f"freeze pin is not a file: {spec.artifact_path}")
        pins.append(
            {
                "artifact_path": spec.artifact_path,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "role": spec.role,
                "enforce_read_only": spec.enforce_read_only,
            }
        )
    source_payload = {
        str(pin["artifact_path"]): str(pin["sha256"]) for pin in pins if pin["role"] == "SOURCE"
    }
    payload = {
        "schema_version": "e07r-freeze-v4.0",
        "stage": "E07R",
        "status": "FROZEN",
        "date": "2026-07-26",
        "split_version": "v4.0-patient-disjoint",
        "custody_manifest_hash": custody_manifest_hash,
        "patient_mapping_hash": patient_mapping_hash,
        "split_manifest_hash": split_manifest_hash,
        "preauthorization_manifest_hash": preauthorization_manifest_hash,
        "source_manifest_hash": hash_canonical(source_payload),
        "pins": pins,
    }
    return E07RFreezeManifestV4.model_validate(
        {**payload, "manifest_hash": hash_canonical(payload)}
    )


def build_e07r_freeze_manifest(project_root: Path) -> E07RFreezeManifestV4:
    """Validate production evidence and derive every freeze header from disk."""
    paths = E07RIntegrityPaths(project_root.resolve())
    verify_preauthorization(paths)
    verify_custody(paths)
    verify_patient_mapping(paths)
    verify_split_bundle(paths)
    verify_quarantine(paths)
    verify_pd_protocol(paths)
    verify_evidence_completion(paths)
    custody = _load_model(
        paths.r5_dir / "stage2_custody_manifest.json",
        Stage2CustodyManifestV4,
    )
    mapping = _load_model(paths.mapping, Stage2PatientMappingV4)
    split = _load_model(
        paths.split_dir / "split_manifest.json",
        PatientDisjointSplitManifestV4,
    )
    preauthorization = _load_model(
        paths.preauthorization_manifest,
        E07RPreauthorizationManifestV4,
    )
    return _assemble_e07r_freeze_manifest(
        paths.project_root,
        production_freeze_pin_specs(paths.project_root),
        custody_manifest_hash=custody.manifest_hash,
        patient_mapping_hash=mapping.mapping_hash,
        split_manifest_hash=split.manifest_hash,
        preauthorization_manifest_hash=preauthorization.manifest_hash,
    )


def protect_freeze_pins(
    project_root: Path,
    manifest: E07RFreezeManifestV4,
    *,
    include_manifest: Path | None = None,
) -> None:
    """Remove write bits from new E07R frozen artifacts without touching legacy bytes."""
    root = project_root.resolve()
    paths = [root / pin.artifact_path for pin in manifest.pins if pin.enforce_read_only]
    if include_manifest is not None:
        paths.append(include_manifest)
    for path in paths:
        if not path.is_file():
            raise E07RIntegrityError(f"cannot protect missing freeze path: {path}")
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise E07RIntegrityError(f"write bits remain on frozen path: {path}")


def verify_freeze_pins(
    project_root: Path,
    manifest: E07RFreezeManifestV4,
) -> str:
    """Rehash and permission-check every freeze pin."""
    root = project_root.resolve()
    for pin in manifest.pins:
        path = root / pin.artifact_path
        if not path.is_file():
            raise E07RIntegrityError(f"frozen artifact missing: {pin.artifact_path}")
        if path.stat().st_size != pin.size_bytes:
            raise E07RIntegrityError(f"frozen artifact size changed: {pin.artifact_path}")
        if sha256_file(path) != pin.sha256:
            raise E07RIntegrityError(f"frozen artifact hash changed: {pin.artifact_path}")
        if pin.enforce_read_only and stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise E07RIntegrityError(
                f"frozen artifact regained write permission: {pin.artifact_path}"
            )
    return f"{len(manifest.pins)} content-addressed pins unchanged"


def verify_freeze_inventory_and_links(
    paths: E07RIntegrityPaths,
    manifest: E07RFreezeManifestV4,
) -> str:
    """Require the exact production inventory and bind freeze headers to components."""
    expected_specs = {
        spec.artifact_path: (spec.role, spec.enforce_read_only)
        for spec in production_freeze_pin_specs(paths.project_root)
    }
    actual_specs = {pin.artifact_path: (pin.role, pin.enforce_read_only) for pin in manifest.pins}
    if actual_specs != expected_specs:
        missing = sorted(set(expected_specs) - set(actual_specs))
        unexpected = sorted(set(actual_specs) - set(expected_specs))
        raise E07RIntegrityError(
            f"freeze inventory mismatch; missing={missing}, unexpected={unexpected}"
        )
    custody = _load_model(
        paths.r5_dir / "stage2_custody_manifest.json",
        Stage2CustodyManifestV4,
    )
    mapping = _load_model(paths.mapping, Stage2PatientMappingV4)
    split = _load_model(
        paths.split_dir / "split_manifest.json",
        PatientDisjointSplitManifestV4,
    )
    preauthorization = _load_model(
        paths.preauthorization_manifest,
        E07RPreauthorizationManifestV4,
    )
    expected_links = {
        "custody_manifest_hash": custody.manifest_hash,
        "patient_mapping_hash": mapping.mapping_hash,
        "split_manifest_hash": split.manifest_hash,
        "preauthorization_manifest_hash": preauthorization.manifest_hash,
    }
    for field, expected in expected_links.items():
        if getattr(manifest, field) != expected:
            raise E07RIntegrityError(f"freeze header link mismatch: {field}")
    if stat.S_IMODE(paths.freeze_manifest.stat().st_mode) & 0o222:
        raise E07RIntegrityError("freeze manifest itself is not read-only")
    return f"exact production freeze inventory and {len(expected_links)} links verified"


def _verify_hash_mapping(project_root: Path, hashes: Mapping[str, str]) -> int:
    checked = 0
    for relative, expected in sorted(hashes.items()):
        if relative.endswith("_tree") or relative in {
            "models_tree",
            "backup_v2_3_tree",
            "quarantine_v3_1_tree",
        }:
            continue
        path = project_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise E07RIntegrityError(f"preauthorization pin mismatch: {relative}")
        checked += 1
    return checked


def verify_preauthorization(paths: E07RIntegrityPaths) -> str:
    """Validate governance signature and every file pin it declares."""
    manifest = _load_model(
        paths.preauthorization_manifest,
        E07RPreauthorizationManifestV4,
    )
    checked = _verify_hash_mapping(paths.project_root, manifest.legacy_hashes)
    checked += _verify_hash_mapping(paths.project_root, manifest.new_artifact_hashes)
    if manifest.publication_authorized or manifest.model_promotion_authorized:
        raise E07RIntegrityError("preauthorization unexpectedly allows publication/promotion")
    if manifest.gate_relaxation_authorized:
        raise E07RIntegrityError("preauthorization unexpectedly allows gate relaxation")
    return f"governance manifest valid; {checked} pinned files unchanged"


def verify_custody(paths: E07RIntegrityPaths) -> str:
    """Validate r5 completion, byte hashes, and ordered NPZ/Parquet binding."""
    manifest_path = paths.r5_dir / "stage2_custody_manifest.json"
    complete_path = paths.r5_dir / "STAGE2_CUSTODY_COMPLETE.json"
    manifest = _load_model(manifest_path, Stage2CustodyManifestV4)
    complete = _load_model(complete_path, Stage2CustodyCompleteV4)
    if complete.manifest_hash != manifest.manifest_hash:
        raise E07RIntegrityError("custody completion marker manifest mismatch")
    if complete.manifest_file_sha256 != sha256_file(manifest_path):
        raise E07RIntegrityError("custody manifest byte hash mismatch")
    expected_complete_hashes = {
        **manifest.output_file_sha256,
        "stage2_custody_manifest.json": sha256_file(manifest_path),
    }
    if complete.artifact_sha256 != expected_complete_hashes:
        raise E07RIntegrityError("custody completion artifact map mismatch")
    for name, expected in manifest.output_file_sha256.items():
        path = paths.r5_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise E07RIntegrityError(f"custody output hash mismatch: {name}")
        if path.stat().st_size != manifest.output_size_bytes[name]:
            raise E07RIntegrityError(f"custody output size mismatch: {name}")
    binding = ordered_row_binding_check(
        paths.r5_dir / "stage2_multiclass.npz",
        paths.r5_dir / "stage2_multiclass.parquet",
        scope="E07R_R5_PREFLIGHT",
    )
    if binding.status.value != "PASS":
        raise E07RIntegrityError("r5 ordered NPZ/Parquet binding failed")
    for name, expected in manifest.output_file_sha256.items():
        if sha256_file(paths.r5_dir / name) != expected:
            raise E07RIntegrityError(f"custody output changed during binding check: {name}")
    return f"r5 custody and ordered binding PASS for {manifest.row_count} rows"


def verify_patient_mapping(paths: E07RIntegrityPaths) -> str:
    """Validate complete authenticated mapping against the frozen r5 Parquet."""
    mapping = _load_model(paths.mapping, Stage2PatientMappingV4)
    parquet_hash = sha256_file(paths.r5_dir / "stage2_multiclass.parquet")
    if mapping.stage2_parquet_sha256 != parquet_hash:
        raise E07RIntegrityError("patient mapping is bound to a different Stage 2 parquet")
    if any(record.patient_id is None for record in mapping.records):
        raise E07RIntegrityError("confirmatory mapping contains unverified biological identity")
    known = {
        record.record_id: record.patient_id
        for record in mapping.records
        if record.dataset == "mitdb" and record.record_id in {"201", "202"}
    }
    if known != {"201": "mitdb:subject:201_202", "202": "mitdb:subject:201_202"}:
        raise E07RIntegrityError("MIT-BIH 201/202 authenticated group is not enforced")
    return (
        f"{mapping.record_count} records mapped to "
        f"{mapping.verified_patient_count} authenticated patients"
    )


def _partition_patients(
    mapping: Stage2PatientMappingV4,
    record_keys: Iterable[str],
) -> set[str]:
    by_key = {record.record_id: record.patient_id for record in mapping.records}
    if len(by_key) != len(mapping.records):
        raise E07RIntegrityError("record IDs are not globally unique in Stage 2 mapping")
    patients: set[str] = set()
    for key in record_keys:
        patient = by_key.get(key)
        if patient is None:
            raise E07RIntegrityError(f"split record lacks authenticated identity: {key}")
        patients.add(patient)
    return patients


def _safe_integral(value: Any, *, context: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise E07RIntegrityError(f"{context} is not an integer") from error
    if integer != value:
        raise E07RIntegrityError(f"{context} is not integral")
    return integer


def _verify_partition_against_rows(
    partition: PatientSplitPartitionV4,
    frame: pd.DataFrame,
    labels: np.ndarray,
    mapping: Stage2PatientMappingV4,
    *,
    context: str,
) -> None:
    indices = np.asarray(partition.indices, dtype=np.int64)
    if indices.size != len(set(indices.tolist())):
        raise E07RIntegrityError(f"{context} contains duplicate indices")
    if tuple(sorted(indices.tolist())) != partition.indices:
        raise E07RIntegrityError(f"{context} indices are not canonical and sorted")
    if indices.size and (
        _safe_integral(indices[0], context=f"{context} first index") < 0
        or _safe_integral(indices[-1], context=f"{context} last index") >= len(frame)
    ):
        raise E07RIntegrityError(f"{context} index is outside the frozen dataset")
    record_values = frame.iloc[indices]["record_id"].astype(str).to_numpy()
    records = tuple(sorted(set(record_values.tolist())))
    patient_by_record = {record.record_id: record.patient_id for record in mapping.records}
    if len(patient_by_record) != len(mapping.records):
        raise E07RIntegrityError("Stage 2 record IDs are not globally unique")
    try:
        patients = tuple(sorted({str(patient_by_record[record]) for record in record_values}))
    except KeyError as error:
        raise E07RIntegrityError(f"{context} references an unmapped record") from error
    label_names = np.asarray(("S", "V", "F"), dtype=object)
    partition_labels = labels[indices]
    if partition_labels.size and (
        _safe_integral(np.min(partition_labels), context=f"{context} minimum label") < 0
        or _safe_integral(np.max(partition_labels), context=f"{context} maximum label") > 2
    ):
        raise E07RIntegrityError(f"{context} contains an invalid class index")
    counts: dict[str, int] = dict.fromkeys(("S", "V", "F"), 0)
    for label, count in zip(
        *np.unique(partition_labels, return_counts=True),
        strict=True,
    ):
        label_index = _safe_integral(label, context=f"{context} class index")
        counts[str(label_names[label_index])] = _safe_integral(
            count,
            context=f"{context} class count",
        )
    f_mask = partition_labels == 2
    expected = {
        "patient_ids": patients,
        "record_ids": records,
        "class_counts": counts,
        "f_208": _safe_integral(
            np.sum(f_mask & (record_values == "208")),
            context=f"{context} F/208 count",
        ),
        "f_213": _safe_integral(
            np.sum(f_mask & (record_values == "213")),
            context=f"{context} F/213 count",
        ),
        "f_outside_208_213": _safe_integral(
            np.sum(f_mask & ~np.isin(record_values, ("208", "213"))),
            context=f"{context} F outside 208/213 count",
        ),
        "n_samples": len(indices),
        "n_patients": len(patients),
        "n_records": len(records),
        "indices_hash": hash_canonical(indices.tolist()),
        "patient_ids_hash": hash_canonical(patients),
        "record_ids_hash": hash_canonical(records),
    }
    for field, value in expected.items():
        if getattr(partition, field) != value:
            raise E07RIntegrityError(f"{context} row binding mismatch: {field}")


def _adapter_rows(document: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = document.get(key)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise E07RIntegrityError(f"split adapter {key} must be a JSON object list")
    return rows


def _adapter_partition(
    row: dict[str, Any],
    side: str,
    *,
    context: str,
) -> dict[str, Any]:
    value = row.get(side)
    if not isinstance(value, dict):
        raise E07RIntegrityError(f"{context} adapter partition {side} is invalid")
    return value


def verify_split_bundle(paths: E07RIntegrityPaths) -> str:
    """Validate all split contracts and independently recompute patient/record overlap."""
    split_dir = paths.split_dir
    mapping = _load_model(paths.mapping, Stage2PatientMappingV4)
    frame = pd.read_parquet(paths.r5_dir / "stage2_multiclass.parquet")
    with np.load(paths.r5_dir / "stage2_multiclass.npz", allow_pickle=False) as archive:
        labels = np.asarray(archive["y"], dtype=np.int64)
    if len(frame) != len(labels):
        raise E07RIntegrityError("split source NPZ/Parquet row counts differ")
    parquet_labels = frame["y"].to_numpy(dtype=np.int64)
    if not np.array_equal(labels, parquet_labels):
        raise E07RIntegrityError("split source NPZ/Parquet labels differ")
    outer = _load_model(split_dir / "outer_folds.json", PatientOuterFoldsV4)
    inner = _load_model(split_dir / "inner_folds.json", PatientInnerFoldsV4)
    leakage = _load_model(split_dir / "leakage_checks.json", SplitLeakageReportV4)
    statistics = _load_model(split_dir / "fold_statistics.json", FoldStatisticsV4)
    split_manifest = _load_model(
        split_dir / "split_manifest.json",
        PatientDisjointSplitManifestV4,
    )
    patient_groups = _load_model(split_dir / "patient_groups.json", Stage2PatientMappingV4)
    outer_adapter = _load_json(split_dir / "outer_splits_stage2.json")
    inner_adapter = _load_json(split_dir / "inner_splits_stage2.json")
    if not isinstance(outer_adapter, dict) or not isinstance(inner_adapter, dict):
        raise E07RIntegrityError("Stage 2 split adapters must be JSON objects")
    outer_adapter_hash = outer_adapter.get("manifest_hash")
    inner_adapter_hash = inner_adapter.get("manifest_hash")
    if (
        not isinstance(outer_adapter_hash, str)
        or hash_canonical(
            {key: value for key, value in outer_adapter.items() if key != "manifest_hash"}
        )
        != outer_adapter_hash
    ):
        raise E07RIntegrityError("outer Stage 2 adapter manifest hash mismatch")
    if (
        not isinstance(inner_adapter_hash, str)
        or hash_canonical(
            {key: value for key, value in inner_adapter.items() if key != "manifest_hash"}
        )
        != inner_adapter_hash
    ):
        raise E07RIntegrityError("inner Stage 2 adapter manifest hash mismatch")
    expected = {
        "patient_mapping_hash": mapping.mapping_hash,
        "outer_manifest_hash": outer.manifest_hash,
        "inner_manifest_hash": inner.manifest_hash,
        "leakage_report_hash": leakage.report_hash,
        "fold_statistics_hash": statistics.statistics_hash,
        "stage2_outer_adapter_hash": outer_adapter_hash,
        "stage2_inner_adapter_hash": inner_adapter_hash,
    }
    for field, value in expected.items():
        if getattr(split_manifest, field) != value:
            raise E07RIntegrityError(f"split manifest component mismatch: {field}")
    if patient_groups.mapping_hash != mapping.mapping_hash:
        raise E07RIntegrityError("split patient_groups differs from canonical mapping")
    all_mapping_records = {record.record_id for record in mapping.records}
    expected_indices = set(range(len(frame)))
    outer_adapter_rows = _adapter_rows(outer_adapter, "outer_folds")
    if len(outer_adapter_rows) != len(outer.folds):
        raise E07RIntegrityError("outer adapter fold count mismatch")
    outer_adapter_by_fold = {
        _safe_integral(row.get("fold"), context="outer adapter fold"): row
        for row in outer_adapter_rows
    }
    for outer_fold in outer.folds:
        train_records = set(outer_fold.train.record_ids)
        test_records = set(outer_fold.test.record_ids)
        train_indices = set(outer_fold.train.indices)
        test_indices = set(outer_fold.test.indices)
        if train_records & test_records or train_indices & test_indices:
            raise E07RIntegrityError(f"outer fold {outer_fold.fold} has overlap")
        if train_records | test_records != all_mapping_records:
            raise E07RIntegrityError(
                f"outer fold {outer_fold.fold} does not partition all mapped records"
            )
        if train_indices | test_indices != expected_indices:
            raise E07RIntegrityError(
                f"outer fold {outer_fold.fold} does not partition every dataset row"
            )
        _verify_partition_against_rows(
            outer_fold.train,
            frame,
            labels,
            mapping,
            context=f"outer {outer_fold.fold} train",
        )
        _verify_partition_against_rows(
            outer_fold.test,
            frame,
            labels,
            mapping,
            context=f"outer {outer_fold.fold} test",
        )
        if _partition_patients(mapping, train_records) & _partition_patients(mapping, test_records):
            raise E07RIntegrityError(f"outer fold {outer_fold.fold} has patient overlap")
        pair_locations = [key in train_records for key in ("201", "202")]
        if len(set(pair_locations)) != 1:
            raise E07RIntegrityError(f"outer fold {outer_fold.fold} separates MITDB 201/202")
        adapter_row = outer_adapter_by_fold.get(outer_fold.fold)
        if adapter_row is None:
            raise E07RIntegrityError(f"outer adapter omits fold {outer_fold.fold}")
        adapter_train = _adapter_partition(
            adapter_row,
            "train",
            context=f"outer {outer_fold.fold}",
        )
        adapter_test = _adapter_partition(
            adapter_row,
            "test",
            context=f"outer {outer_fold.fold}",
        )
        if adapter_train.get("indices") != list(outer_fold.train.indices):
            raise E07RIntegrityError(f"outer fold {outer_fold.fold} train adapter differs")
        if adapter_test.get("indices") != list(outer_fold.test.indices):
            raise E07RIntegrityError(f"outer fold {outer_fold.fold} test adapter differs")
        if adapter_train.get("groups") != list(outer_fold.train.patient_ids):
            raise E07RIntegrityError(f"outer fold {outer_fold.fold} train groups differ")
        if adapter_test.get("groups") != list(outer_fold.test.patient_ids):
            raise E07RIntegrityError(f"outer fold {outer_fold.fold} test groups differ")
    outer_train = {outer_fold.fold: set(outer_fold.train.record_ids) for outer_fold in outer.folds}
    selected_inner = {
        inner_fold.outer_fold: inner_fold
        for inner_fold in inner.folds
        if inner_fold.selected_for_training
    }
    if set(selected_inner) != set(range(1, len(outer.folds) + 1)):
        raise E07RIntegrityError("selected inner-fold coverage is incomplete")
    inner_adapter_rows = _adapter_rows(inner_adapter, "inner_folds")
    inner_adapter_by_fold = {
        _safe_integral(row.get("fold"), context="inner adapter fold"): row
        for row in inner_adapter_rows
    }
    if set(inner_adapter_by_fold) != set(selected_inner):
        raise E07RIntegrityError("inner adapter does not contain exactly selected folds")
    for inner_fold in inner.folds:
        train_records = set(inner_fold.train.record_ids)
        validation_records = set(inner_fold.validation.record_ids)
        if train_records & validation_records:
            raise E07RIntegrityError(
                f"inner fold {inner_fold.outer_fold}/{inner_fold.inner_fold} " "has record overlap"
            )
        if train_records | validation_records != outer_train[inner_fold.outer_fold]:
            raise E07RIntegrityError("inner split does not partition its outer training fold")
        _verify_partition_against_rows(
            inner_fold.train,
            frame,
            labels,
            mapping,
            context=f"inner {inner_fold.outer_fold}/{inner_fold.inner_fold} train",
        )
        _verify_partition_against_rows(
            inner_fold.validation,
            frame,
            labels,
            mapping,
            context=f"inner {inner_fold.outer_fold}/{inner_fold.inner_fold} validation",
        )
        if _partition_patients(mapping, train_records) & _partition_patients(
            mapping, validation_records
        ):
            raise E07RIntegrityError(
                f"inner fold {inner_fold.outer_fold}/{inner_fold.inner_fold} " "has patient overlap"
            )
    for outer_fold_number, selected_fold in selected_inner.items():
        adapter_row = inner_adapter_by_fold[outer_fold_number]
        adapter_train = _adapter_partition(
            adapter_row,
            "train",
            context=f"inner selected outer {outer_fold_number}",
        )
        adapter_validation = _adapter_partition(
            adapter_row,
            "validation",
            context=f"inner selected outer {outer_fold_number}",
        )
        if adapter_train.get("indices") != list(selected_fold.train.indices):
            raise E07RIntegrityError(f"inner outer {outer_fold_number} train adapter differs")
        if adapter_validation.get("indices") != list(selected_fold.validation.indices):
            raise E07RIntegrityError(f"inner outer {outer_fold_number} validation adapter differs")
        if adapter_train.get("groups") != list(selected_fold.train.patient_ids):
            raise E07RIntegrityError(f"inner outer {outer_fold_number} train groups differ")
        if adapter_validation.get("groups") != list(selected_fold.validation.patient_ids):
            raise E07RIntegrityError(f"inner outer {outer_fold_number} validation groups differ")
    if leakage.status != "PASS" or not leakage.patient_disjoint or not leakage.record_disjoint:
        raise E07RIntegrityError("frozen leakage report is not PASS")
    if leakage.structural_zero_folds or leakage.low_support_folds:
        raise E07RIntegrityError("frozen split bundle contains invalid-support folds")
    for row in statistics.rows:
        if any(value <= 0 for value in row.test_class_counts.values()):
            raise E07RIntegrityError(f"outer fold {row.fold} has structural zero")
    return (
        f"{len(outer.folds)} outer and {len(inner.folds)} inner folds are "
        "patient-disjoint and record-disjoint"
    )


def verify_quarantine(paths: E07RIntegrityPaths) -> str:
    """Validate additive quarantine while proving legacy bytes remain unchanged."""
    manifest = _load_model(
        paths.quarantine_manifest,
        LegacySplitQuarantineManifestV4,
    )
    if manifest.active_for_e07r:
        raise E07RIntegrityError("legacy quarantine is active for E07R")
    for artifact in manifest.artifacts:
        path = paths.project_root / artifact.artifact_path
        if not path.is_file() or sha256_file(path) != artifact.sha256:
            raise E07RIntegrityError(f"quarantined legacy split changed: {artifact.artifact_path}")
    return f"{len(manifest.artifacts)} legacy split artifacts quarantined, not deleted"


def verify_pd_protocol(paths: E07RIntegrityPaths) -> str:
    """Bind current PD execution code, data, split, mapping, config, and gates."""
    protocol = _load_model(paths.pd_protocol_manifest, E07RPDProtocolManifestV4)
    for relative, expected in protocol.source_file_sha256.items():
        path = paths.project_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise E07RIntegrityError(f"PD protocol source mismatch: {relative}")
    expected_files = {
        paths.r5_dir / "stage2_multiclass.npz": protocol.stage2_npz_sha256,
        paths.r5_dir / "stage2_multiclass.parquet": protocol.stage2_parquet_sha256,
        paths.project_root
        / "data/features/v3.1.0-r4/finetuning_mitbih_family.npz": protocol.full_npz_sha256,
        paths.project_root
        / "data/features/v3.1.0-r4/finetuning_mitbih_family.parquet": protocol.full_parquet_sha256,
        paths.project_root / "config/stage2_research.yaml": protocol.base_config_sha256,
    }
    for path, expected in expected_files.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise E07RIntegrityError(f"PD protocol input mismatch: {path}")
    mapping = _load_model(paths.mapping, Stage2PatientMappingV4)
    split = _load_model(
        paths.split_dir / "split_manifest.json",
        PatientDisjointSplitManifestV4,
    )
    if protocol.patient_mapping_hash != mapping.mapping_hash:
        raise E07RIntegrityError("PD protocol mapping link mismatch")
    if protocol.split_manifest_hash != split.manifest_hash:
        raise E07RIntegrityError("PD protocol split link mismatch")
    return f"PD protocol source/data/gates PASS ({protocol.manifest_hash})"


def verify_evidence_completion(paths: E07RIntegrityPaths) -> str:
    """Validate the final cross-directory marker and additive producer attestation."""
    producer = _load_model(paths.producer_attestation, E07RProducerAttestationV4)
    completion = _load_model(paths.evidence_complete, E07REvidenceCompleteV4)
    pd_protocol = _load_model(
        paths.pd_protocol_manifest,
        E07RPDProtocolManifestV4,
    )
    custody = _load_model(
        paths.r5_dir / "stage2_custody_manifest.json",
        Stage2CustodyManifestV4,
    )
    mapping = _load_model(paths.mapping, Stage2PatientMappingV4)
    split = _load_model(
        paths.split_dir / "split_manifest.json",
        PatientDisjointSplitManifestV4,
    )
    leakage = _load_model(
        paths.split_dir / "leakage_checks.json",
        SplitLeakageReportV4,
    )
    quarantine = _load_model(
        paths.quarantine_manifest,
        LegacySplitQuarantineManifestV4,
    )
    preauthorization = _load_model(
        paths.preauthorization_manifest,
        E07RPreauthorizationManifestV4,
    )
    expected_links = {
        "custody_manifest_hash": custody.manifest_hash,
        "patient_mapping_hash": mapping.mapping_hash,
        "split_manifest_hash": split.manifest_hash,
        "leakage_report_hash": leakage.report_hash,
        "quarantine_manifest_hash": quarantine.manifest_hash,
        "preauthorization_manifest_hash": preauthorization.manifest_hash,
        "pd_protocol_manifest_hash": pd_protocol.manifest_hash,
        "producer_attestation_hash": producer.attestation_hash,
    }
    for field, expected in expected_links.items():
        if getattr(completion, field) != expected:
            raise E07RIntegrityError(f"evidence completion link mismatch: {field}")
    if producer.custody_manifest_hash != custody.manifest_hash:
        raise E07RIntegrityError("producer attestation custody link mismatch")
    for relative, expected in producer.producer_file_sha256.items():
        path = paths.project_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise E07RIntegrityError(f"producer attestation file mismatch: {relative}")
    for relative, expected in completion.artifact_file_sha256.items():
        path = paths.project_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise E07RIntegrityError(f"evidence completion artifact mismatch: {relative}")
    return (
        f"cross-directory evidence COMPLETE with "
        f"{len(completion.artifact_file_sha256)} byte pins"
    )


def _capture_check(code: str, function: Callable[[], str]) -> E07RIntegrityCheckV4:
    try:
        evidence = function()
    except Exception as error:  # fail closed and preserve all check evidence
        return E07RIntegrityCheckV4(
            code=code,
            status="BLOCKED",
            evidence=f"{type(error).__name__}: {error}",
        )
    return E07RIntegrityCheckV4(code=code, status="PASS", evidence=evidence)


def run_e07r_preflight(
    project_root: Path,
    *,
    workflow: Literal["FREEZE_VALIDATION", "E06_5_PD", "E07_PD"],
    run_id: str,
) -> E07RPreflightReportV4:
    """Run the mandatory fail-closed E07R preflight without mutating inputs."""
    paths = E07RIntegrityPaths(project_root.resolve())
    checks: tuple[E07RIntegrityCheckV4, ...]
    try:
        freeze = _load_model(paths.freeze_manifest, E07RFreezeManifestV4)
    except E07RIntegrityError as error:
        checks = (
            E07RIntegrityCheckV4(
                code="FREEZE_MANIFEST",
                status="BLOCKED",
                evidence=str(error),
            ),
        )
        freeze_hash = "0" * 64
    else:
        freeze_hash = freeze.manifest_hash
        checks = (
            _capture_check(
                "FREEZE_INVENTORY_LINKS",
                lambda: verify_freeze_inventory_and_links(paths, freeze),
            ),
            _capture_check(
                "FREEZE_PINS",
                lambda: verify_freeze_pins(paths.project_root, freeze),
            ),
            _capture_check("PREAUTHORIZATION", lambda: verify_preauthorization(paths)),
            _capture_check("CUSTODY_BINDING", lambda: verify_custody(paths)),
            _capture_check("PATIENT_MAPPING", lambda: verify_patient_mapping(paths)),
            _capture_check("SPLIT_BUNDLE", lambda: verify_split_bundle(paths)),
            _capture_check("LEGACY_QUARANTINE", lambda: verify_quarantine(paths)),
            _capture_check("PD_PROTOCOL", lambda: verify_pd_protocol(paths)),
            _capture_check(
                "EVIDENCE_COMPLETION",
                lambda: verify_evidence_completion(paths),
            ),
        )
    status = "PASS" if all(check.status == "PASS" for check in checks) else "BLOCKED"
    payload = {
        "schema_version": "e07r-preflight-v4.0",
        "stage": "E07R",
        "workflow": workflow,
        "run_id": run_id,
        "split_version": "v4.0-patient-disjoint",
        "freeze_manifest_hash": freeze_hash,
        "checks": [check.model_dump(mode="json") for check in checks],
        "status": status,
    }
    return E07RPreflightReportV4.model_validate({**payload, "report_hash": hash_canonical(payload)})


def require_e07r_preflight(
    project_root: Path,
    *,
    workflow: Literal["E06_5_PD", "E07_PD"],
    run_id: str,
) -> E07RPreflightReportV4:
    """Publish a write-once preflight report and block the workflow on any failure."""
    report = run_e07r_preflight(project_root, workflow=workflow, run_id=run_id)
    path = project_root / (
        "experiments/stage2_v2.4_research/integrity/preflight/" f"{workflow.lower()}_{run_id}.json"
    )
    if path.exists():
        stored = _load_model(path, E07RPreflightReportV4)
        if stored != report:
            raise E07RIntegrityError(f"preflight evidence drift for run_id={run_id}")
        report = stored
    else:
        write_json_exclusive(path, report.model_dump(mode="json"))
    if report.status != "PASS":
        raise E07RIntegrityError(f"{workflow} blocked by E07R preflight; evidence={path}")
    return report


def _append_violation(paths: E07RIntegrityPaths, event: E07RIntegrityViolationV4) -> None:
    path = paths.violation_log
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (
        json.dumps(
            event.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "a", encoding="utf-8", closefd=False) as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _record_violation(
    paths: E07RIntegrityPaths,
    *,
    event_type: Literal[
        "FORBIDDEN_WRITE",
        "LEGACY_SPLIT_USE",
        "MODEL_PROMOTION_ATTEMPT",
        "FROZEN_ARTIFACT_MUTATION",
    ],
    workflow: str,
    run_id: str,
    attempted_path: Path,
    reason: str,
    freeze_manifest_hash: str,
) -> None:
    payload = {
        "schema_version": "e07r-integrity-violation-v4.0",
        "event_type": event_type,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": workflow,
        "run_id": run_id,
        "attempted_path": _relative(paths.project_root, attempted_path),
        "reason": reason,
        "freeze_manifest_hash": freeze_manifest_hash,
    }
    event = E07RIntegrityViolationV4.model_validate(
        {**payload, "event_hash": hash_canonical(payload)}
    )
    _append_violation(paths, event)


def _is_descendant(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def guard_e07r_write(
    project_root: Path,
    candidate: Path,
    *,
    workflow: str,
    run_id: str,
) -> None:
    """Reject writes outside PD namespaces, especially models and frozen inputs."""
    paths = E07RIntegrityPaths(project_root.resolve())
    freeze = _load_model(paths.freeze_manifest, E07RFreezeManifestV4)
    target = candidate.resolve(strict=False)
    models_root = (paths.project_root / "models").resolve()
    if _is_descendant(target, models_root):
        _record_violation(
            paths,
            event_type="MODEL_PROMOTION_ATTEMPT",
            workflow=workflow,
            run_id=run_id,
            attempted_path=target,
            reason="model promotion is not authorized for E07R",
            freeze_manifest_hash=freeze.manifest_hash,
        )
        raise E07RIntegrityError("E07R model promotion attempt blocked")
    protected = {
        (paths.project_root / pin.artifact_path).resolve()
        for pin in freeze.pins
        if pin.enforce_read_only
    }
    if target in protected:
        _record_violation(
            paths,
            event_type="FORBIDDEN_WRITE",
            workflow=workflow,
            run_id=run_id,
            attempted_path=target,
            reason="write targets a frozen E07R artifact",
            freeze_manifest_hash=freeze.manifest_hash,
        )
        raise E07RIntegrityError("write to frozen E07R artifact blocked")
    allowed_roots = (
        paths.project_root / "experiments/stage2_v2.4_research/E06_5_PD",
        paths.project_root / "experiments/stage2_v2.4_research/E07_PD",
        paths.project_root / "experiments/stage2_v2.4_research/cache_pd",
        paths.project_root / "experiments/stage2_v2.4_research/integrity/preflight",
    )
    if not any(_is_descendant(target, root.resolve()) for root in allowed_roots):
        _record_violation(
            paths,
            event_type="FORBIDDEN_WRITE",
            workflow=workflow,
            run_id=run_id,
            attempted_path=target,
            reason="write is outside authorized patient-disjoint namespaces",
            freeze_manifest_hash=freeze.manifest_hash,
        )
        raise E07RIntegrityError("write outside E07R PD namespace blocked")


def assert_authorized_split_path(
    project_root: Path,
    candidate: Path,
    *,
    workflow: str,
    run_id: str,
) -> None:
    """Reject any accidental use of quarantined record-disjoint split artifacts."""
    paths = E07RIntegrityPaths(project_root.resolve())
    freeze = _load_model(paths.freeze_manifest, E07RFreezeManifestV4)
    target = candidate.resolve(strict=False)
    split_root = paths.split_dir.resolve()
    if target == split_root or _is_descendant(target, split_root):
        return
    _record_violation(
        paths,
        event_type="LEGACY_SPLIT_USE",
        workflow=workflow,
        run_id=run_id,
        attempted_path=target,
        reason="only v4.0-patient-disjoint splits are authorized",
        freeze_manifest_hash=freeze.manifest_hash,
    )
    raise E07RIntegrityError("legacy or unauthorized split use blocked")
