"""Exact input, source, environment, and generation manifest construction."""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .contracts import (
    AdvancedTrainingConfig,
    FileManifest,
    PatientIdentityManifest,
    PatientIdentityPolicy,
    PatientSplitManifest,
    TrainingGenerationManifest,
)
from .integrity import (
    build_file_manifest,
    hash_canonical,
    resolve_project_path,
    sha256_file,
)


def _policy_by_dataset(policy: PatientIdentityPolicy) -> dict[str, Any]:
    return {dataset.dataset_id: dataset for dataset in policy.datasets}


def _wfdb_record_files(header: Path) -> tuple[Path, ...]:
    try:
        lines = header.read_text(encoding="utf-8", errors="strict").splitlines()
        if not lines:
            raise ValueError("empty header")
        first = lines[0].split()
        if len(first) < 2:
            raise ValueError("invalid first header line")
        signal_count = int(first[1])
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid WFDB header: {header}") from error
    if signal_count < 1 or len(lines) < signal_count + 1:
        raise ValueError(f"invalid WFDB signal count: {header}")
    signal_paths: set[Path] = set()
    for line in lines[1 : signal_count + 1]:
        tokens = line.split()
        if not tokens:
            raise ValueError(f"invalid WFDB signal line: {header}")
        signal_paths.add(header.parent / tokens[0])
    return tuple(sorted({header, *signal_paths}))


def build_raw_annotation_manifests(
    project_root: Path,
    identity: PatientIdentityManifest,
    policy: PatientIdentityPolicy,
) -> tuple[FileManifest, FileManifest]:
    """Hash exact WFDB headers/signals separately from consumed annotations."""
    policies = _policy_by_dataset(policy)
    raw_paths: set[Path] = set()
    annotation_paths: set[Path] = set()
    for record in identity.records:
        dataset_policy = policies.get(record.dataset_id)
        if dataset_policy is None:
            raise ValueError(f"identity record has no dataset policy: {record.dataset_id}")
        raw_dir = resolve_project_path(project_root, dataset_policy.raw_dir)
        header = raw_dir / f"{record.record_id}.hea"
        raw_paths.update(_wfdb_record_files(header))
        annotation_paths.add(raw_dir / f"{record.record_id}.atr")
    return (
        build_file_manifest(project_root, raw_paths, category="raw-data"),
        build_file_manifest(project_root, annotation_paths, category="annotations"),
    )


def _source_paths(project_root: Path, globs: tuple[str, ...]) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for pattern in globs:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError(f"unsafe source glob: {pattern}")
        matches = [path for path in project_root.glob(pattern) if path.is_file()]
        if not matches:
            raise ValueError(f"source glob matched no files: {pattern}")
        paths.update(matches)
    return tuple(sorted(paths))


def _git_payload(
    project_root: Path,
    source_paths: tuple[str, ...],
) -> dict[str, Any]:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise ValueError("git executable is unavailable")

    def run(*args: str) -> bytes:
        try:
            return subprocess.run(
                [git_executable, *args],
                cwd=project_root,
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(f"cannot resolve Git identity: {' '.join(args)}") from error

    head = run("rev-parse", "HEAD").decode("ascii").strip()
    tree = run("rev-parse", "HEAD^{tree}").decode("ascii").strip()
    status = run("status", "--porcelain=v1", "--", *source_paths)
    diff = run("diff", "--binary", "HEAD", "--", *source_paths)
    return {
        "git_head": head,
        "git_tree": tree,
        "git_dirty": bool(status.strip()),
        "git_status_sha256": hashlib.sha256(status).hexdigest(),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _environment_payload(project_root: Path) -> dict[str, Any]:
    packages = {}
    for distribution in (
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scipy",
        "tensorflow",
        "wfdb",
        "pydantic",
    ):
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = "NOT_INSTALLED"
    uv_lock = project_root / "uv.lock"
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "uv_lock_sha256": sha256_file(uv_lock),
    }


def build_training_generation_manifest(
    *,
    project_root: Path,
    config_path: Path,
    config: AdvancedTrainingConfig,
    identity: PatientIdentityManifest,
    split: PatientSplitManifest,
    policy: PatientIdentityPolicy,
) -> tuple[TrainingGenerationManifest, dict[str, Any]]:
    """Build the required ten-hash generation identity from exact bytes."""
    raw_manifest, annotation_manifest = build_raw_annotation_manifests(
        project_root, identity, policy
    )
    processed_paths = tuple(
        resolve_project_path(project_root, value)
        for value in (
            config.family_npz,
            config.family_parquet,
            config.stage1_npz,
            config.stage1_parquet,
            config.afdb_rhythm_npz,
            config.afdb_rhythm_parquet,
        )
    ) + _source_paths(project_root, config.processed_signal_globs)
    processed_manifest = build_file_manifest(
        project_root, processed_paths, category="processed-data"
    )
    preprocessing_paths = tuple(
        resolve_project_path(project_root, value) for value in config.preprocessing_files
    )
    preprocessing_manifest = build_file_manifest(
        project_root, preprocessing_paths, category="preprocessing"
    )
    source_manifest = build_file_manifest(
        project_root,
        _source_paths(project_root, config.source_globs),
        category="research-source",
    )
    git_payload = _git_payload(
        project_root,
        tuple(file.project_relative_path for file in source_manifest.files),
    )
    source_payload = {
        "git": git_payload,
        "files": source_manifest.model_dump(mode="json"),
    }
    environment_payload = _environment_payload(project_root)
    feature_schema_payload = {
        "input_shape": list(config.input_shape),
        "target_sampling_rate": config.target_sampling_rate,
        "feature_columns": list(config.feature_columns),
        "quality_heads": list(config.quality_heads),
    }
    ontology_path = project_root / "src" / "features" / "ontology_v3.py"
    ontology_manifest = build_file_manifest(
        project_root, (ontology_path,), category="ontology-source"
    )
    training_config_manifest = build_file_manifest(
        project_root, (config_path,), category="training-config"
    )
    generation = TrainingGenerationManifest(
        schema_version="training-generation-v3.1.0",
        generation_id=config.generation_id,
        raw_data_hash=raw_manifest.payload_hash,
        annotation_hash=annotation_manifest.payload_hash,
        processed_data_hash=processed_manifest.payload_hash,
        ontology_hash=ontology_manifest.files[0].sha256,
        preprocessing_hash=preprocessing_manifest.payload_hash,
        feature_schema_hash=hash_canonical("feature-schema-v3.1.0", feature_schema_payload),
        patient_split_hash=hash_canonical("patient-split", split),
        training_config_hash=training_config_manifest.files[0].sha256,
        source_revision=hash_canonical("research-source-snapshot", source_payload),
        environment_hash=hash_canonical("research-environment", environment_payload),
        research_execution_authorized=config.research_execution_authorized,
        promotion_authorized=False,
    )
    evidence = {
        "raw_data_manifest": raw_manifest.model_dump(mode="json"),
        "annotation_manifest": annotation_manifest.model_dump(mode="json"),
        "processed_data_manifest": processed_manifest.model_dump(mode="json"),
        "preprocessing_manifest": preprocessing_manifest.model_dump(mode="json"),
        "source_manifest": source_manifest.model_dump(mode="json"),
        "ontology_manifest": ontology_manifest.model_dump(mode="json"),
        "training_config_manifest": training_config_manifest.model_dump(mode="json"),
        "git": git_payload,
        "environment": environment_payload,
        "feature_schema": feature_schema_payload,
    }
    return generation, evidence
