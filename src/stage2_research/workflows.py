"""Canonical preflight, plan, execution, resume, and status workflows."""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from src.stage2_research.config import config_hash
from src.stage2_research.contracts import (
    ExitCode,
    InnerSplitManifest,
    ProfileName,
    ResearchConfig,
    ResearchError,
    RunCell,
    SplitManifest,
)
from src.stage2_research.data import (
    FullTemplateDataset,
    Stage2Dataset,
    load_full_template_dataset,
    load_stage2_dataset,
)
from src.stage2_research.features import (
    FeatureBundle,
    build_feature_bundle,
    freeze_static_feature_manifests,
)
from src.stage2_research.integrity import (
    atomic_write_json,
    atomic_write_text,
    collect_environment_without_reseeding,
    git_identity,
    hash_canonical,
    load_json,
    runtime_identity_hash,
    sha256_file,
    source_fingerprint,
    utc_now,
    validate_done_marker,
    validate_project_output_root,
)
from src.stage2_research.splits import freeze_or_validate_splits, split_indices
from src.stage2_research.training import (
    stage_run_dir,
    train_e06_cell,
)
from src.stage2_research.validation import matches_bool as _matches_bool
from src.stage2_research.validation import safe_float as _safe_float
from src.stage2_research.validation import validate_template_source_groups


@dataclass(frozen=True)
class PreparedResearch:
    """Validated immutable inputs shared by commands."""

    config: ResearchConfig
    dataset: Stage2Dataset
    full: FullTemplateDataset
    outer_splits: SplitManifest
    inner_splits: InnerSplitManifest
    feature_manifests: dict[str, dict[str, Any]]


def _all_json_numbers_finite(value: Any) -> bool:
    """Recursively reject NaN/Inf while preserving non-numeric metadata."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value))
    if isinstance(value, dict):
        return all(_all_json_numbers_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_json_numbers_finite(item) for item in value)
    return True


def _history_numeric_contract_valid(history: pd.DataFrame) -> bool:
    """Allow structurally absent outer validation metrics, never invalid measurements."""
    required = {"phase", "epoch", "accuracy", "loss", "val_accuracy", "val_loss"}
    if not required <= set(history.columns):
        return False
    core = history.loc[:, ["epoch", "accuracy", "loss"]].to_numpy(dtype=np.float64)
    validation_rows = history["phase"].astype(str).str.startswith("inner")
    validation = history.loc[
        validation_rows,
        ["val_accuracy", "val_loss"],
    ].to_numpy(dtype=np.float64)
    all_numeric = history.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    return (
        np.isfinite(core).all()
        and np.isfinite(validation).all()
        and not np.isinf(all_numeric).any()
    )


def prepare_research(config: ResearchConfig) -> PreparedResearch:
    validate_project_output_root(config.project_root, config.output_root)
    dataset = load_stage2_dataset(config)
    full = load_full_template_dataset(config)
    outer, inner, _ = freeze_or_validate_splits(config, dataset)
    feature_manifests_typed = freeze_static_feature_manifests(config)
    feature_manifests = {str(name): manifest for name, manifest in feature_manifests_typed.items()}
    return PreparedResearch(
        config=config,
        dataset=dataset,
        full=full,
        outer_splits=outer,
        inner_splits=inner,
        feature_manifests=feature_manifests,
    )


def _command_slug(command: Sequence[str], index: int) -> str:
    executable = Path(command[0]).name if command else "empty"
    return f"{index:02d}_{executable}"


def _run_validation_commands(config: ResearchConfig) -> list[dict[str, Any]]:
    output_dir = config.output_root / "reports" / "preflight_commands"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, command in enumerate(config.validation_commands, start=1):
        slug = _command_slug(command, index)
        try:
            process = subprocess.run(
                list(command),
                cwd=config.project_root,
                text=True,
                capture_output=True,
                timeout=1800,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ResearchError(
                f"preflight validation command failed to execute: {' '.join(command)}",
                ExitCode.BLOCKED_PRECONDITION,
            ) from error
        atomic_write_text(output_dir / f"{slug}.stdout.log", process.stdout)
        atomic_write_text(output_dir / f"{slug}.stderr.log", process.stderr)
        results.append(
            {
                "command": list(command),
                "exit_code": process.returncode,
                "passed": process.returncode == 0,
                "stdout_log": str(output_dir / f"{slug}.stdout.log"),
                "stderr_log": str(output_dir / f"{slug}.stderr.log"),
            }
        )
    return results


def _preflight_markdown(report: dict[str, Any]) -> str:
    candidates = ", ".join(report["matrix"]["candidates"])
    folds = ",".join(str(item) for item in report["matrix"]["folds"])
    seeds = ",".join(str(item) for item in report["matrix"]["seeds"])
    checks = "\n".join(
        f"- [{'PASS' if item['passed'] else 'FAIL'}] {' '.join(item['command'])}"
        for item in report["validation_commands"]
    )
    return (
        "# Stage 2 research preflight\n\n"
        f"**Status:** {report['status']}\n\n"
        f"- candidates: {candidates}\n"
        f"- folds: {folds}\n"
        f"- seeds: {seeds}\n"
        f"- planned runs: {report['matrix']['planned_runs']}\n"
        f"- estimated artifacts: {report['resources']['estimated_artifacts_mib']:.1f} MiB\n"
        f"- outer overlap: {report['integrity']['outer_overlap_count']}\n"
        f"- template leakage: {report['integrity']['template_leakage_count']}\n"
        f"- split hash: `{report['integrity']['outer_split_manifest_hash']}`\n"
        f"- dataset hash: `{report['integrity']['dataset_manifest_hash']}`\n\n"
        "## Validation commands\n\n"
        f"{checks or '- none configured'}\n"
    )


def run_preflight(
    config: ResearchConfig,
    *,
    deterministic: bool,
    device: str,
    dry_run: bool = False,
) -> tuple[PreparedResearch, dict[str, Any]]:
    """Validate runtime, immutable data, splits, features, leakage, and resources."""
    if dry_run:
        raise ResearchError(
            "preflight dry-run cannot freeze required split manifests",
            ExitCode.BLOCKED_PRECONDITION,
        )
    if not deterministic or device != "cpu":
        raise ResearchError(
            "canonical preflight requires deterministic CPU mode",
            ExitCode.INVALID_EXPERIMENT,
        )
    prepared = prepare_research(config)
    actual_uv_hash = sha256_file(config.project_root / "uv.lock")
    if actual_uv_hash != config.uv_lock_sha256:
        raise ResearchError("uv.lock hash mismatch", ExitCode.DATA_INTEGRITY)
    head, dirty = git_identity(config.project_root)
    source_identity = source_fingerprint(config.project_root)
    runtime = collect_environment_without_reseeding(
        deterministic=deterministic,
        device=device,
        split_random_state=config.split_contract.random_seed,
    )
    disk = shutil.disk_usage(config.output_root)
    free_gib = disk.free / (1024**3)
    if free_gib < config.resources.minimum_free_disk_gib:
        raise ResearchError(
            "insufficient free disk for Stage 2 research",
            ExitCode.BLOCKED_PRECONDITION,
            details={"free_gib": free_gib},
        )
    validation_results = _run_validation_commands(config)
    failed_commands = [item for item in validation_results if not item["passed"]]
    if failed_commands:
        raise ResearchError(
            "preflight validation commands failed",
            ExitCode.BLOCKED_PRECONDITION,
            details={"commands": failed_commands},
        )
    outer_overlap = sum(len(item.overlap_groups) for item in prepared.outer_splits.outer_folds)
    template_leakage = 0
    scaler_leakage = 0
    split_scope_evidence: list[dict[str, Any]] = []
    all_indices = np.arange(len(prepared.dataset.labels), dtype=np.int64)
    for fold in config.folds:
        outer_train, outer_test, inner_train, inner_val = split_indices(
            prepared.outer_splits,
            prepared.inner_splits,
            fold,
        )
        groups = prepared.dataset.groups.astype(str)
        outer_train_groups = set(groups[outer_train].tolist())
        outer_test_groups = set(groups[outer_test].tolist())
        inner_train_groups = set(groups[inner_train].tolist())
        inner_val_groups = set(groups[inner_val].tolist())
        template_overlap = len(outer_train_groups & outer_test_groups) + len(
            inner_train_groups & (inner_val_groups | outer_test_groups)
        )
        scaler_overlap = len(set(outer_train.tolist()) & set(outer_test.tolist())) + len(
            set(inner_train.tolist()) & set(inner_val.tolist())
        )
        partition_coverage = len(set(outer_train.tolist()) | set(outer_test.tolist()))
        if partition_coverage != len(all_indices):
            raise ResearchError("outer split coverage mismatch", ExitCode.DATA_INTEGRITY)
        template_leakage += template_overlap
        scaler_leakage += scaler_overlap
        split_scope_evidence.append(
            {
                "fold": fold,
                "template_source_group_overlap": template_overlap,
                "preprocessor_fit_index_overlap": scaler_overlap,
                "outer_partition_coverage": partition_coverage,
            }
        )
    planned_runs = len(config.candidates) * len(config.folds) * len(config.seeds)
    report: dict[str, Any] = {
        "schema_version": "stage2-preflight-v1",
        "status": "PREFLIGHT_PASS",
        "created_at": utc_now(),
        "config_hash": config_hash(config),
        "git": {"head": head, "dirty": dirty},
        "source_identity": source_identity,
        "source_manifest_hash": source_identity["source_manifest_hash"],
        "runtime": runtime,
        "runtime_identity_hash": runtime_identity_hash(runtime),
        "matrix": {
            "candidates": ["baseline", "H6", "H11", "H12"],
            "folds": list(config.folds),
            "seeds": list(config.seeds),
            "planned_runs": planned_runs,
        },
        "integrity": {
            "dataset_manifest_hash": prepared.dataset.manifest_hash,
            "outer_split_manifest_hash": prepared.outer_splits.manifest_hash,
            "inner_split_manifest_hash": prepared.inner_splits.manifest_hash,
            "feature_manifest_hashes": {
                name: str(value["manifest_hash"])
                for name, value in prepared.feature_manifests.items()
            },
            "uv_lock_hash": actual_uv_hash,
            "outer_overlap_count": outer_overlap,
            "template_leakage_count": template_leakage,
            "scaler_leakage_count": scaler_leakage,
            "early_stopping_source": "inner_validation",
            "outer_test_used_for_selection": False,
            "split_scope_evidence": split_scope_evidence,
            "leakage_evidence_source": "frozen split manifests and exact fit partitions",
        },
        "dataset": prepared.dataset.manifest,
        "full_template_dataset": prepared.full.manifest,
        "resources": {
            "free_disk_gib": free_gib,
            "estimated_mib_per_run": config.resources.estimated_mib_per_run,
            "estimated_artifacts_mib": planned_runs * config.resources.estimated_mib_per_run,
        },
        "validation_commands": validation_results,
        "warnings": ["Git worktree is dirty; identity is recorded in every run."] if dirty else [],
    }
    report["preflight_hash"] = hash_canonical(
        {key: value for key, value in report.items() if key != "created_at"}
    )
    report_path = config.output_root / "reports" / "preflight_report.json"
    atomic_write_json(report_path, report)
    atomic_write_text(
        config.output_root / "reports" / "preflight_report.md",
        _preflight_markdown(report),
    )
    atomic_write_json(config.output_root / "manifests" / "preflight.json", report)
    return prepared, report


def require_preflight(config: ResearchConfig) -> dict[str, Any]:
    """Require a matching successful preflight before any training."""
    path = config.output_root / "manifests" / "preflight.json"
    if not path.is_file():
        raise ResearchError("PREFLIGHT PASS is required", ExitCode.BLOCKED_PRECONDITION)
    report = cast(dict[str, Any], load_json(path))
    if report.get("status") != "PREFLIGHT_PASS":
        raise ResearchError("stored preflight did not pass", ExitCode.BLOCKED_PRECONDITION)
    if report.get("config_hash") != config_hash(config):
        raise ResearchError("preflight config hash mismatch", ExitCode.INCOMPATIBLE_ARTIFACT)
    expected_preflight_hash = hash_canonical(
        {key: value for key, value in report.items() if key not in {"created_at", "preflight_hash"}}
    )
    if report.get("preflight_hash") != expected_preflight_hash:
        raise ResearchError("preflight hash mismatch", ExitCode.INCOMPATIBLE_ARTIFACT)
    current_source = source_fingerprint(config.project_root)
    if report.get("source_manifest_hash") != current_source["source_manifest_hash"]:
        raise ResearchError(
            "preflight source identity mismatch; rerun preflight",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    stored_runtime = cast(dict[str, Any], report.get("runtime", {}))
    deterministic = _matches_bool(stored_runtime.get("deterministic_requested"), True)
    if not deterministic:
        raise ResearchError(
            "canonical preflight must be deterministic",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    current_runtime = collect_environment_without_reseeding(
        deterministic=True,
        device=str(stored_runtime.get("device_requested", "cpu")),
        split_random_state=config.split_contract.random_seed,
    )
    if report.get("runtime_identity_hash") != runtime_identity_hash(current_runtime):
        raise ResearchError(
            "preflight runtime identity mismatch; rerun preflight",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    return report


_E065_CANDIDATES = ("baseline", "H6", "H11", "H12")
_E065_SMOKE_FOLDS = (1,)
_E065_SMOKE_SEEDS = (17,)


def _require_e065_execution_contract(
    config: ResearchConfig,
    *,
    candidates: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
    profile_name: ProfileName,
    deterministic: bool,
    device: str,
    max_parallel: int,
) -> None:
    """Reject non-canonical E06.5 matrices before any cell is touched."""
    expected_matrix = {
        "smoke": (_E065_CANDIDATES, _E065_SMOKE_FOLDS, _E065_SMOKE_SEEDS),
        "audit": (_E065_CANDIDATES, config.folds, config.seeds),
    }
    if profile_name not in expected_matrix:
        raise ResearchError(
            "E06.5 supports only smoke and audit profiles",
            ExitCode.INVALID_EXPERIMENT,
        )
    expected_candidates, expected_folds, expected_seeds = expected_matrix[profile_name]
    if (
        tuple(candidates) != tuple(expected_candidates)
        or tuple(folds) != tuple(expected_folds)
        or tuple(seeds) != tuple(expected_seeds)
    ):
        raise ResearchError(
            f"non-canonical E06.5 {profile_name} matrix",
            ExitCode.INVALID_EXPERIMENT,
            details={
                "expected_candidates": list(expected_candidates),
                "expected_folds": list(expected_folds),
                "expected_seeds": list(expected_seeds),
            },
        )
    profile = config.profiles[profile_name]
    if (
        not deterministic
        or not profile.deterministic
        or max_parallel != 1
        or profile.max_parallel != 1
        or device != "cpu"
    ):
        raise ResearchError(
            f"E06.5 {profile_name} requires deterministic serial CPU execution",
            ExitCode.INVALID_EXPERIMENT,
        )


def verify_template_sources(
    prepared: PreparedResearch,
    bundle: FeatureBundle,
    *,
    fold: int,
    error_message: str = "template source leakage detected",
) -> None:
    """Prove persisted template sources stay within each training partition."""
    if not bundle.template_state:
        return
    outer_train, outer_test, inner_train, inner_val = split_indices(
        prepared.outer_splits,
        prepared.inner_splits,
        fold,
    )
    validate_template_source_groups(
        prepared.dataset.groups,
        outer_train=outer_train,
        outer_test=outer_test,
        inner_train=inner_train,
        inner_validation=inner_val,
        template_state=bundle.template_state,
        error_message=error_message,
    )


def _build_e065_smoke_gate(
    config: ResearchConfig,
    preflight: dict[str, Any],
    prepared: PreparedResearch,
    *,
    experiment_id: str,
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    """Derive the canonical smoke checkpoint from persisted cell evidence."""
    cells: dict[str, Any] = {}
    shape_checks = 0
    finite_checks = 0
    template_checks = 0
    scaler_checks = 0
    sampling_checks = 0
    method_checks = 0
    outer_selection_checks = 0
    reload_checks = 0
    for candidate in _E065_CANDIDATES:
        run_dir = stage_run_dir(
            config,
            stage="e06.5",
            experiment_id=experiment_id,
            candidate=candidate,
            fold=1,
            seed=17,
        )
        marker = validate_done_marker(run_dir)
        if marker is None:
            raise ResearchError("canonical smoke cell is incomplete", ExitCode.REGRESSION)
        manifest = cast(dict[str, Any], load_json(run_dir / "run_manifest.json"))
        metrics = cast(dict[str, Any], load_json(run_dir / "metrics.json"))
        environment = cast(dict[str, Any], load_json(run_dir / "environment.json"))
        preprocessing = cast(
            dict[str, Any],
            load_json(run_dir / "preprocessing_manifest.json"),
        )
        sampling = cast(dict[str, Any], load_json(run_dir / "sampling_manifest.json"))
        method = cast(dict[str, Any], load_json(run_dir / "method_manifest.json"))
        predictions = pd.read_parquet(run_dir / "predictions.parquet")
        bundle = build_feature_bundle(
            config,
            prepared.dataset,
            prepared.full,
            prepared.outer_splits,
            prepared.inner_splits,
            candidate_name=candidate,
            fold=1,
        )
        outer_train, outer_test, inner_train, _ = split_indices(
            prepared.outer_splits,
            prepared.inner_splits,
            1,
        )
        if len(predictions) != len(outer_test) or len(predictions.columns) == 0:
            raise ResearchError("smoke prediction shape mismatch", ExitCode.REGRESSION)
        shape_checks += 1
        numeric = predictions.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        history = pd.read_csv(run_dir / "training_history.csv")
        if (
            not np.isfinite(numeric).all()
            or not _history_numeric_contract_valid(history)
            or not _all_json_numbers_finite(metrics)
        ):
            raise ResearchError("smoke outputs contain NaN/Inf", ExitCode.REGRESSION)
        finite_checks += 1
        verify_template_sources(prepared, bundle, fold=1)
        template_checks += 1
        expected_index_hashes = {
            "inner_train_indices_hash": hash_canonical(inner_train.tolist()),
            "outer_train_indices_hash": hash_canonical(outer_train.tolist()),
            "outer_test_indices_hash": hash_canonical(outer_test.tolist()),
        }
        if any(preprocessing.get(key) != value for key, value in expected_index_hashes.items()):
            raise ResearchError("smoke preprocessing scope mismatch", ExitCode.LEAKAGE)
        if not _matches_bool(preprocessing.get("outer_test_used_for_fit"), False):
            raise ResearchError("outer test used for preprocessing", ExitCode.LEAKAGE)
        scaler_checks += 1
        expected_train_hashes = {
            "inner": hash_canonical(inner_train.tolist()),
            "outer": hash_canonical(outer_train.tolist()),
        }
        for partition, expected_hash in expected_train_hashes.items():
            sample_state = cast(dict[str, Any], sampling.get(partition, {}))
            if (
                sample_state.get("input_partition_index_hash") != expected_hash
                or sample_state.get("source_outside_partition_count") != 0
                or not _matches_bool(
                    sample_state.get("validation_or_test_sampled"),
                    False,
                )
            ):
                raise ResearchError("smoke sampler scope mismatch", ExitCode.LEAKAGE)
        sampling_checks += 1
        if (
            method.get("inner_fit_partition_index_hash") != expected_train_hashes["inner"]
            or method.get("outer_fit_partition_index_hash") != expected_train_hashes["outer"]
            or not _matches_bool(method.get("outer_test_used_for_method_fit"), False)
        ):
            raise ResearchError("smoke method fit scope mismatch", ExitCode.LEAKAGE)
        method_checks += 1
        if not _matches_bool(
            manifest.get("outer_test_used_for_selection"), False
        ) or not _matches_bool(preprocessing.get("outer_test_used_for_selection"), False):
            raise ResearchError("outer test used for model selection", ExitCode.LEAKAGE)
        outer_selection_checks += 1
        reload_delta = _safe_float(
            metrics.get("save_reload_max_abs_delta"),
            "smoke save/reload prediction delta",
        )
        if not _matches_bool(metrics.get("prediction_equivalence"), True) or reload_delta > 1.0e-7:
            raise ResearchError("smoke save/reload mismatch", ExitCode.REGRESSION)
        reload_checks += 1
        if (
            manifest.get("profile") != "smoke"
            or not _matches_bool(manifest.get("publication_eligible"), False)
            or not _matches_bool(manifest.get("deterministic"), True)
            or not _matches_bool(environment.get("deterministic_enabled"), True)
            or manifest.get("feature_manifest_hash") != bundle.fold_manifest_hash
            or manifest.get("preflight_hash") != preflight["preflight_hash"]
            or manifest.get("source_manifest_hash") != preflight["source_manifest_hash"]
            or manifest.get("runtime_identity_hash") != preflight["runtime_identity_hash"]
        ):
            raise ResearchError("smoke identity mismatch", ExitCode.INCOMPATIBLE_ARTIFACT)
        cells[candidate] = {
            "config_hash": marker.config_hash,
            "done_sha256": sha256_file(run_dir / "DONE"),
            "feature_manifest_hash": bundle.fold_manifest_hash,
        }
    if aggregate.get("run_count") != 4:
        raise ResearchError("smoke aggregate is not the canonical four cells", ExitCode.REGRESSION)
    gate: dict[str, Any] = {
        "schema_version": "stage2-e065-smoke-gate-v2",
        "status": "E06_5_SMOKE_PASS",
        "experiment_id": experiment_id,
        "canonical_matrix": {
            "candidates": list(_E065_CANDIDATES),
            "folds": list(_E065_SMOKE_FOLDS),
            "seeds": list(_E065_SMOKE_SEEDS),
            "profile": "smoke",
            "deterministic": True,
            "device": "cpu",
            "max_parallel": 1,
        },
        "config_hash": config_hash(config),
        "preflight_hash": preflight["preflight_hash"],
        "source_manifest_hash": preflight["source_manifest_hash"],
        "runtime_identity_hash": preflight["runtime_identity_hash"],
        "aggregate_hash": aggregate["aggregate_hash"],
        "run_count": 4,
        "cells": cells,
        "checks": {
            "shapes": {"status": "PASS", "cells": shape_checks},
            "nan_inf": {"status": "PASS", "cells": finite_checks},
            "template_train_only": {"status": "PASS", "cells": template_checks},
            "scaler_train_only": {"status": "PASS", "cells": scaler_checks},
            "sampling_train_only": {"status": "PASS", "cells": sampling_checks},
            "method_train_only": {"status": "PASS", "cells": method_checks},
            "outer_test_without_selection": {
                "status": "PASS",
                "cells": outer_selection_checks,
            },
            "save_reload_prediction_equivalence": {
                "status": "PASS",
                "cells": reload_checks,
            },
        },
        "created_at": utc_now(),
    }
    gate["gate_hash"] = hash_canonical(
        {key: value for key, value in gate.items() if key != "created_at"}
    )
    return gate


def validate_e065_smoke_gate(
    config: ResearchConfig,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the smoke gate and every content-addressed source cell."""
    resolved_preflight = preflight or require_preflight(config)
    path = config.output_root / "manifests" / "e065_smoke_gate.json"
    if not path.is_file():
        raise ResearchError("E06_5_SMOKE_PASS is required", ExitCode.BLOCKED_PRECONDITION)
    gate = cast(dict[str, Any], load_json(path))
    expected_matrix = {
        "candidates": list(_E065_CANDIDATES),
        "folds": list(_E065_SMOKE_FOLDS),
        "seeds": list(_E065_SMOKE_SEEDS),
        "profile": "smoke",
        "deterministic": True,
        "device": "cpu",
        "max_parallel": 1,
    }
    expected_checks = {
        "shapes": {"status": "PASS", "cells": 4},
        "nan_inf": {"status": "PASS", "cells": 4},
        "template_train_only": {"status": "PASS", "cells": 4},
        "scaler_train_only": {"status": "PASS", "cells": 4},
        "sampling_train_only": {"status": "PASS", "cells": 4},
        "method_train_only": {"status": "PASS", "cells": 4},
        "outer_test_without_selection": {"status": "PASS", "cells": 4},
        "save_reload_prediction_equivalence": {"status": "PASS", "cells": 4},
    }
    expected_gate_hash = hash_canonical(
        {key: value for key, value in gate.items() if key not in {"created_at", "gate_hash"}}
    )
    if (
        gate.get("schema_version") != "stage2-e065-smoke-gate-v2"
        or gate.get("status") != "E06_5_SMOKE_PASS"
        or gate.get("canonical_matrix") != expected_matrix
        or gate.get("run_count") != 4
        or gate.get("checks") != expected_checks
        or gate.get("config_hash") != config_hash(config)
        or gate.get("preflight_hash") != resolved_preflight.get("preflight_hash")
        or gate.get("source_manifest_hash") != resolved_preflight.get("source_manifest_hash")
        or gate.get("runtime_identity_hash") != resolved_preflight.get("runtime_identity_hash")
        or gate.get("gate_hash") != expected_gate_hash
    ):
        raise ResearchError("invalid or stale E06.5 smoke gate", ExitCode.INCOMPATIBLE_ARTIFACT)
    experiment_id = str(gate["experiment_id"])
    cells = cast(dict[str, Any], gate.get("cells", {}))
    if set(cells) != set(_E065_CANDIDATES):
        raise ResearchError("smoke gate cell set mismatch", ExitCode.INCOMPATIBLE_ARTIFACT)
    for candidate in _E065_CANDIDATES:
        run_dir = stage_run_dir(
            config,
            stage="e06.5",
            experiment_id=experiment_id,
            candidate=candidate,
            fold=1,
            seed=17,
        )
        marker = validate_done_marker(run_dir)
        cell = cast(dict[str, Any], cells[candidate])
        manifest = cast(dict[str, Any], load_json(run_dir / "run_manifest.json"))
        if (
            marker is None
            or marker.config_hash != cell.get("config_hash")
            or sha256_file(run_dir / "DONE") != cell.get("done_sha256")
            or manifest.get("candidate") != candidate
            or manifest.get("fold") != 1
            or manifest.get("seed") != 17
            or manifest.get("profile") != "smoke"
            or not _matches_bool(manifest.get("deterministic"), True)
            or manifest.get("preflight_hash") != resolved_preflight.get("preflight_hash")
            or manifest.get("source_manifest_hash")
            != resolved_preflight.get("source_manifest_hash")
            or manifest.get("runtime_identity_hash")
            != resolved_preflight.get("runtime_identity_hash")
        ):
            raise ResearchError("smoke gate cell identity mismatch", ExitCode.INCOMPATIBLE_ARTIFACT)
    summary = cast(
        dict[str, Any],
        load_json(config.output_root / "E06_5" / experiment_id / "summary.json"),
    )
    summary_hash = hash_canonical(
        {
            key: value
            for key, value in summary.items()
            if key not in {"created_at", "aggregate_hash"}
        }
    )
    if summary.get("aggregate_hash") != summary_hash or gate.get("aggregate_hash") != summary_hash:
        raise ResearchError("smoke aggregate hash mismatch", ExitCode.INCOMPATIBLE_ARTIFACT)
    return gate


def _cell_status(run_dir: Path) -> str:
    if not run_dir.exists():
        return "PLANNED"
    try:
        marker = validate_done_marker(run_dir)
    except ResearchError:
        return "INCOMPATIBLE"
    return "DONE" if marker is not None else "RESUMABLE"


def build_e065_plan(
    config: ResearchConfig,
    *,
    candidates: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
    profile_name: ProfileName = "audit",
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Build and persist the deterministic E06.5 run matrix."""
    preflight = require_preflight(config)
    unknown = set(candidates) - set(config.candidates)
    if unknown:
        raise ResearchError(f"unknown candidates: {sorted(unknown)}", ExitCode.ARGUMENT_ERROR)
    if not set(folds).issubset(set(config.folds)) or not folds:
        raise ResearchError("invalid fold selection", ExitCode.ARGUMENT_ERROR)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ResearchError("seeds must be non-empty and unique", ExitCode.ARGUMENT_ERROR)
    resolved_experiment_id = experiment_id or config.default_experiment_ids["e065_audit"]
    cells: list[RunCell] = []
    for candidate in (name for name in ("baseline", "H6", "H11", "H12") if name in candidates):
        for fold in sorted(folds):
            for seed in seeds:
                run_dir = stage_run_dir(
                    config,
                    stage="e06.5",
                    experiment_id=resolved_experiment_id,
                    candidate=candidate,
                    fold=fold,
                    seed=seed,
                )
                status = _cell_status(run_dir)
                cells.append(
                    RunCell(
                        stage="e06.5",
                        experiment_id=resolved_experiment_id,
                        candidate=candidate,
                        fold=fold,
                        seed=seed,
                        profile=profile_name,
                        run_dir=run_dir,
                        status=cast(Any, status),
                        dependencies=("PREFLIGHT_PASS",),
                    )
                )
    counts = {
        status: sum(cell.status == status for cell in cells)
        for status in ("PLANNED", "RESUMABLE", "DONE", "INCOMPATIBLE")
    }
    semantic = {
        "schema_version": "stage2-plan-v1",
        "stage": "e06.5",
        "experiment_id": resolved_experiment_id,
        "profile": profile_name,
        "candidates": list(candidates),
        "folds": sorted(folds),
        "seeds": list(seeds),
        "preflight_hash": preflight["preflight_hash"],
        "run_count": len(cells),
        "counts": counts,
        "cells": [cell.model_dump(mode="json") for cell in cells],
    }
    semantic["plan_hash"] = hash_canonical(semantic)
    destination = config.output_root / "manifests" / f"plan_e065_{resolved_experiment_id}.json"
    atomic_write_json(destination, semantic)
    matrix_csv = io.StringIO()
    writer = csv.DictWriter(
        matrix_csv,
        fieldnames=["order", "candidate", "fold", "seed", "status", "run_dir"],
    )
    writer.writeheader()
    for order, cell in enumerate(cells, start=1):
        writer.writerow(
            {
                "order": order,
                "candidate": cell.candidate,
                "fold": cell.fold,
                "seed": cell.seed,
                "status": cell.status,
                "run_dir": str(cell.run_dir),
            }
        )
    atomic_write_text(
        config.output_root / "reports" / f"run_matrix_e065_{resolved_experiment_id}.csv",
        matrix_csv.getvalue(),
    )
    return semantic


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p10": 0.0,
            "p90": 0.0,
        }
    return {
        "mean": _safe_float(np.mean(array), "distribution mean"),
        "std": _safe_float(np.std(array), "distribution std"),
        "median": _safe_float(np.median(array), "distribution median"),
        "min": _safe_float(np.min(array), "distribution min"),
        "max": _safe_float(np.max(array), "distribution max"),
        "p10": _safe_float(np.percentile(array, 10), "distribution p10"),
        "p90": _safe_float(np.percentile(array, 90), "distribution p90"),
    }


def aggregate_experiment(
    config: ResearchConfig,
    *,
    stage: str,
    experiment_id: str,
    names: Sequence[str],
) -> dict[str, Any]:
    """Aggregate every completed fold/seed cell without dropping bad runs."""
    rows: list[dict[str, Any]] = []
    stage_dir = {"e06.5": "E06_5", "e07": "E07", "e08": "E08"}[stage]
    root = config.output_root / stage_dir / experiment_id
    for name in names:
        candidate_dir = root / name
        if not candidate_dir.exists():
            continue
        for done_path in sorted(candidate_dir.glob("fold_*/seed_*/DONE")):
            run_dir = done_path.parent
            validate_done_marker(run_dir)
            metrics = cast(dict[str, Any], load_json(run_dir / "metrics.json"))
            rows.append(metrics)
    if not rows:
        return {"stage": stage, "experiment_id": experiment_id, "candidates": {}}
    frame = pd.DataFrame(rows)
    summaries: dict[str, Any] = {}
    for name, subset in frame.groupby("candidate", sort=False):
        f_values = subset["F1_F"].astype(float).tolist()
        macro_values = subset["macro_F1"].astype(float).tolist()
        outside_values = [
            _safe_float(item["outside_208_213"]["F1_F"], "outside F1") for item in subset["scopes"]
        ]
        zero_folds = sorted(
            set(subset.loc[subset["F1_F"].astype(float) == 0.0, "fold"].astype(int).tolist())
        )
        summaries[str(name)] = {
            "run_count": len(subset),
            "F1_F": _distribution(f_values),
            "macro_F1": _distribution(macro_values),
            "outside_208_213_F1_F": _distribution(outside_values),
            "zero_F1_run_count": sum(value == 0.0 for value in f_values),
            "zero_F1_fold_count": len(zero_folds),
            "zero_F1_folds": zero_folds,
            "precision_F": _distribution(subset["precision_F"].astype(float).tolist()),
            "recall_F": _distribution(subset["recall_F"].astype(float).tolist()),
            "AP_F": _distribution(subset["AP_F"].astype(float).tolist()),
        }
    paired: dict[str, Any] = {}
    if "baseline" in set(frame["candidate"]):
        baseline = frame.loc[:, ["fold", "seed", "candidate", "F1_F"]]
        baseline = baseline.loc[baseline["candidate"] == "baseline"].rename(
            columns={"F1_F": "baseline_F1_F"}
        )
        for name in names:
            if name == "baseline":
                continue
            candidate_rows = frame.loc[frame["candidate"] == name, ["fold", "seed", "F1_F"]]
            merged = candidate_rows.merge(baseline, on=["fold", "seed"], how="inner")
            deltas = (merged["F1_F"] - merged["baseline_F1_F"]).astype(float).tolist()
            paired[name] = {
                "paired_count": len(deltas),
                "F1_F_gain": _distribution(deltas),
            }
    result = {
        "schema_version": "stage2-aggregate-v1",
        "stage": stage,
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "run_count": len(frame),
        "candidates": summaries,
        "paired_vs_baseline": paired,
        "publication_target_F1_F": config.gates.publication_f1_f,
    }
    result["aggregate_hash"] = hash_canonical(
        {key: value for key, value in result.items() if key != "created_at"}
    )
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "summary.json", result)
    return result


def run_e065(
    config: ResearchConfig,
    *,
    candidates: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
    profile_name: ProfileName,
    experiment_id: str | None,
    deterministic: bool,
    device: str,
    resume: bool,
    force: bool,
    dry_run: bool,
    max_parallel: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Execute E06.5 cells in candidate/fold/seed order."""
    preflight = require_preflight(config)
    _require_e065_execution_contract(
        config,
        candidates=candidates,
        folds=folds,
        seeds=seeds,
        profile_name=profile_name,
        deterministic=deterministic,
        device=device,
        max_parallel=max_parallel,
    )
    if profile_name == "audit" and not dry_run:
        validate_e065_smoke_gate(config, preflight)
    resolved_id = (
        experiment_id
        or config.default_experiment_ids["e065_smoke" if profile_name == "smoke" else "e065_audit"]
    )
    plan = build_e065_plan(
        config,
        candidates=candidates,
        folds=folds,
        seeds=seeds,
        profile_name=profile_name,
        experiment_id=resolved_id,
    )
    if dry_run:
        return plan, {
            "planned": plan["run_count"],
            "executed": 0,
            "resumed": 0,
            "skipped": 0,
            "failed": 0,
        }
    prepared = prepare_research(config)
    counts = {"planned": plan["run_count"], "executed": 0, "resumed": 0, "skipped": 0, "failed": 0}
    ordered_candidates = [name for name in ("baseline", "H6", "H11", "H12") if name in candidates]
    for candidate in ordered_candidates:
        for fold in sorted(folds):
            bundle = build_feature_bundle(
                config,
                prepared.dataset,
                prepared.full,
                prepared.outer_splits,
                prepared.inner_splits,
                candidate_name=candidate,
                fold=fold,
            )
            for seed in seeds:
                try:
                    result = train_e06_cell(
                        config,
                        prepared.dataset,
                        prepared.outer_splits,
                        prepared.inner_splits,
                        bundle,
                        candidate=candidate,
                        fold=fold,
                        seed=seed,
                        profile_name=profile_name,
                        experiment_id=resolved_id,
                        deterministic=deterministic,
                        device=device,
                        preflight_hash=str(preflight["preflight_hash"]),
                        source_manifest_hash=str(preflight["source_manifest_hash"]),
                        runtime_identity_hash=str(preflight["runtime_identity_hash"]),
                        resume=resume,
                        force=force,
                    )
                except ResearchError:
                    counts["failed"] += 1
                    raise
                if result.status == "SKIPPED_DONE":
                    counts["skipped"] += 1
                else:
                    counts["executed"] += 1
    aggregate = aggregate_experiment(
        config,
        stage="e06.5",
        experiment_id=resolved_id,
        names=ordered_candidates,
    )
    if profile_name == "smoke":
        if counts["executed"] + counts["skipped"] != 4:
            raise ResearchError("E06.5 smoke matrix incomplete", ExitCode.REGRESSION)
        gate = _build_e065_smoke_gate(
            config,
            preflight,
            prepared,
            experiment_id=resolved_id,
            aggregate=aggregate,
        )
        atomic_write_json(config.output_root / "manifests" / "e065_smoke_gate.json", gate)
        atomic_write_text(
            config.output_root / "E06_5" / resolved_id / "checkpoint.md",
            "# E06.5 smoke checkpoint\n\n**E06_5_SMOKE_PASS**\n",
        )
    return aggregate, counts


def status_report(config: ResearchConfig) -> dict[str, Any]:
    """Summarize planned/running/passed/failed/blocked cells and selections."""
    rows: list[dict[str, Any]] = []
    stage_roots = {
        "E06.5": config.output_root / "E06_5",
        "FoldAudit": config.output_root / "fold_audits",
        "E07": config.output_root / "E07",
        "E08": config.output_root / "E08",
    }
    for stage, root in stage_roots.items():
        done = len(list(root.glob("**/DONE"))) if root.exists() else 0
        running = len(list(root.glob("**/.RUNNING.lock"))) if root.exists() else 0
        failed = 0
        incomplete = 0
        if root.exists():
            for manifest_path in root.glob("**/run_manifest.json"):
                run_dir = manifest_path.parent
                if (run_dir / "DONE").exists():
                    continue
                incomplete += 1
                try:
                    status = cast(dict[str, Any], load_json(manifest_path)).get("status")
                except ResearchError:
                    status = "FAILED"
                if status == "FAILED":
                    failed += 1
        planned = 100 if stage == "E06.5" else done + incomplete
        rows.append(
            {
                "stage": stage,
                "planned": planned,
                "running": running,
                "passed": done,
                "failed": failed,
                "blocked": max(0, incomplete - running - failed),
            }
        )
    selection_dir = config.output_root / "selections"
    selected: dict[str, Any] = {}
    for filename, key in (
        ("representation_selection.json", "representation"),
        ("e07_selection.json", "sampler"),
        ("e08_selection.json", "method"),
    ):
        path = selection_dir / filename
        if path.exists():
            selected[key] = cast(dict[str, Any], load_json(path)).get("selected_name")
    return {
        "stages": rows,
        "selected": selected,
        "publication_target_F1_F": config.gates.publication_f1_f,
        "next_action": (
            "preflight"
            if not (config.output_root / "manifests" / "preflight.json").exists()
            else "plan"
        ),
    }
