"""Patient-disjoint E06.5-PD and E07-PD workflows for E07R."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.stage2_research.config import load_research_config
from src.stage2_research.contracts import (
    DatasetConfig,
    HashedPath,
    InnerSplitManifest,
    ProfileConfig,
    ProfileName,
    ResearchConfig,
    SamplerName,
    SplitManifest,
)
from src.stage2_research.data import (
    FullTemplateDataset,
    Stage2Dataset,
    load_full_template_dataset,
    load_stage2_dataset,
)
from src.stage2_research.e07r_contracts import (
    E065PDSelectionV4,
    E07PDResultV4,
    E07RPDProtocolManifestV4,
    Stage2PatientMappingV4,
)
from src.stage2_research.e07r_integrity import (
    E07RIntegrityError,
    E07RIntegrityPaths,
    assert_authorized_split_path,
    guard_e07r_write,
    require_e07r_preflight,
)
from src.stage2_research.features import build_feature_bundle
from src.stage2_research.integrity import (
    collect_environment_without_reseeding,
    hash_canonical,
    load_json,
    runtime_identity_hash,
    validate_done_marker,
)
from src.stage2_research.training import train_e06_cell
from src.training_integrity.integrity import (
    sha256_file,
    write_bytes_exclusive,
    write_json_exclusive,
)

R5_NPZ_SHA256 = "74f421e7b60c8befab1ce240c892d9db4eaadf2e0af1ee8d0fe0b2f8c5ef2658"
R5_PARQUET_SHA256 = "cc76acf5f33d0a9d38d4f94269a8ef3faead7009c1f45ca0e4f77a1fc0d7a56d"
R4_FULL_NPZ_SHA256 = "d8ce5061634a22aafc01cc7489552b2b4b1112338bba3c870e5ce22486168f57"
R4_FULL_PARQUET_SHA256 = "92e0018a59bf9bad945ac833e038377d256414b2ea63486ce0efc614386b22e3"
E065_PD_EXPERIMENT_ID = "e06-5-pd-v4-0"
E07_PD_EXPERIMENT_ID = "e07-pd-v4-0"
PD_CANDIDATES = ("baseline", "H6", "H11", "H12")
PD_SAMPLERS: tuple[SamplerName, ...] = (
    "pd_s0_natural",
    "pd_s1_f_target",
    "pd_s2_patient_uniform_capped",
    "pd_s3_patient_sqrt_capped",
    "pd_s4_focal_gentle",
    "pd_s5_smote_feature",
)
F1_SCORE_UNTYPED: Any = f1_score
PD_SOURCE_FILES = (
    "src/models/e06_protocol.py",
    "src/stage2_research/config.py",
    "src/stage2_research/contracts.py",
    "src/stage2_research/data.py",
    "src/stage2_research/e07r_contracts.py",
    "src/stage2_research/e07r_integrity.py",
    "src/stage2_research/features.py",
    "src/stage2_research/integrity.py",
    "src/stage2_research/pd_workflows.py",
    "src/stage2_research/splits.py",
    "src/stage2_research/training.py",
    "scripts/run_stage2_e07r_pd.py",
    "config/stage2_research.yaml",
    "uv.lock",
)


def _safe_int(value: Any, *, context: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise E07RIntegrityError(f"{context} is not an integer") from error
    if integer != value:
        raise E07RIntegrityError(f"{context} is not integral")
    return integer


def _safe_float(value: Any, *, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise E07RIntegrityError(f"{context} is not numeric") from error
    if not np.isfinite(number):
        raise E07RIntegrityError(f"{context} is not finite")
    return number


@dataclass(frozen=True)
class PreparedPDResearch:
    """In-memory authenticated dataset and frozen patient-disjoint partitions."""

    config: ResearchConfig
    dataset: Stage2Dataset
    full: FullTemplateDataset
    outer_splits: SplitManifest
    inner_splits: InnerSplitManifest
    mapping: Stage2PatientMappingV4


def _base_config_path(project_root: Path) -> Path:
    return project_root / "config/stage2_research.yaml"


def _pd_config(project_root: Path) -> ResearchConfig:
    config = load_research_config(_base_config_path(project_root))
    datasets = DatasetConfig(
        stage2_npz=HashedPath(
            path=project_root / "data/features/v3.1.0-r5-stage2-pd/stage2_multiclass.npz",
            sha256=R5_NPZ_SHA256,
        ),
        stage2_parquet=HashedPath(
            path=project_root / "data/features/v3.1.0-r5-stage2-pd/stage2_multiclass.parquet",
            sha256=R5_PARQUET_SHA256,
        ),
        full_npz=HashedPath(
            path=project_root / "data/features/v3.1.0-r4/finetuning_mitbih_family.npz",
            sha256=R4_FULL_NPZ_SHA256,
        ),
        full_parquet=HashedPath(
            path=project_root / "data/features/v3.1.0-r4/finetuning_mitbih_family.parquet",
            sha256=R4_FULL_PARQUET_SHA256,
        ),
    )
    audit = config.profiles["audit"].model_copy(update={"publication_eligible": False})
    profiles: dict[ProfileName, ProfileConfig] = dict(config.profiles)
    profiles["audit"] = audit
    return config.model_copy(update={"datasets": datasets, "profiles": profiles})


def build_pd_protocol_manifest(project_root: Path) -> E07RPDProtocolManifestV4:
    """Build the immutable source/hyperparameter contract used by both PD matrices."""
    root = project_root.resolve()
    paths = E07RIntegrityPaths(root)
    mapping = Stage2PatientMappingV4.model_validate_json(paths.mapping.read_text(encoding="utf-8"))
    split = load_json(paths.split_dir / "split_manifest.json")
    source_hashes = {relative: sha256_file(root / relative) for relative in PD_SOURCE_FILES}
    payload = {
        "schema_version": "e07r-pd-protocol-v4.0",
        "status": "FROZEN",
        "date": "2026-07-26",
        "stage2_npz_sha256": R5_NPZ_SHA256,
        "stage2_parquet_sha256": R5_PARQUET_SHA256,
        "full_npz_sha256": R4_FULL_NPZ_SHA256,
        "full_parquet_sha256": R4_FULL_PARQUET_SHA256,
        "patient_mapping_hash": mapping.mapping_hash,
        "split_manifest_hash": str(split["manifest_hash"]),
        "candidates": list(PD_CANDIDATES),
        "samplers": list(PD_SAMPLERS),
        "folds": [1, 2, 3, 4, 5],
        "seeds": [17, 29, 43, 71, 101],
        "profile": "audit",
        "deterministic": True,
        "device": "cpu",
        "f_target_fraction": 0.125,
        "patient_cap_multiplier": 2.0,
        "f1_f_gate": 0.15,
        "primary_target": 0.50,
        "bootstrap_repetitions": 10_000,
        "bootstrap_seed": 42,
        "source_file_sha256": source_hashes,
        "source_manifest_hash": hash_canonical(source_hashes),
        "base_config_sha256": sha256_file(_base_config_path(root)),
    }
    return E07RPDProtocolManifestV4.model_validate(
        {**payload, "manifest_hash": hash_canonical(payload)}
    )


def _patient_mapping_by_record(mapping: Stage2PatientMappingV4) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in mapping.records:
        if record.patient_id is None:
            raise E07RIntegrityError("PD mapping contains unverified biological identity")
        if record.record_id in result:
            raise E07RIntegrityError("PD record IDs are not globally unique")
        result[record.record_id] = record.patient_id
    return result


def prepare_pd_research(project_root: Path) -> PreparedPDResearch:
    """Load one stable r5 snapshot and replace record groups with authenticated patients."""
    root = project_root.resolve()
    paths = E07RIntegrityPaths(root)
    assert_authorized_split_path(
        root,
        paths.split_dir,
        workflow="E06_5_PD",
        run_id="prepare",
    )
    before_hashes = {
        "npz": sha256_file(paths.r5_dir / "stage2_multiclass.npz"),
        "parquet": sha256_file(paths.r5_dir / "stage2_multiclass.parquet"),
    }
    config = _pd_config(root)
    dataset = load_stage2_dataset(config)
    full = load_full_template_dataset(config)
    mapping = Stage2PatientMappingV4.model_validate_json(paths.mapping.read_text(encoding="utf-8"))
    patient_by_record = _patient_mapping_by_record(mapping)
    frame = dataset.frame.copy()
    record_ids = np.asarray(frame.loc[:, "record_id"].astype(str), dtype=str)
    patient_values = [patient_by_record.get(record) for record in record_ids]
    if any(patient is None for patient in patient_values):
        raise E07RIntegrityError("Stage 2 row lacks authenticated patient identity")
    patient_ids = np.asarray(patient_values, dtype=str)
    frame["patient_id"] = patient_ids
    manifest = dict(dataset.manifest)
    manifest.pop("manifest_hash", None)
    manifest.update(
        {
            "group_key": "patient_id",
            "n_groups": _safe_int(
                frame["patient_id"].nunique(),
                context="authenticated patient count",
            ),
            "patient_mapping_hash": mapping.mapping_hash,
            "generation_namespace": "E07R_PD",
        }
    )
    dataset_manifest_hash = hash_canonical(manifest)
    manifest["manifest_hash"] = dataset_manifest_hash
    pd_dataset = replace(
        dataset,
        frame=frame,
        groups=patient_ids,
        manifest=manifest,
        manifest_hash=dataset_manifest_hash,
    )

    full_frame = full.frame.copy()
    full_records = full_frame["record_id"].astype(str)
    full_datasets = full_frame["dataset"].astype(str)
    full_patient_ids = np.asarray(
        [
            patient_by_record.get(record, f"EXCLUDED::{dataset_name}::{record}")
            for record, dataset_name in zip(full_records, full_datasets, strict=True)
        ],
        dtype=str,
    )
    full_frame["patient_id"] = full_patient_ids
    full_manifest = dict(full.manifest)
    full_manifest.update(
        {
            "group_key": "patient_id_or_explicit_excluded_barrier",
            "patient_mapping_hash": mapping.mapping_hash,
            "generation_namespace": "E07R_PD",
        }
    )
    pd_full = replace(
        full,
        frame=full_frame,
        groups=full_patient_ids,
        manifest=full_manifest,
    )
    outer = SplitManifest.model_validate_json(
        (paths.split_dir / "outer_splits_stage2.json").read_text(encoding="utf-8")
    )
    inner = InnerSplitManifest.model_validate_json(
        (paths.split_dir / "inner_splits_stage2.json").read_text(encoding="utf-8")
    )
    after_hashes = {
        "npz": sha256_file(paths.r5_dir / "stage2_multiclass.npz"),
        "parquet": sha256_file(paths.r5_dir / "stage2_multiclass.parquet"),
    }
    if before_hashes != after_hashes:
        raise E07RIntegrityError("r5 dataset changed while loading PD snapshot")
    return PreparedPDResearch(config, pd_dataset, pd_full, outer, inner, mapping)


def _runtime_hash(config: ResearchConfig) -> str:
    identity = collect_environment_without_reseeding(
        deterministic=True,
        device="cpu",
        split_random_state=config.split_contract.random_seed,
    )
    return runtime_identity_hash(identity)


def _patient_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for patient_id, patient_frame in predictions.groupby("patient_id", sort=True):
        y_true = np.asarray(patient_frame.loc[:, "y_true"], dtype=np.int64)
        y_pred = np.asarray(patient_frame.loc[:, "y_pred"], dtype=np.int64)
        rows.append(
            {
                "patient_id": str(patient_id),
                "n_samples": len(patient_frame),
                "F_support": _safe_int(
                    np.sum(y_true == 2),
                    context="patient F support",
                ),
                "F_predicted": _safe_int(
                    np.sum(y_pred == 2),
                    context="patient F predictions",
                ),
                "F1_F": _safe_float(
                    F1_SCORE_UNTYPED(
                        y_true,
                        y_pred,
                        labels=[2],
                        average="macro",
                        zero_division=0.0,
                    ),
                    context="patient F1(F)",
                ),
                "macro_F1": _safe_float(
                    F1_SCORE_UNTYPED(
                        y_true,
                        y_pred,
                        labels=[0, 1, 2],
                        average="macro",
                        zero_division=0.0,
                    ),
                    context="patient macro F1",
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_once_patient_metrics(run_dir: Path) -> None:
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    if "patient_id" not in predictions:
        raise E07RIntegrityError("PD predictions omit patient_id")
    content = _patient_metrics(predictions).to_csv(index=False, lineterminator="\n").encode()
    path = run_dir / "patient_metrics.csv"
    if path.exists():
        if path.read_bytes() != content:
            raise E07RIntegrityError("patient metrics drift in finalized PD cell")
        return
    write_bytes_exclusive(path, content)


def run_e065_pd(
    project_root: Path,
    *,
    run_id: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute or plan the exact 4×5×5 E06.5-PD audit matrix."""
    root = project_root.resolve()
    paths = E07RIntegrityPaths(root)
    protocol = E07RPDProtocolManifestV4.model_validate_json(
        paths.pd_protocol_manifest.read_text(encoding="utf-8")
    )
    plan = {
        "stage": "E06_5_PD",
        "experiment_id": E065_PD_EXPERIMENT_ID,
        "candidates": list(PD_CANDIDATES),
        "folds": list(protocol.folds),
        "seeds": list(protocol.seeds),
        "planned_cells": 100,
        "protocol_manifest_hash": protocol.manifest_hash,
    }
    if dry_run:
        return {**plan, "status": "PLANNED"}
    preflight = require_e07r_preflight(
        root,
        workflow="E06_5_PD",
        run_id=run_id,
    )
    guard_e07r_write(
        root,
        paths.project_root / "experiments/stage2_v2.4_research/E06_5_PD",
        workflow="E06_5_PD",
        run_id=run_id,
    )
    guard_e07r_write(
        root,
        paths.project_root / "experiments/stage2_v2.4_research/cache_pd",
        workflow="E06_5_PD",
        run_id=run_id,
    )
    prepared = prepare_pd_research(root)
    runtime_hash = _runtime_hash(prepared.config)
    counts = {"executed": 0, "resumed": 0, "failed": 0}
    for candidate in PD_CANDIDATES:
        for fold in protocol.folds:
            bundle = build_feature_bundle(
                prepared.config,
                prepared.dataset,
                prepared.full,
                prepared.outer_splits,
                prepared.inner_splits,
                candidate_name=candidate,
                fold=fold,
            )
            for seed in protocol.seeds:
                try:
                    result = train_e06_cell(
                        prepared.config,
                        prepared.dataset,
                        prepared.outer_splits,
                        prepared.inner_splits,
                        bundle,
                        candidate=candidate,
                        fold=fold,
                        seed=seed,
                        profile_name="audit",
                        experiment_id=E065_PD_EXPERIMENT_ID,
                        deterministic=True,
                        device="cpu",
                        sampler="natural",
                        method="ce_control",
                        stage="e06.5-pd",
                        preflight_hash=preflight.report_hash,
                        source_manifest_hash=protocol.source_manifest_hash,
                        runtime_identity_hash=runtime_hash,
                        resume=True,
                        force=False,
                    )
                    _write_once_patient_metrics(result.run_dir)
                except Exception:
                    counts["failed"] += 1
                    raise
                if result.resumed:
                    counts["resumed"] += 1
                else:
                    counts["executed"] += 1
    selection = derive_e065_pd_selection(root, protocol)
    return {
        **plan,
        **counts,
        "status": selection.status,
        "selection_hash": selection.selection_hash,
    }


def _average_oof_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    required = {"sample_id", "patient_id", "y_true", "p_S", "p_V", "p_F"}
    if not required.issubset(combined.columns):
        raise E07RIntegrityError("PD predictions lack OOF provenance columns")
    metadata = combined.groupby("sample_id", sort=True).agg(
        patient_count=("patient_id", "nunique"),
        label_count=("y_true", "nunique"),
        seed_count=("y_true", "size"),
    )
    if (
        (metadata["patient_count"] != 1).any()
        or (metadata["label_count"] != 1).any()
        or (metadata["seed_count"] != 5).any()
    ):
        raise E07RIntegrityError("OOF seed aggregation identity/count mismatch")
    averaged = (
        combined.groupby("sample_id", sort=True)
        .agg(
            patient_id=("patient_id", "first"),
            y_true=("y_true", "first"),
            p_S=("p_S", "mean"),
            p_V=("p_V", "mean"),
            p_F=("p_F", "mean"),
        )
        .reset_index()
    )
    averaged["y_pred"] = np.argmax(
        np.asarray(averaged.loc[:, ["p_S", "p_V", "p_F"]], dtype=np.float64),
        axis=1,
    )
    return averaged


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = np.asarray(frame.loc[:, "y_true"], dtype=np.int64)
    y_pred = np.asarray(frame.loc[:, "y_pred"], dtype=np.int64)
    return {
        "F1_F": _safe_float(
            F1_SCORE_UNTYPED(
                y_true,
                y_pred,
                labels=[2],
                average="macro",
                zero_division=0.0,
            ),
            context="aggregate F1(F)",
        ),
        "macro_F1": _safe_float(
            F1_SCORE_UNTYPED(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0.0,
            ),
            context="aggregate macro F1",
        ),
    }


def _patient_confusion(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    patients = sorted(frame["patient_id"].astype(str).unique().tolist())
    tp = np.zeros(len(patients), dtype=np.int64)
    fp = np.zeros(len(patients), dtype=np.int64)
    fn = np.zeros(len(patients), dtype=np.int64)
    for index, patient in enumerate(patients):
        rows = frame[frame["patient_id"].astype(str) == patient]
        true_f = np.asarray(rows.loc[:, "y_true"], dtype=np.int64) == 2
        pred_f = np.asarray(rows.loc[:, "y_pred"], dtype=np.int64) == 2
        tp[index] = np.sum(true_f & pred_f)
        fp[index] = np.sum(~true_f & pred_f)
        fn[index] = np.sum(true_f & ~pred_f)
    return tp, fp, fn, patients


def _f1_from_confusion(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denominator = 2 * tp + fp + fn
    return np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )


def _patient_cluster_delta(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, tuple[float, float]]:
    if not np.array_equal(
        np.asarray(baseline.loc[:, "sample_id"]),
        np.asarray(candidate.loc[:, "sample_id"]),
    ):
        raise E07RIntegrityError("baseline/H6 OOF samples are not aligned")
    b_tp, b_fp, b_fn, b_patients = _patient_confusion(baseline)
    c_tp, c_fp, c_fn, c_patients = _patient_confusion(candidate)
    if b_patients != c_patients:
        raise E07RIntegrityError("baseline/H6 OOF patient sets differ")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(b_patients), size=(repetitions, len(b_patients)))
    baseline_f1 = _f1_from_confusion(
        b_tp[draws].sum(axis=1),
        b_fp[draws].sum(axis=1),
        b_fn[draws].sum(axis=1),
    )
    candidate_f1 = _f1_from_confusion(
        c_tp[draws].sum(axis=1),
        c_fp[draws].sum(axis=1),
        c_fn[draws].sum(axis=1),
    )
    deltas = candidate_f1 - baseline_f1
    point = _metrics(candidate)["F1_F"] - _metrics(baseline)["F1_F"]
    low, high = np.quantile(deltas, [0.025, 0.975])
    return point, (
        _safe_float(low, context="bootstrap CI lower bound"),
        _safe_float(high, context="bootstrap CI upper bound"),
    )


def derive_e065_pd_selection(
    project_root: Path,
    protocol: E07RPDProtocolManifestV4,
) -> E065PDSelectionV4:
    """Derive the precommitted H6-vs-baseline decision from all 100 DONE cells."""
    root = project_root.resolve()
    output_root = root / "experiments/stage2_v2.4_research/E06_5_PD" / E065_PD_EXPERIMENT_ID
    predictions: dict[str, list[pd.DataFrame]] = {name: [] for name in PD_CANDIDATES}
    done_hashes: dict[str, str] = {}
    completed = 0
    for candidate in PD_CANDIDATES:
        for fold in protocol.folds:
            for seed in protocol.seeds:
                run_dir = output_root / candidate / f"fold_{fold}" / f"seed_{seed}"
                validate_done_marker(run_dir)
                key = f"{candidate}/fold_{fold}/seed_{seed}"
                done_hashes[key] = sha256_file(run_dir / "DONE")
                frame = pd.read_parquet(run_dir / "predictions.parquet")
                if "patient_id" not in frame:
                    raise E07RIntegrityError(f"PD cell omits patient_id: {key}")
                predictions[candidate].append(frame)
                completed += 1
    if completed != 100:
        raise E07RIntegrityError(f"E06.5-PD matrix incomplete: {completed}/100")
    averaged = {
        candidate: _average_oof_predictions(frames) for candidate, frames in predictions.items()
    }
    aggregate_metrics = {candidate: _metrics(frame) for candidate, frame in averaged.items()}
    delta, interval = _patient_cluster_delta(
        averaged["baseline"],
        averaged["H6"],
        repetitions=protocol.bootstrap_repetitions,
        seed=protocol.bootstrap_seed,
    )
    h6 = aggregate_metrics["H6"]
    reasons: list[str] = []
    if h6["F1_F"] < protocol.f1_f_gate:
        reasons.append("H6_F1_F_BELOW_QG5_PRIME")
    if h6["macro_F1"] < protocol.primary_target:
        reasons.append("H6_PRIMARY_TARGET_BELOW_0_50")
    if delta <= 0.0 or interval[0] <= 0.0:
        reasons.append("H6_GAIN_OVER_BASELINE_NOT_ROBUST")
    valid = not reasons
    model_reference = (
        {
            "representation": "H6",
            "source_experiment_id": E065_PD_EXPERIMENT_ID,
            "protocol_manifest_hash": protocol.manifest_hash,
            "promotion_status": "NOT_PROMOTED_INTERNAL_REFERENCE_ONLY",
        }
        if valid
        else None
    )
    payload = {
        "schema_version": "e06-5-pd-selection-v4.0",
        "status": "VALID_H_STAR_PD" if valid else "NO_VALID_CANDIDATE",
        "selected_candidate": "H6" if valid else None,
        "experiment_id": E065_PD_EXPERIMENT_ID,
        "completed_cells": 100,
        "protocol_manifest_hash": protocol.manifest_hash,
        "source_done_hashes": done_hashes,
        "aggregate_metrics": aggregate_metrics,
        "h6_minus_baseline_f1_f": delta,
        "h6_minus_baseline_ci95": list(interval),
        "f1_f_gate": protocol.f1_f_gate,
        "primary_target": protocol.primary_target,
        "decision_reasons": reasons,
        "model_reference": model_reference,
    }
    selection = E065PDSelectionV4.model_validate(
        {**payload, "selection_hash": hash_canonical(payload)}
    )
    path = output_root / "h_star_pd_selection.json"
    if path.exists():
        stored = E065PDSelectionV4.model_validate_json(path.read_text(encoding="utf-8"))
        if stored != selection:
            raise E07RIntegrityError("frozen H*-PD selection drift")
        return stored
    write_json_exclusive(path, selection.model_dump(mode="json"))
    return selection


def run_e07_pd(
    project_root: Path,
    *,
    run_id: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute or plan the exact 6×5×5 train-only E07-PD matrix."""
    root = project_root.resolve()
    paths = E07RIntegrityPaths(root)
    protocol = E07RPDProtocolManifestV4.model_validate_json(
        paths.pd_protocol_manifest.read_text(encoding="utf-8")
    )
    selection = load_valid_e065_pd_selection(root)
    plan = {
        "stage": "E07_PD",
        "experiment_id": E07_PD_EXPERIMENT_ID,
        "representation": "H6",
        "samplers": list(PD_SAMPLERS),
        "folds": list(protocol.folds),
        "seeds": list(protocol.seeds),
        "planned_cells": 150,
        "protocol_manifest_hash": protocol.manifest_hash,
        "h_star_selection_hash": selection.selection_hash,
    }
    if dry_run:
        return {**plan, "status": "PLANNED"}
    preflight = require_e07r_preflight(
        root,
        workflow="E07_PD",
        run_id=run_id,
    )
    guard_e07r_write(
        root,
        paths.project_root / "experiments/stage2_v2.4_research/E07_PD",
        workflow="E07_PD",
        run_id=run_id,
    )
    prepared = prepare_pd_research(root)
    runtime_hash = _runtime_hash(prepared.config)
    counts = {"executed": 0, "resumed": 0, "failed": 0}
    bundles = {
        fold: build_feature_bundle(
            prepared.config,
            prepared.dataset,
            prepared.full,
            prepared.outer_splits,
            prepared.inner_splits,
            candidate_name="H6",
            fold=fold,
        )
        for fold in protocol.folds
    }
    for sampler in PD_SAMPLERS:
        method: Literal["ce_control", "focal_legacy"] = (
            "focal_legacy" if sampler == "pd_s4_focal_gentle" else "ce_control"
        )
        for fold in protocol.folds:
            for seed in protocol.seeds:
                try:
                    result = train_e06_cell(
                        prepared.config,
                        prepared.dataset,
                        prepared.outer_splits,
                        prepared.inner_splits,
                        bundles[fold],
                        candidate="H6",
                        fold=fold,
                        seed=seed,
                        profile_name="audit",
                        experiment_id=E07_PD_EXPERIMENT_ID,
                        deterministic=True,
                        device="cpu",
                        sampler=sampler,
                        method=method,
                        stage="e07-pd",
                        run_name=sampler,
                        representation_name="H6",
                        preflight_hash=preflight.report_hash,
                        source_manifest_hash=protocol.source_manifest_hash,
                        runtime_identity_hash=runtime_hash,
                        resume=True,
                        force=False,
                    )
                    _write_once_patient_metrics(result.run_dir)
                except Exception:
                    counts["failed"] += 1
                    raise
                if result.resumed:
                    counts["resumed"] += 1
                else:
                    counts["executed"] += 1
    scientific_result = derive_e07_pd_result(root, protocol, selection)
    return {
        **plan,
        **counts,
        "status": scientific_result.status,
        "result_hash": scientific_result.result_hash,
        "selected_sampler": scientific_result.selected_sampler,
    }


def derive_e07_pd_result(
    project_root: Path,
    protocol: E07RPDProtocolManifestV4,
    selection: E065PDSelectionV4,
) -> E07PDResultV4:
    """Derive cluster-aware E07-PD comparisons from all 150 DONE cells."""
    root = project_root.resolve()
    output_root = root / "experiments/stage2_v2.4_research/E07_PD" / E07_PD_EXPERIMENT_ID
    predictions: dict[str, list[pd.DataFrame]] = {name: [] for name in PD_SAMPLERS}
    done_hashes: dict[str, str] = {}
    completed = 0
    for sampler in PD_SAMPLERS:
        for fold in protocol.folds:
            for seed in protocol.seeds:
                run_dir = output_root / sampler / f"fold_{fold}" / f"seed_{seed}"
                validate_done_marker(run_dir)
                key = f"{sampler}/fold_{fold}/seed_{seed}"
                done_hashes[key] = sha256_file(run_dir / "DONE")
                frame = pd.read_parquet(run_dir / "predictions.parquet")
                if "patient_id" not in frame:
                    raise E07RIntegrityError(f"E07-PD cell omits patient_id: {key}")
                predictions[sampler].append(frame)
                completed += 1
    if completed != 150:
        raise E07RIntegrityError(f"E07-PD matrix incomplete: {completed}/150")
    averaged = {
        sampler: _average_oof_predictions(frames) for sampler, frames in predictions.items()
    }
    aggregate_metrics = {sampler: _metrics(frame) for sampler, frame in averaged.items()}
    comparisons: dict[str, dict[str, Any]] = {}
    baseline = averaged["pd_s0_natural"]
    for sampler in PD_SAMPLERS:
        if sampler == "pd_s0_natural":
            comparisons[sampler] = {
                "delta_F1_F": 0.0,
                "ci95": [0.0, 0.0],
                "robust_gain": False,
            }
            continue
        delta, interval = _patient_cluster_delta(
            baseline,
            averaged[sampler],
            repetitions=protocol.bootstrap_repetitions,
            seed=protocol.bootstrap_seed,
        )
        comparisons[sampler] = {
            "delta_F1_F": delta,
            "ci95": list(interval),
            "robust_gain": delta > 0.0 and interval[0] > 0.0,
        }
    ranking = tuple(
        sorted(
            PD_SAMPLERS,
            key=lambda name: (
                aggregate_metrics[name]["F1_F"],
                aggregate_metrics[name]["macro_F1"],
                name,
            ),
            reverse=True,
        )
    )
    best = ranking[0]
    best_metrics = aggregate_metrics[best]
    best_is_valid = (
        best_metrics["F1_F"] >= protocol.f1_f_gate
        and best_metrics["macro_F1"] >= protocol.primary_target
        and (best == "pd_s0_natural" or bool(comparisons[best]["robust_gain"]))
    )
    payload = {
        "schema_version": "e07-pd-result-v4.0",
        "status": "COMPLETE",
        "experiment_id": E07_PD_EXPERIMENT_ID,
        "completed_cells": 150,
        "protocol_manifest_hash": protocol.manifest_hash,
        "h_star_selection_hash": selection.selection_hash,
        "source_done_hashes": done_hashes,
        "aggregate_metrics": aggregate_metrics,
        "comparisons_vs_s0": comparisons,
        "ranking": list(ranking),
        "selected_sampler": best if best_is_valid else None,
        "f1_f_gate": protocol.f1_f_gate,
        "primary_target": protocol.primary_target,
        "publication_authorized": False,
        "model_promotion_authorized": False,
    }
    result = E07PDResultV4.model_validate({**payload, "result_hash": hash_canonical(payload)})
    path = output_root / "e07_pd_result.json"
    if path.exists():
        stored = E07PDResultV4.model_validate_json(path.read_text(encoding="utf-8"))
        if stored != result:
            raise E07RIntegrityError("frozen E07-PD result drift")
        return stored
    write_json_exclusive(path, result.model_dump(mode="json"))
    return result


def load_valid_e065_pd_selection(project_root: Path) -> E065PDSelectionV4:
    """Block E07-PD unless the complete E06.5-PD decision is valid and hash-bound."""
    root = project_root.resolve()
    protocol_path = root / (
        "experiments/stage2_v2.4_research/integrity/e07r_pd_protocol_manifest.json"
    )
    try:
        protocol = E07RPDProtocolManifestV4.model_validate_json(
            protocol_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise E07RIntegrityError("E07-PD blocked: PD protocol is absent") from error
    selection_path = root / (
        "experiments/stage2_v2.4_research/E06_5_PD/"
        f"{E065_PD_EXPERIMENT_ID}/h_star_pd_selection.json"
    )
    try:
        selection = E065PDSelectionV4.model_validate_json(
            selection_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise E07RIntegrityError("E07-PD blocked: E06.5-PD selection is absent") from error
    if selection.protocol_manifest_hash != protocol.manifest_hash:
        raise E07RIntegrityError("E07-PD blocked: E06.5-PD protocol drift")
    if selection.status != "VALID_H_STAR_PD" or selection.selected_candidate != "H6":
        raise E07RIntegrityError("E07-PD blocked: no valid H*-PD candidate")
    if len(selection.source_done_hashes) != 100:
        raise E07RIntegrityError("E07-PD blocked: E06.5-PD matrix is incomplete")
    run_root = root / "experiments/stage2_v2.4_research/E06_5_PD" / E065_PD_EXPERIMENT_ID
    for relative, expected in selection.source_done_hashes.items():
        path = run_root / relative / "DONE"
        if not path.is_file() or sha256_file(path) != expected:
            raise E07RIntegrityError("E07-PD blocked: E06.5-PD DONE evidence drift")
    return selection
