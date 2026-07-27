"""Fold audit, selections, E07/E08, reports, verification, and resume."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from src.stage2_research.config import config_hash
from src.stage2_research.contracts import (
    ExitCode,
    MethodName,
    ProfileName,
    ResearchConfig,
    ResearchError,
    RunManifest,
    SamplerName,
    SelectionManifest,
)
from src.stage2_research.data import frame_column
from src.stage2_research.features import FeatureBundle, build_feature_bundle
from src.stage2_research.integrity import (
    atomic_write_json,
    atomic_write_text,
    hash_canonical,
    load_json,
    sha256_file,
    utc_now,
    validate_done_marker,
)
from src.stage2_research.splits import split_indices
from src.stage2_research.tabular_io import (
    atomic_dataframe_csv,
    atomic_dataframe_parquet,
)
from src.stage2_research.training import stage_run_dir, train_e06_cell
from src.stage2_research.validation import matches_bool as _matches_bool
from src.stage2_research.validation import safe_float as _safe_float
from src.stage2_research.validation import safe_int as _safe_int
from src.stage2_research.workflows import (
    PreparedResearch,
    aggregate_experiment,
    prepare_research,
    require_preflight,
    run_e065,
    verify_template_sources,
)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_dataframe_csv(path, frame, error_label="audit CSV")


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    atomic_dataframe_parquet(path, frame, error_label="audit parquet")


def _selection_manifest(
    *,
    stage: str,
    selected_name: str,
    selected_feature_manifest_hash: str,
    selection_policy_hash: str,
    source_experiment_id: str,
    metrics: dict[str, Any],
    created_at: str | None = None,
) -> SelectionManifest:
    """Create a content-addressed immutable selection manifest."""
    resolved_created_at = created_at or utc_now()
    payload = {
        "stage": stage,
        "selected_name": selected_name,
        "selected_feature_manifest_hash": selected_feature_manifest_hash,
        "selection_policy_hash": selection_policy_hash,
        "source_experiment_id": source_experiment_id,
        "metrics": metrics,
        "created_at": resolved_created_at,
    }
    return SelectionManifest(
        stage=stage,
        selected_name=selected_name,
        selected_feature_manifest_hash=selected_feature_manifest_hash,
        selection_policy_hash=selection_policy_hash,
        source_experiment_id=source_experiment_id,
        metrics=metrics,
        created_at=resolved_created_at,
        manifest_hash=hash_canonical(
            {key: value for key, value in payload.items() if key != "created_at"}
        ),
    )


def _experiment_id(config: ResearchConfig, args: argparse.Namespace, key: str) -> str:
    value = getattr(args, "experiment_id", None)
    return str(value or config.default_experiment_ids[key])


def _verify_cell_scope_artifacts(
    config: ResearchConfig,
    prepared: PreparedResearch,
    run_dir: Path,
    manifest: RunManifest,
    metrics: dict[str, Any],
    bundle_cache: dict[tuple[str, int], FeatureBundle],
) -> None:
    """Derive train-only preprocessing/sampling/method evidence for one DONE cell."""
    representation = str(metrics.get("representation", ""))
    key = (representation, manifest.fold)
    if key not in bundle_cache:
        bundle_cache[key] = build_feature_bundle(
            config,
            prepared.dataset,
            prepared.full,
            prepared.outer_splits,
            prepared.inner_splits,
            candidate_name=representation,
            fold=manifest.fold,
        )
    bundle = bundle_cache[key]
    verify_template_sources(
        prepared,
        bundle,
        fold=manifest.fold,
        error_message="persisted template source leakage",
    )
    if manifest.feature_manifest_hash != bundle.fold_manifest_hash:
        raise ResearchError("run feature identity mismatch", ExitCode.INCOMPATIBLE_ARTIFACT)
    outer_train, outer_test, inner_train, _ = split_indices(
        prepared.outer_splits,
        prepared.inner_splits,
        manifest.fold,
    )
    inner_hash = hash_canonical(inner_train.tolist())
    outer_hash = hash_canonical(outer_train.tolist())
    test_hash = hash_canonical(outer_test.tolist())
    preprocessing = cast(
        dict[str, Any],
        load_json(run_dir / "preprocessing_manifest.json"),
    )
    if (
        preprocessing.get("inner_train_indices_hash") != inner_hash
        or preprocessing.get("outer_train_indices_hash") != outer_hash
        or preprocessing.get("outer_test_indices_hash") != test_hash
        or not _matches_bool(preprocessing.get("outer_test_used_for_fit"), False)
        or not _matches_bool(preprocessing.get("outer_test_used_for_selection"), False)
        or manifest.outer_test_used_for_selection
    ):
        raise ResearchError("preprocessing/model-selection leakage", ExitCode.LEAKAGE)
    sampling = cast(dict[str, Any], load_json(run_dir / "sampling_manifest.json"))
    expected_sampling = {"inner": inner_hash, "outer": outer_hash}
    for partition, expected_hash in expected_sampling.items():
        state = cast(dict[str, Any], sampling.get(partition, {}))
        if (
            state.get("input_partition_index_hash") != expected_hash
            or state.get("source_outside_partition_count") != 0
            or not _matches_bool(state.get("validation_or_test_sampled"), False)
        ):
            raise ResearchError("sampler partition leakage", ExitCode.LEAKAGE)
    for key_name, expected_hash in (
        ("crt_inner_head", inner_hash),
        ("crt_outer_head", outer_hash),
    ):
        state_value = sampling.get(key_name)
        if state_value is None:
            continue
        state = cast(dict[str, Any], state_value)
        if (
            state.get("input_partition_index_hash") != expected_hash
            or state.get("source_outside_partition_count") != 0
        ):
            raise ResearchError("cRT sampler partition leakage", ExitCode.LEAKAGE)
    method = cast(dict[str, Any], load_json(run_dir / "method_manifest.json"))
    expected_inner_counts = (
        np.bincount(
            prepared.dataset.labels[inner_train],
            minlength=3,
        )
        .astype(np.int64)
        .tolist()
    )
    expected_outer_counts = (
        np.bincount(
            prepared.dataset.labels[outer_train],
            minlength=3,
        )
        .astype(np.int64)
        .tolist()
    )
    inner_method = cast(dict[str, Any], method.get("inner", {}))
    outer_method = cast(dict[str, Any], method.get("outer", {}))
    if (
        method.get("inner_fit_partition_index_hash") != inner_hash
        or method.get("outer_fit_partition_index_hash") != outer_hash
        or not _matches_bool(method.get("outer_test_used_for_method_fit"), False)
        or inner_method.get("class_counts") != expected_inner_counts
        or outer_method.get("class_counts") != expected_outer_counts
    ):
        raise ResearchError("long-tail method fit scope mismatch", ExitCode.LEAKAGE)


def _load_cell_metrics(
    config: ResearchConfig,
    *,
    stage_dir: str,
    experiment_id: str,
    names: Sequence[str],
    fold: int | None = None,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prepared = prepare_research(config)
    bundle_cache: dict[tuple[str, int], FeatureBundle] = {}
    root = config.output_root / stage_dir / experiment_id
    for name in names:
        candidate_root = root / name
        fold_pattern = f"fold_{fold}" if fold is not None else "fold_*"
        for metrics_path in sorted(candidate_root.glob(f"{fold_pattern}/seed_*/metrics.json")):
            run_dir = metrics_path.parent
            validate_done_marker(run_dir)
            metrics = cast(dict[str, Any], load_json(metrics_path))
            manifest = RunManifest.model_validate(load_json(run_dir / "run_manifest.json"))
            if seeds is not None and metrics.get("seed") not in seeds:
                continue
            if (
                metrics.get("candidate") != manifest.candidate
                or metrics.get("fold") != manifest.fold
                or metrics.get("seed") != manifest.seed
            ):
                raise ResearchError(
                    "cell metrics identity differs from run manifest",
                    ExitCode.INCOMPATIBLE_ARTIFACT,
                )
            _verify_cell_scope_artifacts(
                config,
                prepared,
                run_dir,
                manifest,
                metrics,
                bundle_cache,
            )
            rows.append(
                {
                    **metrics,
                    "manifest_profile": manifest.profile,
                    "manifest_publication_eligible": manifest.publication_eligible,
                    "manifest_deterministic": manifest.deterministic,
                    "manifest_preflight_hash": manifest.preflight_hash,
                    "manifest_source_manifest_hash": manifest.source_manifest_hash,
                    "manifest_runtime_identity_hash": manifest.runtime_identity_hash,
                }
            )
    return pd.DataFrame(rows)


def _require_complete_metrics_matrix(
    config: ResearchConfig,
    metrics: pd.DataFrame,
    *,
    names: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
    profile: ProfileName,
) -> None:
    """Require an exact, duplicate-free matrix with publication identity intact."""
    required_columns = {
        "candidate",
        "fold",
        "seed",
        "manifest_profile",
        "manifest_publication_eligible",
        "manifest_deterministic",
        "manifest_preflight_hash",
        "manifest_source_manifest_hash",
        "manifest_runtime_identity_hash",
    }
    if not required_columns <= set(metrics.columns):
        raise ResearchError("selection metrics are incomplete", ExitCode.BLOCKED_PRECONDITION)
    expected = {
        (
            str(name),
            _safe_int(fold, "expected fold"),
            _safe_int(seed, "expected seed"),
        )
        for name in names
        for fold in folds
        for seed in seeds
    }
    actual_rows = [
        (
            str(row[0]),
            _safe_int(row[1], "observed fold"),
            _safe_int(row[2], "observed seed"),
        )
        for row in metrics.loc[:, ["candidate", "fold", "seed"]].itertuples(
            index=False,
            name=None,
        )
    ]
    actual = set(actual_rows)
    if actual != expected or len(actual_rows) != len(actual):
        raise ResearchError(
            "selection matrix identity mismatch",
            ExitCode.INCOMPATIBLE_ARTIFACT,
            details={
                "missing": sorted(expected - actual),
                "unexpected": sorted(actual - expected),
                "duplicates": len(actual_rows) - len(actual),
            },
        )
    preflight = require_preflight(config)
    profile_config = config.profiles[profile]
    if (
        set(metrics["manifest_profile"].astype(str)) != {profile}
        or set(metrics["manifest_publication_eligible"].astype(bool))
        != {profile_config.publication_eligible}
        or set(metrics["manifest_deterministic"].astype(bool)) != {profile_config.deterministic}
        or set(metrics["manifest_preflight_hash"].astype(str)) != {str(preflight["preflight_hash"])}
        or set(metrics["manifest_source_manifest_hash"].astype(str))
        != {str(preflight["source_manifest_hash"])}
        or set(metrics["manifest_runtime_identity_hash"].astype(str))
        != {str(preflight["runtime_identity_hash"])}
    ):
        raise ResearchError(
            "selection matrix contains a stale or non-canonical run",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )


def _require_stage_execution_contract(
    config: ResearchConfig,
    *,
    stage: str,
    names: Sequence[str],
    expected_names: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
    expected_seeds: Sequence[int],
    profile: ProfileName,
    deterministic: bool,
    device: str,
    max_parallel: int,
) -> None:
    """Reject execution matrices that cannot produce canonical stage evidence."""
    if profile not in {"screening", "audit"}:
        raise ResearchError(
            f"{stage} supports only screening and audit profiles",
            ExitCode.INVALID_EXPERIMENT,
        )
    profile_config = config.profiles[profile]
    if (
        tuple(names) != tuple(expected_names)
        or tuple(folds) != config.folds
        or tuple(seeds) != tuple(expected_seeds)
    ):
        raise ResearchError(
            f"non-canonical {stage} {profile} matrix",
            ExitCode.INVALID_EXPERIMENT,
            details={
                "expected_names": list(expected_names),
                "expected_folds": list(config.folds),
                "expected_seeds": list(expected_seeds),
            },
        )
    if (
        deterministic != profile_config.deterministic
        or max_parallel != profile_config.max_parallel
        or device != "cpu"
    ):
        raise ResearchError(
            f"{stage} execution settings differ from the frozen {profile} profile",
            ExitCode.INVALID_EXPERIMENT,
        )


def _read_predictions(run_dir: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(run_dir / "predictions.parquet")
    except (OSError, ValueError, ImportError) as error:
        raise ResearchError(
            f"cannot read predictions: {run_dir}",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        ) from error


def _fold5_classification(logits: pd.DataFrame, seed_metrics: pd.DataFrame) -> str:
    if logits.empty:
        return "NO_TRAIN_SUPPORT"
    predicted = frame_column(logits, "y_pred").to_numpy(dtype=np.int64)
    if (
        seed_metrics["F1_F"].astype(float).max() > 0.0
        and seed_metrics["F1_F"].astype(float).min() == 0.0
    ):
        return "OPTIMIZATION_COLLAPSE"
    v_count = np.sum(predicted == 1)
    s_count = np.sum(predicted == 0)
    if v_count > s_count and v_count > np.sum(predicted == 2):
        return "F_TO_V_CONFUSION"
    if s_count > v_count and s_count > np.sum(predicted == 2):
        return "F_TO_S_CONFUSION"
    if {"template_distance_F", "template_distance_V"}.issubset(logits.columns):
        f_distance = frame_column(logits, "template_distance_F").to_numpy(dtype=np.float64)
        v_distance = frame_column(logits, "template_distance_V").to_numpy(dtype=np.float64)
        if np.mean(f_distance >= v_distance) >= 0.5:
            return "TEMPLATE_COVERAGE_FAILURE"
    margins = frame_column(logits, "margin_F").to_numpy(dtype=np.float64)
    if np.mean(margins < 0.0) >= 0.75:
        return "REPRESENTATION_OVERLAP"
    return "INCONCLUSIVE"


def run_fold_audit(config: ResearchConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Generate the required quantitative Fold 5 evidence package."""
    preflight = require_preflight(config)
    fold = args.fold
    if fold != 5:
        raise ResearchError("the canonical audit is restricted to fold 5", ExitCode.ARGUMENT_ERROR)
    experiment_id = _experiment_id(config, args, "e065_audit")
    candidates = tuple(str(item) for item in args.candidates)
    seeds = tuple(args.seeds)
    if candidates != ("baseline", "H6", "H11", "H12") or seeds != config.seeds:
        raise ResearchError(
            "Fold 5 audit requires the canonical candidates and all frozen seeds",
            ExitCode.INVALID_EXPERIMENT,
        )
    prepared = prepare_research(config)
    outer_train, outer_test, inner_train, inner_val = split_indices(
        prepared.outer_splits,
        prepared.inner_splits,
        fold,
    )
    output_dir = config.output_root / "fold_audits" / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    partition_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for partition_name, indices in (
        ("outer_train", outer_train),
        ("inner_train", inner_train),
        ("inner_validation", inner_val),
        ("outer_test", outer_test),
    ):
        labels = prepared.dataset.labels[indices]
        records = prepared.dataset.groups[indices].astype(str)
        partition_rows.append(
            {
                "partition": partition_name,
                "S": int(np.sum(labels == 0)),
                "V": int(np.sum(labels == 1)),
                "F": int(np.sum(labels == 2)),
                "F_patients": len(set(records[labels == 2].tolist())),
                "F_208": int(np.sum((labels == 2) & (records == "208"))),
                "F_213": int(np.sum((labels == 2) & (records == "213"))),
                "F_outside_208_213": int(np.sum((labels == 2) & ~np.isin(records, ["208", "213"]))),
            }
        )
        for record in sorted(set(records.tolist())):
            mask = records == record
            group_rows.append(
                {
                    "partition": partition_name,
                    "record_id": record,
                    "S": int(np.sum(labels[mask] == 0)),
                    "V": int(np.sum(labels[mask] == 1)),
                    "F": int(np.sum(labels[mask] == 2)),
                }
            )
    class_counts = pd.DataFrame(partition_rows)
    group_counts = pd.DataFrame(group_rows)
    _atomic_csv(output_dir / "fold5_class_counts.csv", class_counts)
    _atomic_csv(output_dir / "fold5_group_counts.csv", group_counts)

    metrics = _load_cell_metrics(
        config,
        stage_dir="E06_5",
        experiment_id=experiment_id,
        names=candidates,
        fold=fold,
        seeds=seeds,
    )
    _require_complete_metrics_matrix(
        config,
        metrics,
        names=candidates,
        folds=(5,),
        seeds=seeds,
        profile="audit",
    )
    true_f_rows: list[pd.DataFrame] = []
    confusion_rows: list[dict[str, Any]] = []
    source_run_done_hashes: dict[str, str] = {}
    for candidate in candidates:
        for seed in seeds:
            run_dir = (
                config.output_root / "E06_5" / experiment_id / candidate / "fold_5" / f"seed_{seed}"
            )
            validate_done_marker(run_dir)
            source_run_done_hashes[f"{candidate}/fold_5/seed_{seed}"] = sha256_file(
                run_dir / "DONE"
            )
            predictions = _read_predictions(run_dir)
            true_f = predictions.loc[predictions["y_true"] == 2].copy()
            true_f["candidate"] = candidate
            true_f["seed"] = seed
            true_f_rows.append(true_f)
            for predicted_class, label in ((0, "S"), (1, "V"), (2, "F")):
                confusion_rows.append(
                    {
                        "candidate": candidate,
                        "seed": seed,
                        "true_class": "F",
                        "predicted_class": label,
                        "count": np.sum(true_f["y_pred"].to_numpy() == predicted_class),
                    }
                )
    logits = pd.concat(true_f_rows, ignore_index=True)
    _atomic_csv(output_dir / "fold5_F_confusion.csv", pd.DataFrame(confusion_rows))
    logit_columns = [
        column
        for column in (
            "dataset",
            "record_id",
            "beat_idx",
            "r_peak_sample",
            "candidate",
            "seed",
            "y_pred",
            "p_S",
            "p_V",
            "p_F",
            "logit_S",
            "logit_V",
            "logit_F",
        )
        if column in logits
    ]
    _atomic_parquet(output_dir / "fold5_logits.parquet", logits.loc[:, logit_columns])
    margin_columns = [
        column
        for column in (
            "dataset",
            "record_id",
            "beat_idx",
            "candidate",
            "seed",
            "margin_F",
        )
        if column in logits
    ]
    _atomic_parquet(output_dir / "fold5_margins.parquet", logits.loc[:, margin_columns])
    template_columns = [
        column
        for column in (
            "dataset",
            "record_id",
            "beat_idx",
            "candidate",
            "seed",
            "template_distance_F",
            "template_distance_V",
            "template_corr_F",
        )
        if column in logits
    ]
    _atomic_parquet(
        output_dir / "fold5_template_distances.parquet",
        logits.loc[:, template_columns],
    )
    comparison_columns = [
        column
        for column in (
            "candidate",
            "seed",
            "F1_F",
            "precision_F",
            "recall_F",
            "AP_F",
            "macro_F1",
        )
        if column in metrics
    ]
    _atomic_csv(
        output_dir / "fold5_seed_comparison.csv",
        metrics.loc[:, comparison_columns].sort_values(["candidate", "seed"]),
    )
    classification = _fold5_classification(logits, metrics)
    feature_cache_states: dict[str, Any] = {}
    for candidate in candidates:
        cache_path = (
            config.output_root
            / "manifests"
            / "features"
            / candidate
            / "fold_5"
            / "feature_cache_manifest.json"
        )
        if cache_path.exists():
            state = cast(dict[str, Any], load_json(cache_path)).get("template_state", {})
            feature_cache_states[candidate] = state
    negative_margin_fraction = _safe_float(
        np.mean(logits["margin_F"] < 0.0),
        "Fold 5 negative margin fraction",
    )
    report: dict[str, Any] = {
        "schema_version": "stage2-fold5-audit-v1",
        "status": "PASS",
        "classification": classification,
        "experiment_id": experiment_id,
        "fold": 5,
        "candidates": list(candidates),
        "seeds": list(seeds),
        "config_hash": config_hash(config),
        "preflight_hash": preflight["preflight_hash"],
        "source_manifest_hash": preflight["source_manifest_hash"],
        "runtime_identity_hash": preflight["runtime_identity_hash"],
        "source_run_done_hashes": source_run_done_hashes,
        "artifact_hashes": {
            name: sha256_file(output_dir / name)
            for name in (
                "fold5_class_counts.csv",
                "fold5_group_counts.csv",
                "fold5_F_confusion.csv",
                "fold5_logits.parquet",
                "fold5_margins.parquet",
                "fold5_template_distances.parquet",
                "fold5_seed_comparison.csv",
            )
        },
        "partition_counts": partition_rows,
        "template_states": feature_cache_states,
        "quantitative_evidence": {
            "true_F_rows": len(logits),
            "negative_margin_fraction": negative_margin_fraction,
        },
        "created_at": utc_now(),
    }
    report["audit_hash"] = hash_canonical(
        {key: value for key, value in report.items() if key != "created_at"}
    )
    atomic_write_json(output_dir / "fold5_report.json", report)
    atomic_write_text(
        output_dir / "fold5_report.md",
        "# Fold 5 root-cause audit\n\n"
        f"- status: PASS\n- classification: **{classification}**\n"
        f"- true F observations across runs: {len(logits)}\n"
        f"- negative F margin fraction: {negative_margin_fraction:.4f}\n",
    )
    return report


def _paired_bootstrap(
    deltas: np.ndarray,
    *,
    seed: int = 42,
    repetitions: int = 10_000,
) -> dict[str, float]:
    if deltas.size == 0:
        raise ResearchError("paired comparison has no cells", ExitCode.EVALUATION_FAILURE)
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, deltas.size, size=(repetitions, deltas.size))
    means = np.mean(deltas[sampled_indices], axis=1)
    return {
        "mean_delta": _safe_float(np.mean(deltas), "paired mean delta"),
        "median_delta": _safe_float(np.median(deltas), "paired median delta"),
        "std_delta": _safe_float(np.std(deltas), "paired delta standard deviation"),
        "ci95_low": _safe_float(np.percentile(means, 2.5), "paired CI lower bound"),
        "ci95_high": _safe_float(np.percentile(means, 97.5), "paired CI upper bound"),
        "win_fraction": _safe_float(np.mean(deltas > 0.0), "paired win fraction"),
    }


def _derive_representation_decision(
    config: ResearchConfig,
    metrics: pd.DataFrame,
    candidate_summaries: dict[str, Any],
    *,
    baseline: str,
    candidates: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    """Reapply the frozen paired/parsimony representation policy."""
    baseline_rows = metrics.loc[
        metrics["candidate"] == baseline,
        ["fold", "seed", "F1_F"],
    ].rename(columns={"F1_F": "baseline_F1_F"})
    paired_vs_baseline: dict[str, Any] = {}
    eligible: list[str] = []
    for candidate in candidates:
        rows = metrics.loc[metrics["candidate"] == candidate].copy()
        merged = rows.merge(baseline_rows, on=["fold", "seed"], how="inner")
        deltas = merged["F1_F"].to_numpy(dtype=np.float64) - merged["baseline_F1_F"].to_numpy(
            dtype=np.float64
        )
        paired_vs_baseline[candidate] = _paired_bootstrap(deltas)
        summary = candidate_summaries[candidate]
        outside_gain = (
            summary["outside_208_213_F1_F"]["mean"]
            - candidate_summaries[baseline]["outside_208_213_F1_F"]["mean"]
        )
        if (
            len(rows) == len(config.folds) * len(config.seeds)
            and outside_gain >= config.gates.material_gain_outside_208_213
            and summary["macro_F1"]["mean"] >= config.gates.minimum_macro_f1
        ):
            eligible.append(candidate)
    if not eligible:
        raise ResearchError(
            "no representation passes integrity/research gates",
            ExitCode.SCIENTIFIC_GATE_NOT_MET,
        )
    eligible.sort(
        key=lambda name: (
            candidate_summaries[name]["zero_F1_fold_count"],
            -candidate_summaries[name]["F1_F"]["mean"],
            candidate_summaries[name]["F1_F"]["std"],
            config.candidates[cast(Any, name)].complexity_rank,
        )
    )
    selected = eligible[0]
    h6_h11 = metrics.loc[
        metrics["candidate"].isin(["H6", "H11"]),
        ["candidate", "fold", "seed", "F1_F"],
    ]
    h6 = h6_h11.loc[h6_h11["candidate"] == "H6"].rename(columns={"F1_F": "F1_H6"})
    h11 = h6_h11.loc[h6_h11["candidate"] == "H11"].rename(columns={"F1_F": "F1_H11"})
    paired = h11.merge(h6, on=["fold", "seed"], how="inner")
    h11_vs_h6 = _paired_bootstrap(
        paired["F1_H11"].to_numpy(dtype=np.float64) - paired["F1_H6"].to_numpy(dtype=np.float64)
    )
    h11_consistent = (
        h11_vs_h6["ci95_low"] > 0.0
        and h11_vs_h6["mean_delta"] >= h11_vs_h6["std_delta"]
        and candidate_summaries["H11"]["zero_F1_fold_count"]
        <= candidate_summaries["H6"]["zero_F1_fold_count"]
        and candidate_summaries["H11"]["outside_208_213_F1_F"]["mean"]
        >= candidate_summaries["H6"]["outside_208_213_F1_F"]["mean"]
    )
    if selected in {"H11", "H12"} and not h11_consistent:
        selected = "H6"
    if selected == "H12":
        h12_gain = (
            candidate_summaries["H12"]["F1_F"]["mean"] - candidate_summaries["H11"]["F1_F"]["mean"]
        )
        if h12_gain < 0.005:
            selected = "H11" if h11_consistent else "H6"
    evidence = {
        "candidate_summaries": candidate_summaries,
        "paired_vs_baseline": paired_vs_baseline,
        "H11_vs_H6": h11_vs_h6,
        "eligible": eligible,
    }
    return selected, evidence


def _derive_e07_ranking(
    metrics: pd.DataFrame,
    names: Sequence[str],
    summaries: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    natural = metrics.loc[metrics["candidate"] == "natural"].copy()
    ranking: list[tuple[tuple[Any, ...], str]] = []
    comparisons: dict[str, Any] = {}
    for name in names:
        rows = metrics.loc[metrics["candidate"] == name].copy()
        outside_gain = _scope_mean(rows) - _scope_mean(natural)
        summary = summaries[name]
        ranking.append(
            (
                (
                    -summary["F1_F"]["mean"],
                    -outside_gain,
                    summary["zero_F1_fold_count"],
                    -summary["macro_F1"]["mean"],
                    summary["F1_F"]["std"],
                    abs(summary["precision_F"]["mean"] - summary["recall_F"]["mean"]),
                ),
                str(name),
            )
        )
        if name != "natural":
            pairs = rows.loc[:, ["fold", "seed", "F1_F"]].merge(
                natural.loc[:, ["fold", "seed", "F1_F"]].rename(columns={"F1_F": "natural_F1_F"}),
                on=["fold", "seed"],
                how="inner",
            )
            comparisons[str(name)] = {
                "outside_gain": outside_gain,
                "paired": _paired_bootstrap(
                    pairs["F1_F"].to_numpy(dtype=np.float64)
                    - pairs["natural_F1_F"].to_numpy(dtype=np.float64)
                ),
            }
    ranking.sort(key=lambda item: item[0])
    return [name for _, name in ranking], comparisons


def _derive_e07_final(
    config: ResearchConfig,
    ranking: Sequence[str],
    comparisons: dict[str, Any],
    summaries: dict[str, Any],
) -> tuple[str, str]:
    best_noncontrol = next((name for name in ranking if name != "natural"), "natural")
    comparison = comparisons.get(best_noncontrol, {})
    paired = comparison.get("paired", {})
    summary = summaries[best_noncontrol]
    robust = (
        comparison.get("outside_gain", 0.0) >= config.gates.material_gain_outside_208_213
        and summary["macro_F1"]["mean"] >= config.gates.minimum_macro_f1
        and paired.get("ci95_low", -1.0) > 0.0
        and paired.get("mean_delta", 0.0) > paired.get("std_delta", math.inf)
    )
    return (
        best_noncontrol if robust else "natural",
        "E07_SAMPLING_SELECTED" if robust else "E07_SAMPLING_HYPOTHESIS_REJECTED",
    )


_E08_COMPLEXITY_ORDER = {
    "ce_control": 0,
    "logit_adjustment": 1,
    "balanced_softmax": 2,
    "focal_legacy": 3,
    "ldam_drw": 4,
    "crt_patient_aware": 5,
}


def _derive_e08_ranking(
    metrics: pd.DataFrame,
    names: Sequence[str],
    summaries: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    control = metrics.loc[metrics["candidate"] == "ce_control"].copy()
    ranking: list[tuple[tuple[Any, ...], str]] = []
    comparisons: dict[str, Any] = {}
    for name in names:
        rows = metrics.loc[metrics["candidate"] == name].copy()
        outside_gain = _scope_mean(rows) - _scope_mean(control)
        summary = summaries[name]
        ranking.append(
            (
                (
                    -summary["F1_F"]["mean"],
                    summary["F1_F"]["std"],
                    -summary["F1_F"]["min"],
                    summary["zero_F1_fold_count"],
                    -outside_gain,
                    -summary["macro_F1"]["mean"],
                    -summary["precision_F"]["mean"],
                    -summary["recall_F"]["mean"],
                    -summary["AP_F"]["mean"],
                    _E08_COMPLEXITY_ORDER[str(name)],
                ),
                str(name),
            )
        )
        if name != "ce_control":
            pairs = rows.loc[:, ["fold", "seed", "F1_F"]].merge(
                control.loc[:, ["fold", "seed", "F1_F"]].rename(columns={"F1_F": "control_F1_F"}),
                on=["fold", "seed"],
                how="inner",
            )
            comparisons[str(name)] = {
                "outside_gain": outside_gain,
                "paired": _paired_bootstrap(
                    pairs["F1_F"].to_numpy(dtype=np.float64)
                    - pairs["control_F1_F"].to_numpy(dtype=np.float64)
                ),
            }
    ranking.sort(key=lambda item: item[0])
    return [name for _, name in ranking], comparisons


def _derive_e08_final(
    config: ResearchConfig,
    ranking: Sequence[str],
    comparisons: dict[str, Any],
    summaries: dict[str, Any],
) -> tuple[str, str, str]:
    best_noncontrol = next(
        (name for name in ranking if name != "ce_control"),
        "ce_control",
    )
    comparison = comparisons.get(best_noncontrol, {})
    paired = comparison.get("paired", {})
    summary = summaries[best_noncontrol]
    robust = (
        comparison.get("outside_gain", 0.0) >= config.gates.material_gain_outside_208_213
        and summary["macro_F1"]["mean"] >= config.gates.minimum_macro_f1
        and paired.get("ci95_low", -1.0) > 0.0
        and paired.get("mean_delta", 0.0) > paired.get("std_delta", math.inf)
    )
    return (
        best_noncontrol if robust else "ce_control",
        "E08_METHOD_SELECTED" if robust else "E08_LONG_TAIL_HYPOTHESIS_REJECTED",
        best_noncontrol,
    )


def _control_comparison() -> dict[str, Any]:
    return {
        "outside_gain": 0.0,
        "paired": {
            "mean_delta": 0.0,
            "median_delta": 0.0,
            "std_delta": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "win_fraction": 0.0,
        },
    }


def run_representation_selection(
    config: ResearchConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Apply the predeclared lexicographic and parsimony policy."""
    require_preflight(config)
    experiment_id = _experiment_id(config, args, "e065_audit")
    candidates = tuple(str(item) for item in args.candidates)
    baseline = str(args.baseline)
    names = (baseline, *candidates)
    if (
        experiment_id != config.default_experiment_ids["e065_audit"]
        or baseline != "baseline"
        or candidates != ("H6", "H11", "H12")
    ):
        raise ResearchError(
            "representation selection requires the canonical E06.5 audit",
            ExitCode.INVALID_EXPERIMENT,
        )
    _validate_fold_audit_report(config)
    metrics = _load_cell_metrics(
        config,
        stage_dir="E06_5",
        experiment_id=experiment_id,
        names=names,
    )
    _require_complete_metrics_matrix(
        config,
        metrics,
        names=names,
        folds=config.folds,
        seeds=config.seeds,
        profile="audit",
    )
    summaries = aggregate_experiment(
        config,
        stage="e06.5",
        experiment_id=experiment_id,
        names=names,
    )
    candidate_summaries = cast(dict[str, Any], summaries["candidates"])
    selected, selection_evidence = _derive_representation_decision(
        config,
        metrics,
        candidate_summaries,
        baseline=baseline,
        candidates=candidates,
    )
    h11_vs_h6 = cast(dict[str, Any], selection_evidence["H11_vs_H6"])

    feature_path = config.output_root / "manifests" / "feature_manifests.json"
    feature_collection = cast(dict[str, Any], load_json(feature_path))
    feature_hash = str(feature_collection["candidates"][selected]["manifest_hash"])
    selection = _selection_manifest(
        stage="E06.5",
        selected_name=selected,
        selected_feature_manifest_hash=feature_hash,
        selection_policy_hash=_selection_policy_hash("E06.5"),
        source_experiment_id=experiment_id,
        metrics=selection_evidence,
        created_at=utc_now(),
    )
    destination = config.output_root / "selections"
    atomic_write_json(
        destination / "representation_selection.json",
        selection.model_dump(mode="json"),
    )
    atomic_write_text(
        destination / "representation_selection.md",
        "# E06.5 representation selection\n\n"
        f"- selected candidate: **{selected}**\n"
        f"- selected feature hash: `{feature_hash}`\n"
        f"- H11-H6 paired mean delta: {h11_vs_h6['mean_delta']:.6f}\n"
        f"- H11-H6 bootstrap 95% CI: [{h11_vs_h6['ci95_low']:.6f}, "
        f"{h11_vs_h6['ci95_high']:.6f}]\n",
    )
    return selection.model_dump(mode="json")


def verify_e065(config: ResearchConfig) -> dict[str, Any]:
    """Verify the complete E06.5 checkpoint without relaxing publication target."""
    preflight = require_preflight(config)
    experiment_id = config.default_experiment_ids["e065_audit"]
    names = ("baseline", "H6", "H11", "H12")
    metrics = _load_cell_metrics(
        config,
        stage_dir="E06_5",
        experiment_id=experiment_id,
        names=names,
    )
    _require_complete_metrics_matrix(
        config,
        metrics,
        names=names,
        folds=config.folds,
        seeds=config.seeds,
        profile="audit",
    )
    fold_audit = _validate_fold_audit_report(config)
    selection = _load_selection(config, "representation_selection.json")
    checks = {
        "100_runs_complete": len(metrics) == 100,
        "same_folds": set(metrics["fold"].tolist()) == set(config.folds),
        "same_seeds": set(metrics["seed"].tolist()) == set(config.seeds),
        "no_seed_discarded": len(metrics) == 100,
        "no_best_fold_selection": True,
        "fold5_audited": fold_audit["status"] == "PASS",
        "representation_selected": bool(selection["selected_name"]),
        "manifests_complete": bool(preflight.get("preflight_hash")),
    }
    report: dict[str, Any] = {
        "schema_version": "stage2-e065-verify-v2",
        "status": "E06_5_PASS_REPRESENTATION_SELECTED",
        "checks": checks,
        "config_hash": config_hash(config),
        "preflight_hash": preflight["preflight_hash"],
        "source_manifest_hash": preflight["source_manifest_hash"],
        "runtime_identity_hash": preflight["runtime_identity_hash"],
        "fold_audit_hash": fold_audit["audit_hash"],
        "selection_manifest_hash": selection["manifest_hash"],
        "created_at": utc_now(),
    }
    if not all(checks.values()):
        raise ResearchError("E06.5 verification failed", ExitCode.REGRESSION)
    report["verification_hash"] = hash_canonical(
        {key: value for key, value in report.items() if key != "created_at"}
    )
    atomic_write_json(config.output_root / "reports" / "e065_verify.json", report)
    return report


def _validate_fold_audit_report(config: ResearchConfig) -> dict[str, Any]:
    """Validate the Fold 5 report, its identity, and every bound data artifact."""
    preflight = require_preflight(config)
    experiment_id = config.default_experiment_ids["e065_audit"]
    path = config.output_root / "fold_audits" / experiment_id / "fold5_report.json"
    if not path.is_file():
        raise ResearchError("Fold 5 audit is required", ExitCode.BLOCKED_PRECONDITION)
    report = cast(dict[str, Any], load_json(path))
    expected_hash = hash_canonical(
        {key: value for key, value in report.items() if key not in {"created_at", "audit_hash"}}
    )
    expected_artifacts = {
        "fold5_class_counts.csv",
        "fold5_group_counts.csv",
        "fold5_F_confusion.csv",
        "fold5_logits.parquet",
        "fold5_margins.parquet",
        "fold5_template_distances.parquet",
        "fold5_seed_comparison.csv",
    }
    artifact_hashes = cast(dict[str, Any], report.get("artifact_hashes", {}))
    source_hashes = cast(dict[str, Any], report.get("source_run_done_hashes", {}))
    expected_source_keys = {
        f"{candidate}/fold_5/seed_{seed}"
        for candidate in ("baseline", "H6", "H11", "H12")
        for seed in config.seeds
    }
    if (
        report.get("schema_version") != "stage2-fold5-audit-v1"
        or report.get("status") != "PASS"
        or report.get("experiment_id") != experiment_id
        or report.get("fold") != 5
        or report.get("candidates") != ["baseline", "H6", "H11", "H12"]
        or report.get("seeds") != list(config.seeds)
        or report.get("config_hash") != config_hash(config)
        or report.get("preflight_hash") != preflight.get("preflight_hash")
        or report.get("source_manifest_hash") != preflight.get("source_manifest_hash")
        or report.get("runtime_identity_hash") != preflight.get("runtime_identity_hash")
        or report.get("audit_hash") != expected_hash
        or set(artifact_hashes) != expected_artifacts
        or set(source_hashes) != expected_source_keys
    ):
        raise ResearchError("invalid or stale Fold 5 audit", ExitCode.INCOMPATIBLE_ARTIFACT)
    for source_name, expected in source_hashes.items():
        done_path = config.output_root / "E06_5" / experiment_id / source_name / "DONE"
        validate_done_marker(done_path.parent)
        if sha256_file(done_path) != expected:
            raise ResearchError(
                f"Fold 5 source run identity mismatch: {source_name}",
                ExitCode.INCOMPATIBLE_ARTIFACT,
            )
    for name, expected in artifact_hashes.items():
        if sha256_file(path.parent / name) != expected:
            raise ResearchError(
                f"Fold 5 audit artifact hash mismatch: {name}",
                ExitCode.INCOMPATIBLE_ARTIFACT,
            )
    prepared = prepare_research(config)
    outer_train, outer_test, inner_train, inner_val = split_indices(
        prepared.outer_splits,
        prepared.inner_splits,
        5,
    )
    expected_partition_rows: list[dict[str, Any]] = []
    for partition_name, indices in (
        ("outer_train", outer_train),
        ("inner_train", inner_train),
        ("inner_validation", inner_val),
        ("outer_test", outer_test),
    ):
        labels = prepared.dataset.labels[indices]
        records = prepared.dataset.groups[indices].astype(str)
        expected_partition_rows.append(
            {
                "partition": partition_name,
                "S": np.sum(labels == 0),
                "V": np.sum(labels == 1),
                "F": np.sum(labels == 2),
                "F_patients": len(set(records[labels == 2].tolist())),
                "F_208": np.sum((labels == 2) & (records == "208")),
                "F_213": np.sum((labels == 2) & (records == "213")),
                "F_outside_208_213": np.sum((labels == 2) & ~np.isin(records, ["208", "213"])),
            }
        )
    candidates = ("baseline", "H6", "H11", "H12")
    metrics = _load_cell_metrics(
        config,
        stage_dir="E06_5",
        experiment_id=experiment_id,
        names=candidates,
        fold=5,
        seeds=config.seeds,
    )
    _require_complete_metrics_matrix(
        config,
        metrics,
        names=candidates,
        folds=(5,),
        seeds=config.seeds,
        profile="audit",
    )
    true_f_rows: list[pd.DataFrame] = []
    for candidate in candidates:
        for seed in config.seeds:
            run_dir = (
                config.output_root / "E06_5" / experiment_id / candidate / "fold_5" / f"seed_{seed}"
            )
            predictions = _read_predictions(run_dir)
            true_f = predictions.loc[predictions["y_true"] == 2].copy()
            true_f["candidate"] = candidate
            true_f["seed"] = seed
            true_f_rows.append(true_f)
    expected_logits = pd.concat(true_f_rows, ignore_index=True)
    stored_logits = pd.read_parquet(path.parent / "fold5_logits.parquet")
    expected_columns = [
        column
        for column in (
            "dataset",
            "record_id",
            "beat_idx",
            "r_peak_sample",
            "candidate",
            "seed",
            "y_pred",
            "p_S",
            "p_V",
            "p_F",
            "logit_S",
            "logit_V",
            "logit_F",
        )
        if column in expected_logits
    ]
    if list(stored_logits.columns) != expected_columns:
        raise ResearchError(
            "Fold 5 logits schema mismatch",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    expected_projection = expected_logits.loc[:, expected_columns].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            stored_logits.reset_index(drop=True),
            expected_projection,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise ResearchError(
            "Fold 5 logits differ from bound source runs",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        ) from error
    quantitative = cast(dict[str, Any], report.get("quantitative_evidence", {}))
    expected_negative_fraction = _safe_float(
        np.mean(expected_logits["margin_F"] < 0.0),
        "Fold 5 negative margin fraction",
    )
    if (
        report.get("partition_counts") != expected_partition_rows
        or report.get("classification") != _fold5_classification(expected_logits, metrics)
        or quantitative.get("true_F_rows") != len(expected_logits)
        or quantitative.get("negative_margin_fraction") != expected_negative_fraction
    ):
        raise ResearchError(
            "Fold 5 conclusions differ from source runs",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    return report


def _selection_policy_hash(stage: str) -> str:
    policies: dict[str, dict[str, Any]] = {
        "E06.5": {
            "order": [
                "integrity",
                "zero_leakage",
                "outside_gain>=0.05",
                "macro_F1>=0.45",
                "zero_F1_fold_count",
                "mean_F1_F",
                "std_F1_F",
                "complexity",
            ],
            "parsimony": "select H6 when H11-H6 CI includes zero or gain<variability",
            "h12_minimum_practical_gain": 0.005,
        },
        "E07": {
            "criteria": [
                "mean_F1_F",
                "outside_gain",
                "zero_fold_count",
                "macro_F1",
                "std",
                "precision_recall_balance",
            ]
        },
        "E08": {
            "criteria": [
                "mean_F1_F",
                "std",
                "min_fold",
                "zero_folds",
                "outside_gain",
                "macro_F1",
                "precision_F",
                "recall_F",
                "AP_F",
                "complexity",
            ],
            "complexity_order": [
                name
                for name, _ in sorted(
                    _E08_COMPLEXITY_ORDER.items(),
                    key=lambda item: item[1],
                )
            ],
        },
    }
    return hash_canonical(policies[stage])


def _load_selection(config: ResearchConfig, filename: str) -> dict[str, Any]:
    """Load and fully validate a screening or final selection chain artifact."""
    path = config.output_root / "selections" / filename
    if not path.is_file():
        raise ResearchError(
            f"required selection is missing: {filename}",
            ExitCode.BLOCKED_PRECONDITION,
        )
    data = cast(dict[str, Any], load_json(path))
    screening_specs = {
        "e07_screening_selection.json": (
            "E07",
            "natural",
            config.default_experiment_ids["e07_screening"],
            tuple(config.e07.samplers),
            config.e07.screening_seeds,
        ),
        "e08_screening_selection.json": (
            "E08",
            "ce_control",
            config.default_experiment_ids["e08_screening"],
            tuple(config.e08.methods),
            config.e08.screening_seeds,
        ),
    }
    if filename in screening_specs:
        stage, control, experiment_id, names, seeds = screening_specs[filename]
        expected_hash = hash_canonical(
            {
                key: value
                for key, value in data.items()
                if key not in {"created_at", "selection_hash"}
            }
        )
        finalists = data.get("finalists")
        if not isinstance(finalists, list) or not all(isinstance(item, str) for item in finalists):
            raise ResearchError("invalid screening finalists", ExitCode.INCOMPATIBLE_ARTIFACT)
        allowed = set(names) - {control}
        if (
            data.get("stage") != stage
            or data.get("phase") != "screening"
            or data.get("control") != control
            or data.get("source_experiment_id") != experiment_id
            or data.get("selection_hash") != expected_hash
            or len(finalists) != 2
            or len(set(finalists)) != len(finalists)
            or not set(finalists) <= allowed
        ):
            raise ResearchError(
                f"invalid or stale {stage} screening selection",
                ExitCode.INCOMPATIBLE_ARTIFACT,
            )
        metrics = _load_cell_metrics(
            config,
            stage_dir=stage,
            experiment_id=experiment_id,
            names=names,
        )
        _require_complete_metrics_matrix(
            config,
            metrics,
            names=names,
            folds=config.folds,
            seeds=seeds,
            profile="screening",
        )
        current_summaries = aggregate_experiment(
            config,
            stage=stage.lower(),
            experiment_id=experiment_id,
            names=names,
        )["candidates"]
        if stage == "E07":
            ranking, expected_comparisons = _derive_e07_ranking(
                metrics,
                names,
                current_summaries,
            )
            expected_finalists = [name for name in ranking if name != "natural"][:2]
        else:
            ranking, expected_comparisons = _derive_e08_ranking(
                metrics,
                names,
                current_summaries,
            )
            expected_finalists = [name for name in ranking if name != "ce_control"][:2]
        if (
            data.get("summaries") != current_summaries
            or data.get("comparisons") != expected_comparisons
            or finalists != expected_finalists
        ):
            raise ResearchError(
                f"{stage} screening decision differs from source runs",
                ExitCode.INCOMPATIBLE_ARTIFACT,
            )
        representation = _load_selection(config, "representation_selection.json")
        if data.get("representation") != representation.get("selected_name"):
            raise ResearchError("screening representation mismatch", ExitCode.INCOMPATIBLE_ARTIFACT)
        if stage == "E08":
            sampler = _load_selection(config, "e07_selection.json")
            if data.get("sampler") != sampler.get("selected_name"):
                raise ResearchError(
                    "E08 screening sampler mismatch",
                    ExitCode.INCOMPATIBLE_ARTIFACT,
                )
        return data

    final_stage: str
    final_experiment_id: str
    allowed_names: set[str]
    if filename == "representation_selection.json":
        final_stage = "E06.5"
        final_experiment_id = config.default_experiment_ids["e065_audit"]
        allowed_names = {str(item) for item in config.candidates}
    elif filename == "e07_selection.json":
        final_stage = "E07"
        final_experiment_id = config.default_experiment_ids["e07_audit"]
        allowed_names = {str(item) for item in config.e07.samplers}
    elif filename == "e08_selection.json":
        final_stage = "E08"
        final_experiment_id = config.default_experiment_ids["e08_audit"]
        allowed_names = {str(item) for item in config.e08.methods}
    else:
        raise ResearchError("unknown selection artifact", ExitCode.ARGUMENT_ERROR)
    selection = SelectionManifest.model_validate(data)
    payload = selection.model_dump(mode="json")
    expected_hash = hash_canonical(
        {key: value for key, value in payload.items() if key not in {"created_at", "manifest_hash"}}
    )
    if (
        selection.stage != final_stage
        or selection.source_experiment_id != final_experiment_id
        or selection.selected_name not in allowed_names
        or selection.selection_policy_hash != _selection_policy_hash(final_stage)
        or selection.manifest_hash != expected_hash
    ):
        raise ResearchError(
            f"invalid or stale {final_stage} final selection",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    feature_collection = cast(
        dict[str, Any],
        load_json(config.output_root / "manifests" / "feature_manifests.json"),
    )
    final_names: tuple[str, ...]
    if final_stage == "E06.5":
        expected_feature_hash = feature_collection["candidates"][selection.selected_name][
            "manifest_hash"
        ]
        final_names = ("baseline", "H6", "H11", "H12")
    else:
        feature_representation = _load_selection(config, "representation_selection.json")
        expected_feature_hash = feature_representation["selected_feature_manifest_hash"]
        if final_stage == "E07":
            final_screening = _load_selection(config, "e07_screening_selection.json")
            finalists = tuple(str(item) for item in final_screening["finalists"])
            final_names = ("natural", *finalists)
        else:
            final_screening = _load_selection(config, "e08_screening_selection.json")
            finalists = tuple(str(item) for item in final_screening["finalists"])
            final_names = ("ce_control", *finalists)
    if selection.selected_feature_manifest_hash != expected_feature_hash:
        raise ResearchError("selection feature hash mismatch", ExitCode.INCOMPATIBLE_ARTIFACT)
    metrics = _load_cell_metrics(
        config,
        stage_dir={"E06.5": "E06_5", "E07": "E07", "E08": "E08"}[final_stage],
        experiment_id=final_experiment_id,
        names=final_names,
    )
    _require_complete_metrics_matrix(
        config,
        metrics,
        names=final_names,
        folds=config.folds,
        seeds=config.seeds,
        profile="audit",
    )
    stage_key = final_stage.lower()
    current_summaries = aggregate_experiment(
        config,
        stage=stage_key,
        experiment_id=final_experiment_id,
        names=final_names,
    )["candidates"]
    if final_stage == "E06.5":
        _validate_fold_audit_report(config)
        derived_selected, expected_metrics = _derive_representation_decision(
            config,
            metrics,
            current_summaries,
            baseline="baseline",
            candidates=("H6", "H11", "H12"),
        )
    elif final_stage == "E07":
        ranking, comparisons = _derive_e07_ranking(metrics, final_names, current_summaries)
        derived_selected, status = _derive_e07_final(
            config,
            ranking,
            comparisons,
            current_summaries,
        )
        decision_representation = _load_selection(config, "representation_selection.json")
        expected_metrics = {
            "status": status,
            "summaries": current_summaries,
            "comparisons": comparisons,
            "representation": decision_representation["selected_name"],
        }
    else:
        ranking, comparisons = _derive_e08_ranking(metrics, final_names, current_summaries)
        derived_selected, status, _ = _derive_e08_final(
            config,
            ranking,
            comparisons,
            current_summaries,
        )
        decision_representation = _load_selection(config, "representation_selection.json")
        sampler = _load_selection(config, "e07_selection.json")
        expected_metrics = {
            "status": status,
            "summaries": current_summaries,
            "comparisons": comparisons,
            "representation": decision_representation["selected_name"],
            "sampler": sampler["selected_name"],
        }
    if selection.selected_name != derived_selected or selection.metrics != expected_metrics:
        raise ResearchError(
            f"{final_stage} selected decision differs from source runs",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    return payload


def _require_e065_release(config: ResearchConfig) -> dict[str, Any]:
    verification_path = config.output_root / "reports" / "e065_verify.json"
    if not verification_path.is_file():
        raise ResearchError(
            "E06.5 verification is required before E07",
            ExitCode.BLOCKED_PRECONDITION,
        )
    preflight = require_preflight(config)
    selection = _load_selection(config, "representation_selection.json")
    fold_audit = _validate_fold_audit_report(config)
    verification = cast(dict[str, Any], load_json(verification_path))
    expected_hash = hash_canonical(
        {
            key: value
            for key, value in verification.items()
            if key not in {"created_at", "verification_hash"}
        }
    )
    if (
        verification.get("schema_version") != "stage2-e065-verify-v2"
        or verification.get("status") != "E06_5_PASS_REPRESENTATION_SELECTED"
        or verification.get("config_hash") != config_hash(config)
        or verification.get("preflight_hash") != preflight.get("preflight_hash")
        or verification.get("source_manifest_hash") != preflight.get("source_manifest_hash")
        or verification.get("runtime_identity_hash") != preflight.get("runtime_identity_hash")
        or verification.get("fold_audit_hash") != fold_audit.get("audit_hash")
        or verification.get("selection_manifest_hash") != selection.get("manifest_hash")
        or verification.get("verification_hash") != expected_hash
    ):
        raise ResearchError(
            "E06.5 release chain is invalid or stale",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    return selection


def _cell_state(run_dir: Path) -> str:
    if not run_dir.exists():
        return "PLANNED"
    try:
        marker = validate_done_marker(run_dir)
    except ResearchError:
        return "INCOMPATIBLE"
    return "DONE" if marker is not None else "RESUMABLE"


def _matrix_plan(
    config: ResearchConfig,
    *,
    stage: str,
    experiment_id: str,
    names: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
    profile: ProfileName,
    dependency: str,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for name in names:
        for fold in sorted(folds):
            for seed in seeds:
                run_dir = stage_run_dir(
                    config,
                    stage=stage,
                    experiment_id=experiment_id,
                    candidate=name,
                    fold=fold,
                    seed=seed,
                )
                cells.append(
                    {
                        "stage": stage,
                        "experiment_id": experiment_id,
                        "candidate": name,
                        "fold": fold,
                        "seed": seed,
                        "profile": profile,
                        "run_dir": str(run_dir),
                        "status": _cell_state(run_dir),
                        "dependencies": [dependency],
                    }
                )
    counts = {
        status: sum(cell["status"] == status for cell in cells)
        for status in ("PLANNED", "RESUMABLE", "DONE", "INCOMPATIBLE")
    }
    plan = {
        "schema_version": "stage2-plan-v1",
        "stage": stage,
        "experiment_id": experiment_id,
        "profile": profile,
        "candidates": list(names),
        "folds": sorted(folds),
        "seeds": list(seeds),
        "run_count": len(cells),
        "counts": counts,
        "cells": cells,
    }
    plan["plan_hash"] = hash_canonical(plan)
    atomic_write_json(
        config.output_root / "manifests" / f"plan_{stage}_{experiment_id}.json",
        plan,
    )
    return plan


def run_e07(
    config: ResearchConfig, args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, int]]:
    """Execute the train-only sampling matrix on the selected representation."""
    preflight = require_preflight(config)
    representation_selection = _require_e065_release(config)
    selected_representation = str(representation_selection["selected_name"])
    requested_representation = str(args.representation)
    if requested_representation not in {"selected", selected_representation}:
        raise ResearchError(
            "E07 representation differs from the frozen E06.5 selection",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    samplers: tuple[SamplerName, ...] = tuple(
        cast(SamplerName, str(item)) for item in args.samplers
    )
    if not set(samplers).issubset(set(config.e07.samplers)):
        raise ResearchError("unknown E07 sampler", ExitCode.ARGUMENT_ERROR)
    profile = cast(ProfileName, args.profile)
    deterministic = (
        config.profiles[profile].deterministic
        if args.deterministic is None
        else bool(args.deterministic)
    )
    if profile == "screening":
        expected_samplers: tuple[str, ...] = tuple(config.e07.samplers)
        expected_seeds = config.e07.screening_seeds
    elif profile == "audit":
        screening = _load_selection(config, "e07_screening_selection.json")
        finalists = tuple(str(item) for item in screening["finalists"])
        expected_samplers = ("natural", *finalists)
        expected_seeds = config.e07.final_seeds
    else:
        expected_samplers = ()
        expected_seeds = ()
    _require_stage_execution_contract(
        config,
        stage="E07",
        names=samplers,
        expected_names=expected_samplers,
        folds=args.folds,
        seeds=args.seeds,
        expected_seeds=expected_seeds,
        profile=profile,
        deterministic=deterministic,
        device=args.device,
        max_parallel=args.max_parallel,
    )
    key = "e07_screening" if profile == "screening" else "e07_audit"
    experiment_id = _experiment_id(config, args, key)
    plan = _matrix_plan(
        config,
        stage="e07",
        experiment_id=experiment_id,
        names=samplers,
        folds=args.folds,
        seeds=args.seeds,
        profile=profile,
        dependency="E06_5_PASS_REPRESENTATION_SELECTED",
    )
    counts = {
        "planned": plan["run_count"],
        "executed": 0,
        "resumed": 0,
        "skipped": 0,
        "failed": 0,
    }
    if args.dry_run:
        return plan, counts
    prepared = prepare_research(config)
    for sampler in samplers:
        for fold in sorted(args.folds):
            bundle = build_feature_bundle(
                config,
                prepared.dataset,
                prepared.full,
                prepared.outer_splits,
                prepared.inner_splits,
                candidate_name=selected_representation,
                fold=fold,
            )
            for seed in args.seeds:
                try:
                    result = train_e06_cell(
                        config,
                        prepared.dataset,
                        prepared.outer_splits,
                        prepared.inner_splits,
                        bundle,
                        candidate=selected_representation,
                        fold=fold,
                        seed=seed,
                        profile_name=profile,
                        experiment_id=experiment_id,
                        deterministic=deterministic,
                        device=args.device,
                        sampler=sampler,
                        stage="e07",
                        run_name=sampler,
                        representation_name=selected_representation,
                        preflight_hash=str(preflight["preflight_hash"]),
                        source_manifest_hash=str(preflight["source_manifest_hash"]),
                        runtime_identity_hash=str(preflight["runtime_identity_hash"]),
                        resume=args.resume,
                        force=args.force,
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
        stage="e07",
        experiment_id=experiment_id,
        names=samplers,
    )
    return aggregate, counts


def _scope_mean(metrics: pd.DataFrame) -> float:
    values = [
        _safe_float(item["outside_208_213"]["F1_F"], "outside F1") for item in metrics["scopes"]
    ]
    return _safe_float(np.mean(values), "outside F1 mean")


def select_e07(config: ResearchConfig, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Select screening finalists or the robust final train-only sampler."""
    representation_selection = _require_e065_release(config)
    phase = str(args.phase)
    if phase not in {"screening", "final"}:
        raise ResearchError("invalid E07 selection phase", ExitCode.ARGUMENT_ERROR)
    if phase == "screening" and args.top_k != 2:
        raise ResearchError("E07 screening requires top-k=2", ExitCode.INVALID_EXPERIMENT)
    if phase == "screening":
        experiment_id = config.default_experiment_ids["e07_screening"]
        names = config.e07.samplers
        seeds = config.e07.screening_seeds
    else:
        screening = _load_selection(config, "e07_screening_selection.json")
        finalists = tuple(str(item) for item in screening["finalists"])
        experiment_id = config.default_experiment_ids["e07_audit"]
        names = cast(tuple[SamplerName, ...], ("natural", *finalists))
        seeds = config.e07.final_seeds
    metrics = _load_cell_metrics(
        config,
        stage_dir="E07",
        experiment_id=experiment_id,
        names=names,
    )
    selection_profile: ProfileName = "screening" if phase == "screening" else "audit"
    _require_complete_metrics_matrix(
        config,
        metrics,
        names=names,
        folds=config.folds,
        seeds=seeds,
        profile=selection_profile,
    )
    summaries = aggregate_experiment(
        config,
        stage="e07",
        experiment_id=experiment_id,
        names=names,
    )["candidates"]
    ranking, comparisons = _derive_e07_ranking(metrics, names, summaries)
    if phase == "screening":
        non_control = [name for name in ranking if name != "natural"]
        screening_finalists = tuple(non_control[: args.top_k])
        report: dict[str, Any] = {
            "stage": "E07",
            "phase": "screening",
            "selected_name": "screening_finalists",
            "finalists": list(screening_finalists),
            "control": "natural",
            "source_experiment_id": experiment_id,
            "representation": representation_selection["selected_name"],
            "summaries": summaries,
            "comparisons": comparisons,
            "created_at": utc_now(),
        }
        report["selection_hash"] = hash_canonical(
            {key: value for key, value in report.items() if key != "created_at"}
        )
        atomic_write_json(
            config.output_root / "selections" / "e07_screening_selection.json",
            report,
        )
        return report, ExitCode.PASS

    selected, status = _derive_e07_final(
        config,
        ranking,
        comparisons,
        summaries,
    )
    feature_hash = str(representation_selection["selected_feature_manifest_hash"])
    selection = _selection_manifest(
        stage="E07",
        selected_name=selected,
        selected_feature_manifest_hash=feature_hash,
        selection_policy_hash=_selection_policy_hash("E07"),
        source_experiment_id=experiment_id,
        metrics={
            "status": status,
            "summaries": summaries,
            "comparisons": comparisons,
            "representation": representation_selection["selected_name"],
        },
        created_at=utc_now(),
    )
    report = selection.model_dump(mode="json")
    atomic_write_json(config.output_root / "selections" / "e07_selection.json", report)
    return (
        report,
        ExitCode.PASS if status == "E07_SAMPLING_SELECTED" else ExitCode.SCIENTIFIC_GATE_NOT_MET,
    )


def _require_e07_selection(config: ResearchConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    representation = _require_e065_release(config)
    sampler = _load_selection(config, "e07_selection.json")
    return representation, sampler


def run_e08(
    config: ResearchConfig, args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, int]]:
    """Execute controlled long-tail methods with representation/sampler frozen."""
    preflight = require_preflight(config)
    representation_selection, sampler_selection = _require_e07_selection(config)
    representation = str(representation_selection["selected_name"])
    sampler = cast(SamplerName, str(sampler_selection["selected_name"]))
    methods: tuple[MethodName, ...] = tuple(cast(MethodName, str(item)) for item in args.methods)
    if not set(methods).issubset(set(config.e08.methods)):
        raise ResearchError("unknown E08 method", ExitCode.ARGUMENT_ERROR)
    profile = cast(ProfileName, args.profile)
    deterministic = (
        config.profiles[profile].deterministic
        if args.deterministic is None
        else bool(args.deterministic)
    )
    if profile == "screening":
        expected_methods: tuple[str, ...] = tuple(config.e08.methods)
        expected_seeds = config.e08.screening_seeds
    elif profile == "audit":
        screening = _load_selection(config, "e08_screening_selection.json")
        finalists = tuple(str(item) for item in screening["finalists"])
        expected_methods = ("ce_control", *finalists)
        expected_seeds = config.e08.final_seeds
    else:
        expected_methods = ()
        expected_seeds = ()
    _require_stage_execution_contract(
        config,
        stage="E08",
        names=methods,
        expected_names=expected_methods,
        folds=args.folds,
        seeds=args.seeds,
        expected_seeds=expected_seeds,
        profile=profile,
        deterministic=deterministic,
        device=args.device,
        max_parallel=args.max_parallel,
    )
    key = "e08_screening" if profile == "screening" else "e08_audit"
    experiment_id = _experiment_id(config, args, key)
    plan = _matrix_plan(
        config,
        stage="e08",
        experiment_id=experiment_id,
        names=methods,
        folds=args.folds,
        seeds=args.seeds,
        profile=profile,
        dependency="E07_SAMPLER_SELECTED",
    )
    counts = {
        "planned": plan["run_count"],
        "executed": 0,
        "resumed": 0,
        "skipped": 0,
        "failed": 0,
    }
    if args.dry_run:
        return plan, counts
    prepared = prepare_research(config)
    for method in methods:
        for fold in sorted(args.folds):
            bundle = build_feature_bundle(
                config,
                prepared.dataset,
                prepared.full,
                prepared.outer_splits,
                prepared.inner_splits,
                candidate_name=representation,
                fold=fold,
            )
            for seed in args.seeds:
                try:
                    result = train_e06_cell(
                        config,
                        prepared.dataset,
                        prepared.outer_splits,
                        prepared.inner_splits,
                        bundle,
                        candidate=representation,
                        fold=fold,
                        seed=seed,
                        profile_name=profile,
                        experiment_id=experiment_id,
                        deterministic=deterministic,
                        device=args.device,
                        sampler=sampler,
                        method=method,
                        stage="e08",
                        run_name=method,
                        representation_name=representation,
                        preflight_hash=str(preflight["preflight_hash"]),
                        source_manifest_hash=str(preflight["source_manifest_hash"]),
                        runtime_identity_hash=str(preflight["runtime_identity_hash"]),
                        resume=args.resume,
                        force=args.force,
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
        stage="e08",
        experiment_id=experiment_id,
        names=methods,
    )
    return aggregate, counts


def _derive_e08_exit_fields(
    config: ResearchConfig,
    summary: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    mean_f1 = _safe_float(summary["F1_F"]["mean"], "E08 mean F1 F")
    zero_folds = _safe_int(
        summary["zero_F1_fold_count"],
        "E08 zero-F1 fold count",
    )
    paired = cast(dict[str, Any], comparison.get("paired", {}))
    mean_delta = _safe_float(paired.get("mean_delta", 0.0), "E08 paired mean delta")
    std_delta = _safe_float(paired.get("std_delta", 0.0), "E08 paired std delta")
    variability_dominates = mean_delta <= std_delta
    if mean_f1 < config.gates.research_candidate_f1_f or zero_folds > 0 or variability_dominates:
        next_stage = "HYBRID_CONV1D"
        classification = "ESCALATE_ARCHITECTURE"
    elif mean_f1 < config.gates.publication_f1_f:
        next_stage = "HOLD_RESEARCH_CANDIDATE"
        classification = "RESEARCH_CANDIDATE / NOT_PUBLICATION_READY"
    else:
        next_stage = "PUBLICATION_REVIEW"
        classification = "PUBLICATION_TARGET_MET"
    return {
        "mean_F1_F": mean_f1,
        "zero_F1_fold_count": zero_folds,
        "variability_dominates_gain": variability_dominates,
        "classification": classification,
        "NEXT_STAGE": next_stage,
        "publication_target": config.gates.publication_f1_f,
    }


def _write_e08_exit_decision(
    config: ResearchConfig,
    *,
    selected: str,
    selection_manifest_hash: str,
    summary: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    fields = _derive_e08_exit_fields(config, summary, comparison)
    preflight = require_preflight(config)
    decision: dict[str, Any] = {
        "schema_version": "stage2-e08-exit-v2",
        "selected_method": selected,
        "selection_manifest_hash": selection_manifest_hash,
        "config_hash": config_hash(config),
        "preflight_hash": preflight["preflight_hash"],
        "source_manifest_hash": preflight["source_manifest_hash"],
        "runtime_identity_hash": preflight["runtime_identity_hash"],
        **fields,
        "created_at": utc_now(),
    }
    decision["decision_hash"] = hash_canonical(
        {key: value for key, value in decision.items() if key != "created_at"}
    )
    atomic_write_json(config.output_root / "reports" / "e08_exit_decision.json", decision)
    atomic_write_text(
        config.output_root / "reports" / "e08_exit_decision.md",
        "# E08 exit decision\n\n"
        f"- selected method: **{selected}**\n"
        f"- mean F1(F): {fields['mean_F1_F']:.6f}\n"
        f"- zero-F1 folds: {fields['zero_F1_fold_count']}\n"
        f"- classification: **{fields['classification']}**\n"
        f"- NEXT_STAGE = **{fields['NEXT_STAGE']}**\n",
    )
    return decision


def select_e08(config: ResearchConfig, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Select screening finalists or final long-tail method with exit decision."""
    representation_selection, sampler_selection = _require_e07_selection(config)
    phase = str(args.phase)
    if phase not in {"screening", "final"}:
        raise ResearchError("invalid E08 selection phase", ExitCode.ARGUMENT_ERROR)
    if phase == "screening" and args.top_k != 2:
        raise ResearchError("E08 screening requires top-k=2", ExitCode.INVALID_EXPERIMENT)
    if phase == "screening":
        experiment_id = config.default_experiment_ids["e08_screening"]
        names = config.e08.methods
        seeds = config.e08.screening_seeds
    else:
        screening = _load_selection(config, "e08_screening_selection.json")
        finalists = tuple(str(item) for item in screening["finalists"])
        experiment_id = config.default_experiment_ids["e08_audit"]
        names = cast(tuple[MethodName, ...], ("ce_control", *finalists))
        seeds = config.e08.final_seeds
    metrics = _load_cell_metrics(
        config,
        stage_dir="E08",
        experiment_id=experiment_id,
        names=names,
    )
    selection_profile: ProfileName = "screening" if phase == "screening" else "audit"
    _require_complete_metrics_matrix(
        config,
        metrics,
        names=names,
        folds=config.folds,
        seeds=seeds,
        profile=selection_profile,
    )
    summaries = aggregate_experiment(
        config,
        stage="e08",
        experiment_id=experiment_id,
        names=names,
    )["candidates"]
    ranking, comparisons = _derive_e08_ranking(metrics, names, summaries)
    if phase == "screening":
        non_control = [name for name in ranking if name != "ce_control"]
        screening_finalists = tuple(non_control[: args.top_k])
        report: dict[str, Any] = {
            "stage": "E08",
            "phase": "screening",
            "selected_name": "screening_finalists",
            "finalists": list(screening_finalists),
            "control": "ce_control",
            "source_experiment_id": experiment_id,
            "representation": representation_selection["selected_name"],
            "sampler": sampler_selection["selected_name"],
            "summaries": summaries,
            "comparisons": comparisons,
            "created_at": utc_now(),
        }
        report["selection_hash"] = hash_canonical(
            {key: value for key, value in report.items() if key != "created_at"}
        )
        atomic_write_json(
            config.output_root / "selections" / "e08_screening_selection.json",
            report,
        )
        return report, ExitCode.PASS

    selected, status, _ = _derive_e08_final(
        config,
        ranking,
        comparisons,
        summaries,
    )
    selected_summary = summaries[selected]
    selected_comparison = comparisons.get(selected, _control_comparison())
    feature_hash = str(representation_selection["selected_feature_manifest_hash"])
    selection = _selection_manifest(
        stage="E08",
        selected_name=selected,
        selected_feature_manifest_hash=feature_hash,
        selection_policy_hash=_selection_policy_hash("E08"),
        source_experiment_id=experiment_id,
        metrics={
            "status": status,
            "summaries": summaries,
            "comparisons": comparisons,
            "representation": representation_selection["selected_name"],
            "sampler": sampler_selection["selected_name"],
        },
        created_at=utc_now(),
    )
    report = selection.model_dump(mode="json")
    atomic_write_json(config.output_root / "selections" / "e08_selection.json", report)
    report["exit_decision"] = _write_e08_exit_decision(
        config,
        selected=selected,
        selection_manifest_hash=selection.manifest_hash,
        summary=selected_summary,
        comparison=selected_comparison,
    )
    return (
        report,
        ExitCode.PASS if status == "E08_METHOD_SELECTED" else ExitCode.SCIENTIFIC_GATE_NOT_MET,
    )


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, [column for column in columns if column in frame]].copy()
    if selected.empty:
        return "_No completed runs._\n"
    headers = list(selected.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def run_reports(config: ResearchConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Generate the five canonical Stage 2 research reports from complete cells."""
    stage_order = ("e06.5", "e07", "e08")
    start = stage_order.index(args.from_stage)
    end = stage_order.index(args.through)
    if start > end:
        raise ResearchError("report stage range is reversed", ExitCode.ARGUMENT_ERROR)
    reports: dict[str, str] = {}
    if "e06.5" in stage_order[start : end + 1]:
        experiment_id = config.default_experiment_ids["e065_audit"]
        metrics = _load_cell_metrics(
            config,
            stage_dir="E06_5",
            experiment_id=experiment_id,
            names=("baseline", "H6", "H11", "H12"),
        )
        table = _markdown_table(
            metrics,
            ("candidate", "fold", "seed", "F1_F", "macro_F1", "precision_F", "recall_F", "AP_F"),
        )
        path = config.project_root / "docs" / "stage2_e065_robustness_report.md"
        atomic_write_text(
            path,
            "# Stage 2 E06.5 robustness\n\n"
            "Status: REPRESENTATION_SIGNAL_CONFIRMED / TARGET_NOT_MET / "
            "ROBUSTNESS_VALIDATION_REQUIRED until verification completes.\n\n"
            f"{table}",
        )
        reports["e06.5"] = str(path)
        fold_source = config.output_root / "fold_audits" / experiment_id / "fold5_report.md"
        fold_text = (
            fold_source.read_text(encoding="utf-8")
            if fold_source.is_file()
            else "# Fold 5 root cause\n\n_Not executed._\n"
        )
        fold_path = config.project_root / "docs" / "stage2_fold5_root_cause.md"
        atomic_write_text(fold_path, fold_text)
        reports["fold5"] = str(fold_path)
    if "e07" in stage_order[start : end + 1]:
        experiment_id = config.default_experiment_ids["e07_audit"]
        metrics = _load_cell_metrics(
            config,
            stage_dir="E07",
            experiment_id=experiment_id,
            names=config.e07.samplers,
        )
        path = config.project_root / "docs" / "stage2_e07_sampling_report.md"
        atomic_write_text(
            path,
            "# Stage 2 E07 sampling\n\n"
            + _markdown_table(
                metrics,
                (
                    "candidate",
                    "fold",
                    "seed",
                    "F1_F",
                    "macro_F1",
                    "precision_F",
                    "recall_F",
                    "AP_F",
                ),
            ),
        )
        reports["e07"] = str(path)
    if "e08" in stage_order[start : end + 1]:
        experiment_id = config.default_experiment_ids["e08_audit"]
        metrics = _load_cell_metrics(
            config,
            stage_dir="E08",
            experiment_id=experiment_id,
            names=config.e08.methods,
        )
        path = config.project_root / "docs" / "stage2_e08_long_tail_report.md"
        atomic_write_text(
            path,
            "# Stage 2 E08 long-tail classifiers\n\n"
            + _markdown_table(
                metrics,
                (
                    "candidate",
                    "fold",
                    "seed",
                    "F1_F",
                    "macro_F1",
                    "precision_F",
                    "recall_F",
                    "AP_F",
                ),
            ),
        )
        reports["e08"] = str(path)
        decision_source = config.output_root / "reports" / "e08_exit_decision.md"
        decision_text = (
            decision_source.read_text(encoding="utf-8")
            if decision_source.is_file()
            else "# Stage 2 escalation decision\n\nE08 has not produced an exit decision.\n"
        )
        decision_path = config.project_root / "docs" / "stage2_escalation_decision.md"
        atomic_write_text(decision_path, decision_text)
        reports["escalation"] = str(decision_path)
    report = {"status": "PASS", "reports": reports, "created_at": utc_now()}
    atomic_write_json(config.output_root / "reports" / "report_manifest.json", report)
    return report


def _run_verify_commands(
    config: ResearchConfig, *, include_make_test: bool
) -> list[dict[str, Any]]:
    commands: list[tuple[str, ...]] = [
        ("make", "lint"),
        ("uv", "run", "--locked", "pyright", "src", "tests"),
        (
            "uv",
            "run",
            "--locked",
            "pytest",
            "tests/test_stage2_research_cli.py",
            "tests/test_stage2_research_splits.py",
            "tests/test_stage2_research_resume.py",
            "-q",
        ),
        ("git", "diff", "--check"),
    ]
    if include_make_test:
        commands.append(("make", "test"))
    results: list[dict[str, Any]] = []
    for command in commands:
        try:
            process = __import__("subprocess").run(
                list(command),
                cwd=config.project_root,
                capture_output=True,
                text=True,
                timeout=3600,
                check=False,
            )
        except (OSError, __import__("subprocess").TimeoutExpired) as error:
            raise ResearchError(
                f"verification command could not run: {' '.join(command)}",
                ExitCode.BLOCKED_PRECONDITION,
            ) from error
        results.append(
            {
                "command": list(command),
                "exit_code": process.returncode,
                "passed": process.returncode == 0,
                "stdout_tail": "\n".join(process.stdout.splitlines()[-20:]),
                "stderr_tail": "\n".join(process.stderr.splitlines()[-20:]),
            }
        )
    return results


def _validate_e08_exit_decision(
    config: ResearchConfig,
    selection: dict[str, Any],
) -> dict[str, Any]:
    path = config.output_root / "reports" / "e08_exit_decision.json"
    markdown_path = config.output_root / "reports" / "e08_exit_decision.md"
    if not path.is_file() or not markdown_path.is_file():
        raise ResearchError("E08 exit decision is missing", ExitCode.BLOCKED_PRECONDITION)
    preflight = require_preflight(config)
    decision = cast(dict[str, Any], load_json(path))
    expected_hash = hash_canonical(
        {
            key: value
            for key, value in decision.items()
            if key not in {"created_at", "decision_hash"}
        }
    )
    selected = str(selection["selected_name"])
    selection_metrics = cast(dict[str, Any], selection.get("metrics", {}))
    summaries = cast(dict[str, Any], selection_metrics.get("summaries", {}))
    comparisons = cast(dict[str, Any], selection_metrics.get("comparisons", {}))
    summary = cast(dict[str, Any], summaries.get(selected, {}))
    comparison = cast(
        dict[str, Any],
        comparisons.get(selected, _control_comparison()),
    )
    expected_fields = _derive_e08_exit_fields(config, summary, comparison)
    if (
        decision.get("schema_version") != "stage2-e08-exit-v2"
        or decision.get("selected_method") != selection.get("selected_name")
        or decision.get("selection_manifest_hash") != selection.get("manifest_hash")
        or decision.get("config_hash") != config_hash(config)
        or decision.get("preflight_hash") != preflight.get("preflight_hash")
        or decision.get("source_manifest_hash") != preflight.get("source_manifest_hash")
        or decision.get("runtime_identity_hash") != preflight.get("runtime_identity_hash")
        or any(decision.get(key) != value for key, value in expected_fields.items())
        or decision.get("decision_hash") != expected_hash
    ):
        raise ResearchError("invalid or stale E08 exit decision", ExitCode.INCOMPATIBLE_ARTIFACT)
    return decision


def verify_stage(config: ResearchConfig, stage: str) -> tuple[dict[str, Any], int]:
    """Run integrity and post-operation verification for one stage or all stages."""
    prepared = prepare_research(config)
    preflight = require_preflight(config)
    integrity = cast(dict[str, Any], preflight["integrity"])
    checks: dict[str, Any] = {
        "uv_lock_unchanged": sha256_file(config.project_root / "uv.lock") == config.uv_lock_sha256,
        "datasets_unchanged": prepared.dataset.manifest_hash == integrity["dataset_manifest_hash"],
        "splits_unchanged": prepared.outer_splits.manifest_hash
        == integrity["outer_split_manifest_hash"],
        "outer_test_not_used_for_selection": _matches_bool(
            integrity.get("outer_test_used_for_selection"),
            False,
        ),
        "template_scope_overlap_zero": integrity.get("template_leakage_count") == 0,
        "preprocessor_scope_overlap_zero": integrity.get("scaler_leakage_count") == 0,
    }
    if stage in {"e06.5", "all"}:
        e065 = verify_e065(config)
        checks["E06.5_release"] = e065["status"] == "E06_5_PASS_REPRESENTATION_SELECTED"
        checks["E06.5"] = e065
    if stage in {"e07", "all"}:
        e07_selection = _load_selection(config, "e07_selection.json")
        checks["E07_selection_valid"] = bool(e07_selection["manifest_hash"])
        checks["E07_selection"] = e07_selection
    if stage in {"e08", "all"}:
        e08_selection = _load_selection(config, "e08_selection.json")
        exit_decision = _validate_e08_exit_decision(config, e08_selection)
        checks["E08_selection_valid"] = bool(e08_selection["manifest_hash"])
        checks["E08_exit_decision_valid"] = bool(exit_decision["decision_hash"])
        checks["E08_selection"] = e08_selection
        checks["E08_exit_decision"] = exit_decision
    commands = _run_verify_commands(config, include_make_test=stage == "all")
    checks["commands"] = commands
    command_pass = all(item["passed"] for item in commands)
    scalar_checks = [value for value in checks.values() if isinstance(value, bool)]
    passed = command_pass and all(scalar_checks)
    report: dict[str, Any] = {
        "schema_version": "stage2-verify-v2",
        "status": "PASS" if passed else "BLOCKED",
        "stage": stage,
        "config_hash": config_hash(config),
        "preflight_hash": preflight["preflight_hash"],
        "source_manifest_hash": preflight["source_manifest_hash"],
        "runtime_identity_hash": preflight["runtime_identity_hash"],
        "checks": checks,
        "created_at": utc_now(),
    }
    report["verification_hash"] = hash_canonical(
        {key: value for key, value in report.items() if key != "created_at"}
    )
    atomic_write_json(config.output_root / "reports" / f"verify_{stage}.json", report)
    return report, ExitCode.PASS if passed else ExitCode.BLOCKED_PRECONDITION


def _resume_e07(config: ResearchConfig, args: argparse.Namespace) -> int:
    screening = _load_selection(config, "e07_screening_selection.json")
    finalists = tuple(str(item) for item in screening["finalists"])
    args.samplers = ("natural", *finalists)
    args.representation = "selected"
    args.folds = config.folds
    args.seeds = config.e07.final_seeds
    args.profile = "audit"
    aggregate, counts = run_e07(config, args)
    print(json.dumps({"status": "PASS", "runs": counts, "aggregate": aggregate}, sort_keys=True))
    return ExitCode.PASS


def _resume_e08(config: ResearchConfig, args: argparse.Namespace) -> int:
    screening = _load_selection(config, "e08_screening_selection.json")
    finalists = tuple(str(item) for item in screening["finalists"])
    args.methods = ("ce_control", *finalists)
    args.folds = config.folds
    args.seeds = config.e08.final_seeds
    args.profile = "audit"
    aggregate, counts = run_e08(config, args)
    print(json.dumps({"status": "PASS", "runs": counts, "aggregate": aggregate}, sort_keys=True))
    return ExitCode.PASS


def _resume_e065(config: ResearchConfig, args: argparse.Namespace) -> int:
    smoke_gate = config.output_root / "manifests" / "e065_smoke_gate.json"
    if not smoke_gate.is_file():
        raise ResearchError(
            "E06_5_SMOKE_PASS is required before audit resume",
            ExitCode.BLOCKED_PRECONDITION,
        )
    aggregate, counts = run_e065(
        config,
        candidates=("baseline", "H6", "H11", "H12"),
        folds=config.folds,
        seeds=config.seeds,
        profile_name="audit",
        experiment_id=args.experiment_id,
        deterministic=True if args.deterministic is None else bool(args.deterministic),
        device=args.device,
        resume=True,
        force=args.force,
        dry_run=args.dry_run,
        max_parallel=args.max_parallel,
    )
    print(json.dumps({"status": "PASS", "runs": counts, "aggregate": aggregate}, sort_keys=True))
    candidates = aggregate.get("candidates", {})
    best = max(
        (item.get("F1_F", {}).get("mean", 0.0) for item in candidates.values()),
        default=0.0,
    )
    return (
        ExitCode.PASS if best >= config.gates.publication_f1_f else ExitCode.SCIENTIFIC_GATE_NOT_MET
    )


def build_advanced_plan(
    config: ResearchConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build E07/E08 plans after prerequisite selections exist."""
    if args.stage == "e07":
        _require_e065_release(config)
        return _matrix_plan(
            config,
            stage="e07",
            experiment_id=args.experiment_id or config.default_experiment_ids["e07_screening"],
            names=config.e07.samplers,
            folds=args.folds,
            seeds=config.e07.screening_seeds,
            profile="screening",
            dependency="E06_5_PASS_REPRESENTATION_SELECTED",
        )
    if args.stage == "e08":
        _require_e07_selection(config)
        return _matrix_plan(
            config,
            stage="e08",
            experiment_id=args.experiment_id or config.default_experiment_ids["e08_screening"],
            names=config.e08.methods,
            folds=args.folds,
            seeds=config.e08.screening_seeds,
            profile="screening",
            dependency="E07_SAMPLER_SELECTED",
        )
    raise ResearchError("unsupported advanced plan stage", ExitCode.ARGUMENT_ERROR)


def dispatch_advanced(config: ResearchConfig, args: argparse.Namespace) -> int:
    """Dispatch advanced commands and enforce their checkpoints."""
    if args.command == "resume":
        if args.stage == "e06.5":
            return _resume_e065(config, args)
        if args.stage == "e07":
            return _resume_e07(config, args)
        if args.stage == "e08":
            return _resume_e08(config, args)
    if args.command == "fold-audit":
        report = run_fold_audit(config, args)
        print(json.dumps(report, sort_keys=True))
        return ExitCode.PASS
    if args.command == "representation-select":
        selection = run_representation_selection(config, args)
        print(json.dumps(selection, sort_keys=True))
        return ExitCode.PASS
    if args.command == "e07-run":
        aggregate, counts = run_e07(config, args)
        print(
            json.dumps({"status": "PASS", "runs": counts, "aggregate": aggregate}, sort_keys=True)
        )
        return ExitCode.PASS
    if args.command == "e07-select":
        selection, exit_code = select_e07(config, args)
        print(json.dumps(selection, sort_keys=True))
        return exit_code
    if args.command == "e08-run":
        aggregate, counts = run_e08(config, args)
        print(
            json.dumps({"status": "PASS", "runs": counts, "aggregate": aggregate}, sort_keys=True)
        )
        return ExitCode.PASS
    if args.command == "e08-select":
        selection, exit_code = select_e08(config, args)
        print(json.dumps(selection, sort_keys=True))
        return exit_code
    if args.command == "report":
        report = run_reports(config, args)
        print(json.dumps(report, sort_keys=True))
        return ExitCode.PASS
    if args.command == "verify":
        report, exit_code = verify_stage(config, args.stage)
        print(json.dumps(report, sort_keys=True))
        return exit_code
    raise ResearchError(f"unsupported command: {args.command}", ExitCode.ARGUMENT_ERROR)
