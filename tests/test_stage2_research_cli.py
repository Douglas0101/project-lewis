"""Canonical Stage 2 research CLI and contract tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.cli.stage2_research import _build_parser
from src.stage2_research import advanced_workflows, workflows
from src.stage2_research.config import config_hash, load_research_config
from src.stage2_research.contracts import (
    CandidateConfig,
    ExitCode,
    ResearchError,
    RunManifest,
)
from src.stage2_research.features import candidate_static_manifest
from src.stage2_research.integrity import atomic_write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "stage2_research.yaml"
EXPECTED_CANDIDATES = ("baseline", "H6", "H11", "H12")
EXPECTED_FOLDS = (1, 2, 3, 4, 5)
EXPECTED_SEEDS = (17, 29, 43, 71, 101)
SMOKE_FOLDS = (1,)
SMOKE_SEEDS = (17,)


def test_research_config_freezes_candidate_matrix() -> None:
    config = load_research_config(CONFIG_PATH)

    assert config.candidates["baseline"].fusion_template_count == 0
    assert config.candidates["H6"].fusion_template_count == 8
    assert config.candidates["H11"].fusion_template_count == 16
    assert config.candidates["H12"].fusion_template_count == 24
    assert config.seeds == EXPECTED_SEEDS
    assert config.folds == EXPECTED_FOLDS
    assert config.profiles["audit"].deterministic
    assert config.profiles["audit"].max_parallel == 1
    assert config.gates.publication_f1_f == 0.50
    assert len(config_hash(config)) == 64


def test_candidate_contract_rejects_feature_or_template_drift() -> None:
    with pytest.raises(ValidationError):
        CandidateConfig(
            name="H11",
            fusion_template_count=24,
            feature_families=("base16", "causal_rr_h3", "class_templates_h5"),
            complexity_rank=2,
        )

    with pytest.raises(ValidationError):
        CandidateConfig(
            name="H6",
            fusion_template_count=8,
            feature_families=("base16",),
            complexity_rank=1,
        )


def test_candidate_feature_manifests_are_content_addressed_and_distinct() -> None:
    config = load_research_config(CONFIG_PATH)

    h6 = candidate_static_manifest(config.candidates["H6"])
    h11 = candidate_static_manifest(config.candidates["H11"])
    h12 = candidate_static_manifest(config.candidates["H12"])

    assert h6["manifest_hash"] != h11["manifest_hash"]
    assert h11["manifest_hash"] != h12["manifest_hash"]
    assert len(h6["feature_names"]) == 54
    assert h6["rr_context_source"] == "stage2_filtered"


def test_parser_exposes_every_canonical_subcommand() -> None:
    parser = _build_parser()
    commands = (
        "preflight",
        "status",
        "plan",
        "e065-run",
        "fold-audit",
        "representation-select",
        "e07-run",
        "e07-select",
        "e08-run",
        "e08-select",
        "report",
        "verify",
        "resume",
    )

    for command in commands:
        required: list[str] = []
        if command in {"plan", "fold-audit", "representation-select"}:
            required = ["--stage", "e06.5"]
        elif command in {"e07-select", "e08-select"}:
            required = ["--phase", "screening"]
        elif command == "report":
            required = ["--from", "e06.5", "--through", "e08"]
        elif command == "verify":
            required = ["--stage", "all"]
        elif command == "resume":
            required = ["--stage", "e06.5"]
        args = parser.parse_args([command, *required])
        assert args.command == command


def test_execution_defaults_are_safe() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "e065-run",
            "--candidates",
            "baseline,H6,H11,H12",
            "--folds",
            "1",
            "--seeds",
            "17",
            "--profile",
            "smoke",
        ]
    )

    assert args.resume
    assert not args.force
    assert not args.dry_run
    assert args.max_parallel == 1
    assert args.candidates == EXPECTED_CANDIDATES
    assert args.folds == SMOKE_FOLDS
    assert args.seeds == SMOKE_SEEDS


def test_run_manifest_cannot_mark_outer_test_as_selection_source() -> None:
    payload = {
        "experiment_stage": "E06.5",
        "experiment_id": "test",
        "candidate": "baseline",
        "fold": 1,
        "seed": 17,
        "model_seed": 17,
        "git_head": "a" * 40,
        "git_dirty": False,
        "dataset_manifest_hash": "b" * 64,
        "split_manifest_hash": "c" * 64,
        "feature_manifest_hash": "d" * 64,
        "config_hash": "e" * 64,
        "preflight_hash": "1" * 64,
        "source_manifest_hash": "2" * 64,
        "runtime_identity_hash": "3" * 64,
        "uv_lock_hash": "f" * 64,
        "python_version": "3.12",
        "tensorflow_version": "2.21",
        "keras_version": "3.14",
        "device": "cpu",
        "deterministic": True,
        "sampling": "natural",
        "loss": "sparse_categorical_crossentropy",
        "architecture": "minimal_mlp_128",
        "early_stopping_source": "inner_validation",
        "outer_test_used_for_selection": True,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "",
        "status": "RUNNING",
        "profile": "audit",
        "publication_eligible": True,
        "split_random_state": 42,
        "sampler_random_state": 17,
    }

    with pytest.raises(ValidationError):
        RunManifest.model_validate(payload)


def test_smoke_history_allows_only_structurally_missing_outer_validation() -> None:
    history = pd.DataFrame(
        [
            {
                "phase": "inner_selection",
                "epoch": 1,
                "accuracy": 0.7,
                "loss": 0.8,
                "val_accuracy": 0.6,
                "val_loss": 0.9,
            },
            {
                "phase": "outer_refit",
                "epoch": 1,
                "accuracy": 0.7,
                "loss": 0.8,
                "val_accuracy": np.nan,
                "val_loss": np.nan,
            },
        ]
    )
    assert workflows._history_numeric_contract_valid(history)

    history.loc[0, "val_loss"] = np.nan
    assert not workflows._history_numeric_contract_valid(history)


def test_e065_smoke_contract_requires_exact_canonical_matrix() -> None:
    config = load_research_config(CONFIG_PATH)

    workflows._require_e065_execution_contract(
        config,
        candidates=EXPECTED_CANDIDATES,
        folds=SMOKE_FOLDS,
        seeds=SMOKE_SEEDS,
        profile_name="smoke",
        deterministic=True,
        device="cpu",
        max_parallel=1,
    )

    invalid = (
        (EXPECTED_CANDIDATES[:-1], SMOKE_FOLDS, SMOKE_SEEDS, True, "cpu"),
        (EXPECTED_CANDIDATES, (2,), SMOKE_SEEDS, True, "cpu"),
        (EXPECTED_CANDIDATES, SMOKE_FOLDS, (29,), True, "cpu"),
        (EXPECTED_CANDIDATES, SMOKE_FOLDS, SMOKE_SEEDS, False, "cpu"),
        (EXPECTED_CANDIDATES, SMOKE_FOLDS, SMOKE_SEEDS, True, "gpu"),
    )
    for candidates, folds, seeds, deterministic, device in invalid:
        with pytest.raises(ResearchError) as captured:
            workflows._require_e065_execution_contract(
                config,
                candidates=candidates,
                folds=folds,
                seeds=seeds,
                profile_name="smoke",
                deterministic=deterministic,
                device=device,
                max_parallel=1,
            )
        assert captured.value.exit_code == ExitCode.INVALID_EXPERIMENT


def test_e065_audit_cannot_start_without_validated_smoke_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_research_config(CONFIG_PATH)
    identity = {
        "preflight_hash": "1" * 64,
        "source_manifest_hash": "2" * 64,
        "runtime_identity_hash": "3" * 64,
    }
    monkeypatch.setattr(workflows, "require_preflight", lambda _: identity)

    def reject_gate(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ResearchError("missing gate", ExitCode.BLOCKED_PRECONDITION)

    monkeypatch.setattr(workflows, "validate_e065_smoke_gate", reject_gate)

    with pytest.raises(ResearchError) as captured:
        workflows.run_e065(
            config,
            candidates=EXPECTED_CANDIDATES,
            folds=EXPECTED_FOLDS,
            seeds=EXPECTED_SEEDS,
            profile_name="audit",
            experiment_id="audit-must-not-start",
            deterministic=True,
            device="cpu",
            resume=True,
            force=False,
            dry_run=False,
            max_parallel=1,
        )

    assert captured.value.exit_code == ExitCode.BLOCKED_PRECONDITION


def test_resume_e07_freezes_selected_representation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_research_config(CONFIG_PATH)
    args = argparse.Namespace()
    monkeypatch.setattr(
        advanced_workflows,
        "_load_selection",
        lambda *_args, **_kwargs: {"finalists": ["patient_uniform", "smote"]},
    )

    def fake_run_e07(
        _config: object,
        received: argparse.Namespace,
    ) -> tuple[dict[str, object], dict[str, int]]:
        assert received.representation == "selected"
        assert received.profile == "audit"
        return {}, {"planned": 0, "executed": 0, "resumed": 0, "skipped": 0, "failed": 0}

    monkeypatch.setattr(advanced_workflows, "run_e07", fake_run_e07)

    assert advanced_workflows._resume_e07(config, args) == ExitCode.PASS


def test_selection_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    config = load_research_config(CONFIG_PATH, output_root_override=tmp_path)
    selection = advanced_workflows._selection_manifest(
        stage="E06.5",
        selected_name="H6",
        selected_feature_manifest_hash="4" * 64,
        selection_policy_hash=advanced_workflows._selection_policy_hash("E06.5"),
        source_experiment_id=config.default_experiment_ids["e065_audit"],
        metrics={"candidate_summaries": {}},
    ).model_dump(mode="json")
    selection["selected_name"] = "H11"
    destination = tmp_path / "selections" / "representation_selection.json"
    atomic_write_json(destination, selection)

    with pytest.raises(ResearchError) as captured:
        advanced_workflows._load_selection(config, "representation_selection.json")

    assert captured.value.exit_code == ExitCode.INCOMPATIBLE_ARTIFACT


def test_load_selection_resolves_e065_stage_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: final-selection validation must read cells from E06_5, not E06.5."""
    config = load_research_config(CONFIG_PATH, output_root_override=tmp_path)
    metrics: dict[str, object] = {"candidate_summaries": {}}
    selection = advanced_workflows._selection_manifest(
        stage="E06.5",
        selected_name="H6",
        selected_feature_manifest_hash="4" * 64,
        selection_policy_hash=advanced_workflows._selection_policy_hash("E06.5"),
        source_experiment_id=config.default_experiment_ids["e065_audit"],
        metrics=metrics,
    ).model_dump(mode="json")
    atomic_write_json(
        tmp_path / "selections" / "representation_selection.json", selection
    )
    atomic_write_json(
        tmp_path / "manifests" / "feature_manifests.json",
        {"candidates": {"H6": {"manifest_hash": "4" * 64}}},
    )
    monkeypatch.setattr(
        advanced_workflows, "_validate_fold_audit_report", lambda _config: {}
    )
    captured: dict[str, str] = {}

    def fake_load_cell_metrics(
        _config: object,
        *,
        stage_dir: str,
        experiment_id: str,
        names: object,
    ) -> pd.DataFrame:
        captured["stage_dir"] = stage_dir
        return pd.DataFrame()

    monkeypatch.setattr(
        advanced_workflows, "_load_cell_metrics", fake_load_cell_metrics
    )
    monkeypatch.setattr(
        advanced_workflows,
        "_require_complete_metrics_matrix",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        advanced_workflows,
        "aggregate_experiment",
        lambda *args, **kwargs: {"candidates": {}},
    )
    monkeypatch.setattr(
        advanced_workflows,
        "_derive_representation_decision",
        lambda *args, **kwargs: ("H6", metrics),
    )

    advanced_workflows._load_selection(config, "representation_selection.json")

    assert captured["stage_dir"] == "E06_5"


def test_fold5_report_partition_counts_are_json_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: np.sum values must not leak into the fold5 JSON report.

    The validator compares partition_counts against native ints; np.int64
    serialized via default=str became strings, and plain json.dumps crashed.
    """
    config = load_research_config(CONFIG_PATH, output_root_override=tmp_path)
    monkeypatch.setattr(
        advanced_workflows,
        "require_preflight",
        lambda _config: {
            "preflight_hash": "p" * 64,
            "source_manifest_hash": "s" * 64,
            "runtime_identity_hash": "r" * 64,
        },
    )
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2], dtype=np.int64)
    groups = np.array(
        ["208", "208", "213", "213", "999", "999", "208", "213", "999", "998", "998", "998"],
        dtype=str,
    )
    prepared = SimpleNamespace(
        dataset=SimpleNamespace(labels=labels, groups=groups),
        outer_splits={"fold_5": np.arange(12)},
        inner_splits={"fold_5": np.arange(12)},
    )
    monkeypatch.setattr(advanced_workflows, "prepare_research", lambda _config: prepared)
    monkeypatch.setattr(
        advanced_workflows,
        "split_indices",
        lambda *_args, **_kwargs: (
            np.arange(8),
            np.arange(8, 12),
            np.arange(6),
            np.arange(6, 12),
        ),
    )
    metric_rows = [
        {
            "candidate": candidate,
            "seed": seed,
            "F1_F": 0.25,
            "precision_F": 0.25,
            "recall_F": 0.25,
            "AP_F": 0.25,
            "macro_F1": 0.5,
        }
        for candidate in EXPECTED_CANDIDATES
        for seed in EXPECTED_SEEDS
    ]
    monkeypatch.setattr(
        advanced_workflows,
        "_load_cell_metrics",
        lambda *args, **kwargs: pd.DataFrame(metric_rows),
    )
    monkeypatch.setattr(
        advanced_workflows,
        "_require_complete_metrics_matrix",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        advanced_workflows, "validate_done_marker", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(advanced_workflows, "sha256_file", lambda _path: "h" * 64)

    predictions = pd.DataFrame(
        {
            "y_true": np.array([2, 2, 2], dtype=np.int64),
            "y_pred": np.array([1, 2, 0], dtype=np.int64),
            "margin_F": np.array([-0.1, 0.2, -0.3], dtype=np.float64),
        }
    )
    monkeypatch.setattr(
        advanced_workflows, "_read_predictions", lambda _run_dir: predictions.copy()
    )
    monkeypatch.setattr(advanced_workflows, "_atomic_csv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        advanced_workflows, "_atomic_parquet", lambda *_args, **_kwargs: None
    )

    args = argparse.Namespace(
        fold=5,
        candidates=EXPECTED_CANDIDATES,
        seeds=EXPECTED_SEEDS,
        experiment_id=None,
    )
    report = advanced_workflows.run_fold_audit(config, args)

    # dispatch print path (json.dumps without default) must not raise.
    json.dumps(report, sort_keys=True)

    numeric = {
        key: value
        for row in report["partition_counts"]
        for key, value in row.items()
        if key != "partition"
    }
    assert numeric, "partition_counts vazio"
    assert all(type(value) is int for value in numeric.values())

    written = json.loads(
        (
            tmp_path
            / "fold_audits"
            / config.default_experiment_ids["e065_audit"]
            / "fold5_report.json"
        ).read_text(encoding="utf-8")
    )
    assert all(
        type(value) is int
        for row in written["partition_counts"]
        for key, value in row.items()
        if key != "partition"
    )
