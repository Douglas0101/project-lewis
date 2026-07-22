"""Cell-level DONE and resume integrity tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.stage2_research.config import load_research_config
from src.stage2_research.contracts import ResearchError, RunManifest
from src.stage2_research.integrity import (
    artifact_hashes,
    atomic_write_json,
    atomic_write_text,
    load_json,
    sha256_file,
    validate_done_marker,
    write_done_marker,
)
from src.stage2_research.training import REQUIRED_RUN_ARTIFACTS, _cell_config_payload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "stage2_research.yaml"


def _manifest(config_hash: str) -> RunManifest:
    return RunManifest(
        experiment_stage="E06.5",
        experiment_id="resume-test",
        candidate="baseline",
        fold=1,
        seed=17,
        model_seed=17,
        git_head="a" * 40,
        git_dirty=False,
        dataset_manifest_hash="b" * 64,
        split_manifest_hash="c" * 64,
        feature_manifest_hash="d" * 64,
        config_hash=config_hash,
        preflight_hash="1" * 64,
        source_manifest_hash="2" * 64,
        runtime_identity_hash="3" * 64,
        uv_lock_hash="f" * 64,
        python_version="3.12",
        tensorflow_version="2.21",
        keras_version="3.14",
        device="cpu",
        deterministic=True,
        sampling="natural",
        loss="sparse_categorical_crossentropy",
        architecture="minimal_mlp_128",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:01:00+00:00",
        status="PASS",
        profile="smoke",
        publication_eligible=False,
        split_random_state=42,
        sampler_random_state=17,
    )


def _complete_run(run_dir: Path, config_hash: str) -> RunManifest:
    run_dir.mkdir(parents=True)
    for name in REQUIRED_RUN_ARTIFACTS:
        atomic_write_text(run_dir / name, f"artifact={name}\n")
    manifest = _manifest(config_hash)
    hashes = artifact_hashes(run_dir, REQUIRED_RUN_ARTIFACTS)
    manifest = manifest.model_copy(update={"artifact_hashes": hashes})
    atomic_write_json(run_dir / "run_manifest.json", manifest.model_dump(mode="json"))
    write_done_marker(run_dir, manifest, REQUIRED_RUN_ARTIFACTS)
    return manifest


def test_done_marker_validates_matching_completed_run(tmp_path: Path) -> None:
    config_hash = "e" * 64
    run_dir = tmp_path / "run"
    _complete_run(run_dir, config_hash)

    marker = validate_done_marker(run_dir, expected_config_hash=config_hash)

    assert marker is not None
    assert marker.config_hash == config_hash
    assert set(marker.artifact_hashes) == set(REQUIRED_RUN_ARTIFACTS)


def test_done_marker_rejects_config_drift(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _complete_run(run_dir, "e" * 64)

    with pytest.raises(ResearchError):
        validate_done_marker(run_dir, expected_config_hash="0" * 64)


def test_done_marker_rejects_corrupted_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _complete_run(run_dir, "e" * 64)
    atomic_write_text(run_dir / "metrics.json", "corrupted\n")

    with pytest.raises(ResearchError):
        validate_done_marker(run_dir)


def test_done_marker_rejects_omitted_artifact_entry(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _complete_run(run_dir, "e" * 64)
    marker = load_json(run_dir / "DONE")
    marker["artifact_hashes"].pop("metrics.json")
    atomic_write_json(run_dir / "DONE", marker)

    with pytest.raises(ResearchError):
        validate_done_marker(run_dir)


def test_done_marker_rejects_nonpassing_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    manifest = _complete_run(run_dir, "e" * 64)
    failed = manifest.model_copy(update={"status": "FAILED"})
    atomic_write_json(run_dir / "run_manifest.json", failed.model_dump(mode="json"))
    marker = load_json(run_dir / "DONE")
    marker["run_manifest_hash"] = sha256_file(run_dir / "run_manifest.json")
    atomic_write_json(run_dir / "DONE", marker)

    with pytest.raises(ResearchError):
        validate_done_marker(run_dir)


def test_done_marker_rejects_manifest_artifact_map_disagreement(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    manifest = _complete_run(run_dir, "e" * 64)
    altered = manifest.model_copy(
        update={"artifact_hashes": {**manifest.artifact_hashes, "metrics.json": "0" * 64}}
    )
    atomic_write_json(run_dir / "run_manifest.json", altered.model_dump(mode="json"))
    marker = load_json(run_dir / "DONE")
    marker["run_manifest_hash"] = sha256_file(run_dir / "run_manifest.json")
    atomic_write_json(run_dir / "DONE", marker)

    with pytest.raises(ResearchError):
        validate_done_marker(run_dir)


def test_done_marker_rejects_missing_required_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _complete_run(run_dir, "e" * 64)
    (run_dir / "metrics.json").unlink()

    with pytest.raises(ResearchError):
        validate_done_marker(run_dir)


def test_cell_identity_binds_preflight_source_and_runtime() -> None:
    config = load_research_config(CONFIG_PATH)

    def payload(preflight_hash: str, source_hash: str, runtime_hash: str) -> dict[str, object]:
        return _cell_config_payload(
            config,
            stage="e06.5",
            experiment_id="identity-test",
            candidate="baseline",
            representation="baseline",
            fold=1,
            seed=17,
            profile_name="smoke",
            deterministic=True,
            device="cpu",
            sampler="natural",
            method="ce_control",
            preflight_hash=preflight_hash,
            source_manifest_hash=source_hash,
            runtime_identity_hash=runtime_hash,
            feature_manifest_hash="4" * 64,
            split_manifest_hash="5" * 64,
        )

    reference = payload("1" * 64, "2" * 64, "3" * 64)
    assert payload("a" * 64, "2" * 64, "3" * 64) != reference
    assert payload("1" * 64, "b" * 64, "3" * 64) != reference
    assert payload("1" * 64, "2" * 64, "c" * 64) != reference


def test_run_without_done_is_resumable(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(
        run_dir / "run_manifest.json",
        _manifest("e" * 64).model_copy(update={"status": "INTERRUPTED"}).model_dump(mode="json"),
    )

    assert validate_done_marker(run_dir) is None
