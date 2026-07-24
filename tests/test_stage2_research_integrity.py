"""Security and ownership contracts for Stage 2 run directories."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.stage2_research.config import load_research_config
from src.stage2_research.contracts import ExitCode, ResearchConfig, ResearchError
from src.stage2_research.integrity import reset_incomplete_run, run_lock
from src.stage2_research.training import stage_run_dir


@pytest.fixture
def research_config(tmp_path: Path) -> ResearchConfig:
    """Use the canonical contract with an isolated experiment root."""
    config = load_research_config(Path("config/stage2_research.yaml"))
    return config.model_copy(update={"output_root": tmp_path / "experiments"})


def _assert_blocked(error: pytest.ExceptionInfo[ResearchError]) -> None:
    assert error.value.exit_code == ExitCode.BLOCKED_PRECONDITION


def test_stage_run_dir_accepts_canonical_segments(research_config: ResearchConfig) -> None:
    run_dir = stage_run_dir(
        research_config,
        stage="e06.5",
        experiment_id="e065-smoke-test",
        candidate="H6",
        fold=1,
        seed=17,
    )

    assert run_dir == (
        research_config.output_root / "E06_5" / "e065-smoke-test" / "H6" / "fold_1" / "seed_17"
    )


@pytest.mark.parametrize(
    ("experiment_id", "candidate"),
    [
        ("", "H6"),
        ("../escape", "H6"),
        ("nested/experiment", "H6"),
        ("nested\\experiment", "H6"),
        ("/tmp/absolute", "H6"),
        (".", "H6"),
        ("e065-safe", ""),
        ("e065-safe", "../escape"),
        ("e065-safe", "nested/candidate"),
        ("e065-safe", "nested\\candidate"),
        ("e065-safe", "/tmp/absolute"),
        ("e065-safe", "."),
    ],
)
def test_stage_run_dir_rejects_unsafe_segments(
    research_config: ResearchConfig,
    experiment_id: str,
    candidate: str,
) -> None:
    with pytest.raises(ResearchError) as captured:
        stage_run_dir(
            research_config,
            stage="e06.5",
            experiment_id=experiment_id,
            candidate=candidate,
            fold=1,
            seed=17,
        )

    _assert_blocked(captured)


def test_run_lock_is_owned_and_removed_after_success(tmp_path: Path) -> None:
    output_root = tmp_path / "experiments"
    run_dir = output_root / "E06_5" / "experiment" / "H6" / "fold_1" / "seed_17"
    lock_path = run_dir / ".RUNNING.lock"

    with run_lock(run_dir, output_root=output_root):
        assert lock_path.exists()
        assert lock_path.stat().st_mode & 0o777 == 0o600
        assert lock_path.read_text(encoding="ascii").startswith("pid=")

    assert not lock_path.exists()


def test_run_lock_owner_cleans_up_after_body_error(tmp_path: Path) -> None:
    output_root = tmp_path / "experiments"
    run_dir = output_root / "E06_5" / "experiment" / "H6" / "fold_1" / "seed_17"
    lock_path = run_dir / ".RUNNING.lock"

    with (
        pytest.raises(RuntimeError, match="body failed"),
        run_lock(
            run_dir,
            output_root=output_root,
        ),
    ):
        assert lock_path.exists()
        raise RuntimeError("body failed")

    assert not lock_path.exists()


def test_run_lock_does_not_reclassify_body_file_exists_error(tmp_path: Path) -> None:
    output_root = tmp_path / "experiments"
    run_dir = output_root / "E06_5" / "experiment" / "H6" / "fold_1" / "seed_17"
    lock_path = run_dir / ".RUNNING.lock"

    with (
        pytest.raises(FileExistsError, match="body collision"),
        run_lock(run_dir, output_root=output_root),
    ):
        raise FileExistsError("body collision")

    assert not lock_path.exists()


def test_failed_lock_contender_cannot_delete_owner_lock(tmp_path: Path) -> None:
    output_root = tmp_path / "experiments"
    run_dir = output_root / "E06_5" / "experiment" / "H6" / "fold_1" / "seed_17"
    lock_path = run_dir / ".RUNNING.lock"

    with run_lock(run_dir, output_root=output_root):
        owner_contents = lock_path.read_bytes()
        with (
            pytest.raises(ResearchError) as captured,
            run_lock(
                run_dir,
                output_root=output_root,
            ),
        ):
            pytest.fail("a second lock owner must never enter")
        _assert_blocked(captured)
        assert lock_path.exists()
        assert lock_path.read_bytes() == owner_contents

    assert not lock_path.exists()


@pytest.mark.parametrize("relative", [Path("..") / "escape", Path(".")])
def test_run_lock_rejects_non_descendant_paths(tmp_path: Path, relative: Path) -> None:
    output_root = tmp_path / "experiments"
    run_dir = (output_root / relative).resolve()

    with (
        pytest.raises(ResearchError) as captured,
        run_lock(
            run_dir,
            output_root=output_root,
        ),
    ):
        pytest.fail("unsafe run path must never be entered")

    _assert_blocked(captured)
    assert not (run_dir / ".RUNNING.lock").exists()


def test_run_lock_rejects_symlink_escape(tmp_path: Path) -> None:
    output_root = tmp_path / "experiments"
    outside = tmp_path / "outside"
    output_root.mkdir()
    outside.mkdir()
    link = output_root / "linked-outside"
    link.symlink_to(outside, target_is_directory=True)
    run_dir = link / "cell"

    with (
        pytest.raises(ResearchError) as captured,
        run_lock(
            run_dir,
            output_root=output_root,
        ),
    ):
        pytest.fail("symlink escape must never be entered")

    _assert_blocked(captured)
    assert not (outside / "cell").exists()


def test_reset_incomplete_run_is_root_bound_and_preserves_requested_files(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiments"
    run_dir = output_root / "E06_5" / "experiment" / "H6" / "fold_1" / "seed_17"
    nested = run_dir / "partial"
    nested.mkdir(parents=True)
    (nested / "payload.bin").write_bytes(b"partial")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "config_resolved.json").write_text("{}", encoding="utf-8")

    reset_incomplete_run(run_dir, output_root=output_root, keep_manifest=True)

    assert not nested.exists()
    assert not (run_dir / "metrics.json").exists()
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "config_resolved.json").exists()


def test_reset_incomplete_run_rejects_outside_root_without_deleting(tmp_path: Path) -> None:
    output_root = tmp_path / "experiments"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "must-remain.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ResearchError) as captured:
        reset_incomplete_run(outside, output_root=output_root)

    _assert_blocked(captured)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_reset_incomplete_run_unlinks_child_symlink_without_following_it(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiments"
    run_dir = output_root / "E06_5" / "experiment" / "H6" / "fold_1" / "seed_17"
    outside = tmp_path / "outside"
    run_dir.mkdir(parents=True)
    outside.mkdir()
    sentinel = outside / "must-remain.txt"
    sentinel.write_text("keep", encoding="utf-8")
    linked_directory = run_dir / "linked-directory"
    linked_directory.symlink_to(outside, target_is_directory=True)

    reset_incomplete_run(run_dir, output_root=output_root)

    assert not linked_directory.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_reset_incomplete_run_never_overwrites_done(tmp_path: Path) -> None:
    output_root = tmp_path / "experiments"
    run_dir = output_root / "E06_5" / "experiment" / "H6" / "fold_1" / "seed_17"
    run_dir.mkdir(parents=True)
    (run_dir / "DONE").write_text("{}", encoding="utf-8")
    sentinel = run_dir / "must-remain.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ResearchError) as captured:
        reset_incomplete_run(run_dir, output_root=output_root)

    assert captured.value.exit_code == ExitCode.INCOMPATIBLE_ARTIFACT
    assert sentinel.read_text(encoding="utf-8") == "keep"
