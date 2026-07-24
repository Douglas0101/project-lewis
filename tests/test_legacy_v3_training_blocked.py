import subprocess
import sys
from pathlib import Path


def _run(project_root: Path, script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script, *args],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_legacy_matrix_cli_blocks_before_training() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = _run(project_root, "scripts/run_stage1_matrix_v3.py", "--campaign")
    assert result.returncode == 10
    assert "LEGACY_V3_TRAINING_BLOCKED" in result.stderr
    assert "src.cli.advanced_training_v3" in result.stderr


def test_legacy_sanity_cell_blocks_before_training() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = _run(project_root, "scripts/run_stage1_sanity_cell_v3.py")
    assert result.returncode == 10
    assert "LEGACY_V3_TRAINING_BLOCKED" in result.stderr


def test_legacy_split_builders_block_before_writes() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for script in (
        "scripts/build_frozen_splits_v3.py",
        "scripts/build_afdb_frozen_splits_v3.py",
    ):
        result = _run(project_root, script)
        assert result.returncode == 10
        assert "LEGACY_V3_SPLIT_BUILDER_BLOCKED" in result.stderr
