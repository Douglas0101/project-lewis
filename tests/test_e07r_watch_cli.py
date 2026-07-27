"""Smoke tests for the E07R training-watch CLI snapshot mode."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "e07r_watch.py"


def _run_once(stage: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--stage", stage, "--once"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )


def test_watch_once_e065_renders_matrix_snapshot() -> None:
    process = _run_once("e065")

    assert process.returncode in (0, 1)
    assert "E06.5-PD" in process.stdout
    assert "E07R training watch" in process.stdout
    assert "células" in process.stdout
    assert "matriz braço × fold" in process.stdout
    assert process.stderr == ""


def test_watch_once_e07_renders_empty_matrix() -> None:
    process = _run_once("e07")

    assert process.returncode in (0, 1)
    assert "E07-PD" in process.stdout
    assert "0/150" in process.stdout
    # braços exibidos com rótulos curtos + legenda com os nomes completos
    assert "S0" in process.stdout
    assert "S5" in process.stdout
    assert "S0=pd_s0_natural" in process.stdout
    assert "S5=pd_s5_smote_feature" in process.stdout
