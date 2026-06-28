"""Testes para o script memory_commit.py."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_db(tmp_path: Path) -> str:
    """Caminho de banco temporario para os testes."""
    db_path = tmp_path / "test_memory.db"
    os.environ["LEWIS_TRACKING_DB"] = str(db_path)
    yield str(db_path)
    if "LEWIS_TRACKING_DB" in os.environ:
        del os.environ["LEWIS_TRACKING_DB"]


def _run_script(run_id: int, path: str, artifact_type: str) -> subprocess.CompletedProcess:
    """Executa memory_commit.py via subprocess e retorna o resultado."""
    project_root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [
            sys.executable,
            "scripts/memory_commit.py",
            "--run-id",
            str(run_id),
            "--path",
            path,
            "--type",
            artifact_type,
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
    )


def test_memory_commit_script_registers_artifact(temp_db: str) -> None:
    """Script deve registrar artefato e imprimir artifact_id."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello artifact")
        artifact_path = f.name

    try:
        result = _run_script(1, artifact_path, "test")
        assert result.returncode == 0, result.stderr
        assert "artifact_id=" in result.stdout
        artifact_id = result.stdout.strip().split("=")[1]
        assert artifact_id.isdigit()
    finally:
        os.unlink(artifact_path)


def test_memory_commit_script_rejects_missing_file(temp_db: str) -> None:
    """Script deve falhar quando o arquivo nao existe."""
    result = _run_script(1, "nonexistent_file_12345.txt", "test")
    assert result.returncode != 0
