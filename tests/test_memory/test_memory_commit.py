"""Testes para o script memory_commit.py."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.tracking.db import init_schema
from src.tracking.repositories import ExperimentRepository, RunRepository
from src.tracking.schemas import ExperimentCreate, RunCreate


@pytest.fixture
def temp_db(tmp_path: Path):
    """Caminho de banco temporário para os testes."""
    db_path = tmp_path / "test_memory.db"
    os.environ["LEWIS_TRACKING_DB"] = str(db_path)
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    init_schema(engine)
    with Session(engine) as session:
        experiment = ExperimentRepository(session).create(
            ExperimentCreate(name="memory_commit_test", stage="stage1")
        )
        run = RunRepository(session).create(RunCreate(experiment_id=experiment.id, run_type="test"))
        session.commit()
        run_id = run.id
    engine.dispose()
    yield str(db_path), run_id
    if "LEWIS_TRACKING_DB" in os.environ:
        del os.environ["LEWIS_TRACKING_DB"]


def _run_script(run_id: int, path: str, artifact_type: str) -> subprocess.CompletedProcess:
    """Executa memory_commit.py via subprocess e retorna o resultado."""
    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
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
        env=env,
    )


def test_memory_commit_script_registers_artifact(temp_db) -> None:
    """Script deve registrar artefato e imprimir artifact_id."""
    db_path, run_id = temp_db
    os.environ["LEWIS_TRACKING_DB"] = db_path
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello artifact")
        artifact_path = f.name

    try:
        result = _run_script(run_id, artifact_path, "test")
        assert result.returncode == 0, result.stderr
        assert "artifact_id=" in result.stdout
        artifact_id = result.stdout.strip().split("=")[1]
        assert artifact_id.isdigit()
    finally:
        os.unlink(artifact_path)


def test_memory_commit_script_rejects_missing_file(temp_db) -> None:
    """Script deve falhar quando o arquivo não existe."""
    db_path, run_id = temp_db
    os.environ["LEWIS_TRACKING_DB"] = db_path
    result = _run_script(run_id, "nonexistent_file_12345.txt", "test")
    assert result.returncode != 0
