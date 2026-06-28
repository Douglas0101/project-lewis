"""Testes do CLI de tracking."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tracking.cli import main


@pytest.fixture
def tmp_db_path(tmp_path: Path, monkeypatch) -> Path:
    """Isola banco de tracking em diretório temporário."""
    db = tmp_path / "lewis_metrics.db"
    monkeypatch.setenv("LEWIS_TRACKING_DB", str(db))
    return db


def test_cli_init_and_list(tmp_db_path: Path) -> None:
    assert main(["init"]) == 0
    assert tmp_db_path.exists()

    assert main(["list-experiments"]) == 0


def test_cli_alerts_empty(tmp_db_path: Path) -> None:
    main(["init"])
    assert main(["alerts"]) == 0


def test_cli_export_nonexistent(tmp_db_path: Path, capsys) -> None:
    main(["init"])
    assert main(["export", "--experiment-id", "999"]) == 1
    captured = capsys.readouterr()
    assert "Experimento 999 não encontrado" in captured.out
