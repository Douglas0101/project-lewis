"""Testes de integração end-to-end do CLI (QG-C11-08)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from src.knowledge.cli import main


@pytest.mark.qg_c11
class TestCliIntegration:
    """QG-C11-08: reindex, query e status funcionam via CLI argparse."""

    def test_cli_reindex_builds_index(
        self,
        mini_index_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(["reindex"])
        captured = capsys.readouterr()

        assert exit_code == 0, f"reindex falhou: {captured.err}"
        assert mini_index_env["db_path"].exists()

    def test_cli_query_with_layer_filter(
        self,
        mini_index_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        main(["reindex"])
        capsys.readouterr()

        exit_code = main(["query", "STM32 firmware", "--layer", "C08", "--k", "3"])
        captured = capsys.readouterr()

        assert exit_code == 0, f"query falhou: {captured.err}"
        assert "C08" in captured.out

    def test_cli_status_with_index(
        self,
        mini_index_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        main(["reindex"])
        capsys.readouterr()

        exit_code = main(["status"])
        captured = capsys.readouterr()

        assert exit_code == 0, f"status falhou: {captured.err}"
        assert "Banco de knowledge" in captured.out
        assert "Chunks" in captured.out

    def test_cli_status_without_index(
        self,
        isolated_paths: Dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from src.knowledge import cli as cli_module
        from src.knowledge import db as db_module

        monkeypatch.setattr(cli_module, "KNOWLEDGE_DB", isolated_paths["db_path"])
        monkeypatch.setattr(db_module, "KNOWLEDGE_DB", isolated_paths["db_path"])

        exit_code = main(["status"])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "não encontrado" in captured.out.lower() or "Index" in captured.out

    def test_cli_query_invalid_layer_is_rejected(
        self,
        mini_index_env: Dict[str, Path],
    ) -> None:
        main(["reindex"])
        with pytest.raises((SystemExit, ValueError)):
            main(["query", "teste", "--layer", "INVALID"])
