"""Testes de compliance LGPD (QG-C11-05)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src.knowledge import constants as kconstants
from src.knowledge.db import get_connection
from src.knowledge.indexer import Chunk, enrich_metadata
from src.knowledge.utils import detect_pii, is_forbidden_path


@pytest.mark.qg_c11
class TestLgpdPatterns:
    """Validação dos padrões de PII e caminhos proibidos."""

    def test_detect_pii_finds_cpf(self) -> None:
        text = "Paciente: João Silva, CPF 123.456.789-00."
        matches = detect_pii(text)
        assert any("123.456.789-00" in m for m in matches)

    def test_detect_pii_finds_email(self) -> None:
        text = "Contato: engenheiro@projectlewis.local"
        matches = detect_pii(text)
        assert any("engenheiro@projectlewis.local" in m for m in matches)

    def test_detect_pii_finds_patient_name_context(self) -> None:
        text = "Patient: Maria Oliveira apresentou arritmia."
        matches = detect_pii(text)
        assert matches

    def test_forbidden_path_blocks_ecg_extensions(self) -> None:
        assert is_forbidden_path(Path("data/raw_mitbih/100.dat"))
        assert is_forbidden_path(Path("data/raw_chapman/record.hea"))
        assert is_forbidden_path(Path("data/raw_ptbxl/sinal.mat"))

    def test_forbidden_path_allows_docs(self) -> None:
        assert not is_forbidden_path(Path("docs/Camada-04-Modelagem-v1.1.md"))
        assert not is_forbidden_path(Path("src/models/stage1_binary.py"))


@pytest.mark.qg_c11
class TestLgpdRejection:
    """Rejeição automática de chunks com PII e zero PII no índice."""

    def test_enrich_metadata_rejects_pii_chunk(
        self,
        isolated_paths: Dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(kconstants, "DLQ_PATH", isolated_paths["dlq_path"])

        chunks = [
            Chunk(
                "O threshold de QG5 é 0.78 de acurácia.",
                "docs/Camada-04-Modelagem-v1.1.md",
            ),
            Chunk(
                "Paciente: Carlos Eduardo, CPF 123.456.789-00.",
                "docs/fake-pii.md",
            ),
        ]
        enriched = enrich_metadata(chunks)

        assert len(enriched) == 1
        assert enriched[0][0].source == "docs/Camada-04-Modelagem-v1.1.md"

        dlq_text = isolated_paths["dlq_path"].read_text(encoding="utf-8").strip()
        assert dlq_text, "DLQ não foi escrita"
        record = json.loads(dlq_text.splitlines()[-1])
        assert record["reason"] == "PII_DETECTED"
        assert record["action"] == "REJECTED"

    def test_zero_pii_in_real_index(self, mini_index_env: Dict[str, Path]) -> None:
        from src.knowledge.indexer import build_index

        build_index()

        conn: Any = get_connection(mini_index_env["db_path"])
        try:
            rows = conn.execute("SELECT content FROM knowledge_chunks").fetchall()
        finally:
            conn.close()

        all_content = "\n".join(row["content"] for row in rows)
        matches = detect_pii(all_content)
        assert not matches, f"PII detectado no índice: {matches[:5]}"
