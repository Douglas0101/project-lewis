"""Testes do indexador semântico da Camada C11.

Quality Gates cobertos:
- QG-C11-01: cobertura de indexação (apenas 1-2 arquivos temporários).
- QG-C11-02: determinismo / idempotência de reindexação.
- QG-C11-07: tamanho do banco de knowledge (< 500 MB).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from src.knowledge import constants as kconstants
from src.knowledge.db import count_chunks, get_connection
from src.knowledge.indexer import (
    Chunk,
    build_index,
    enrich_metadata,
    load_code_documents,
    load_markdown_documents,
    resolve_layer,
    resolve_version,
    split_text_recursive,
)
from src.knowledge.utils import is_forbidden_path


@pytest.mark.qg_c11
class TestIndexerUnit:
    """Testes unitários leves que não carregam todo o projeto."""

    def test_resolve_layer_maps_known_prefixes(self) -> None:
        assert resolve_layer(Path("docs/Camada-04-Modelagem-v1.1.md")) == "C04"
        assert resolve_layer(Path("docs/SDD_Project-Lewis_v3.md")) == "SDD"
        assert resolve_layer(Path("docs/PRD_foo.md")) == "PRD"

    def test_resolve_layer_fallback_general(self) -> None:
        assert resolve_layer(Path("src/models/stage1_binary.py")) == "GENERAL"
        assert resolve_layer(Path("README.md")) == "GENERAL"

    def test_resolve_version_extracts_semver(self) -> None:
        assert resolve_version(Path("docs/Camada-04-Modelagem-v1.1.md")) == "v1.1"
        assert resolve_version(Path("docs/Camada-09-Energia-v1.4.md")) == "v1.4"

    def test_resolve_version_unversioned(self) -> None:
        assert resolve_version(Path("src/models/foo.py")) == "unversioned"

    def test_split_text_recursive_returns_non_empty_chunks(self) -> None:
        text = " ".join(["palavra"] * 2000)
        chunks = split_text_recursive(text, chunk_size=512, overlap=64)
        assert isinstance(chunks, list)
        assert chunks
        assert all(isinstance(c, str) for c in chunks)

    def test_deterministic_chunk_id_is_stable(self) -> None:
        from src.knowledge.indexer import deterministic_chunk_id

        cid1 = deterministic_chunk_id("docs/a.md", "conteudo identico")
        cid2 = deterministic_chunk_id("docs/a.md", "conteudo identico")
        cid3 = deterministic_chunk_id("docs/b.md", "conteudo identico")
        assert cid1 == cid2
        assert cid1 != cid3
        assert len(cid1) == 16

    def test_is_forbidden_path_blocks_raw_ecg(self) -> None:
        assert is_forbidden_path(Path("data/raw_chapman/record.dat")) is True
        assert is_forbidden_path(Path("data/raw_mitbih/100.hea")) is True
        assert is_forbidden_path(Path("docs/Camada-04-Modelagem-v1.1.md")) is False

    def test_enrich_metadata_rejects_pii_and_extracts_tags(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        dlq = tmp_path / "dlq.jsonl"
        monkeypatch.setattr(kconstants, "DLQ_PATH", dlq)

        clean = Chunk("O modelo CNN usa GroupKFold.", "docs/clean.md")
        dirty = Chunk("CPF do autor: 123.456.789-00.", "docs/dirty.md")

        enriched = enrich_metadata([clean, dirty])
        sources = {meta.source for meta, _ in enriched}

        assert "docs/clean.md" in sources
        assert "docs/dirty.md" not in sources
        assert any("ml" in meta.tags for meta, _ in enriched)
        assert dlq.exists()


@pytest.mark.qg_c11
class TestIndexerCoverage:
    """QG-C11-01: cobertura de indexação com apenas 1-2 arquivos temporários."""

    def test_load_markdown_documents_with_temp_files(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "Camada-04-Modelagem-v1.1.md").write_text(
            "# Modelagem\n\nO QG5 exige F1-macro maior que 30 por cento.",
            encoding="utf-8",
        )
        (docs / "Camada-08-Firmware-v1.1.md").write_text(
            "# Firmware\n\nO STM32 roda TFLM e CMSIS-NN.", encoding="utf-8"
        )

        monkeypatch.setattr("src.knowledge.indexer.DOCS_DIR", docs)
        monkeypatch.setattr("src.knowledge.indexer.PROJECT_ROOT", tmp_path)

        chunks = load_markdown_documents()
        assert chunks
        assert all(Path(c.source).suffix == ".md" for c in chunks)
        assert any("Camada-04" in c.source for c in chunks)
        assert any("Camada-08" in c.source for c in chunks)

    def test_load_code_documents_with_temp_files(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        src = tmp_path / "src"
        firmware = tmp_path / "firmware" / "src"
        (src / "models").mkdir(parents=True)
        firmware.mkdir(parents=True)

        (src / "models" / "stage1_binary.py").write_text(
            "def build_model(): return None\n", encoding="utf-8"
        )
        (firmware / "main.c").write_text(
            "int main(void) { while(1); }\n", encoding="utf-8"
        )

        monkeypatch.setattr("src.knowledge.indexer.SRC_DIR", src)
        monkeypatch.setattr("src.knowledge.indexer.FIRMWARE_DIR", firmware)
        monkeypatch.setattr("src.knowledge.indexer.PROJECT_ROOT", tmp_path)

        chunks = load_code_documents()
        assert chunks
        assert all(Path(c.source).suffix in {".py", ".c", ".h", ".cpp"} for c in chunks)

    def test_forbidden_extensions_not_indexed(self) -> None:
        assert ".dat" in kconstants.FORBIDDEN_EXTENSIONS
        assert ".mat" in kconstants.FORBIDDEN_EXTENSIONS
        assert "raw_chapman/" in kconstants.FORBIDDEN_PATH_PATTERNS


@pytest.mark.qg_c11
class TestIndexerDeterminism:
    """QG-C11-02: reindexação gera IDs idênticos."""

    def test_build_index_is_idempotent(self, mini_index_env: Dict[str, Path]) -> None:
        build_index()
        first_ids = _fetch_chunk_ids(mini_index_env["db_path"])
        first_count = count_chunks()
        assert first_count > 0

        build_index()
        second_ids = _fetch_chunk_ids(mini_index_env["db_path"])
        second_count = count_chunks()

        assert first_count == second_count
        assert first_ids == second_ids, "IDs de chunks divergiram entre reindexações"


@pytest.mark.qg_c11
class TestIndexerSize:
    """QG-C11-07: tamanho do banco SQLite deve permanecer abaixo de 500 MB."""

    def test_database_size_under_500mb(self, mini_index_env: Dict[str, Path]) -> None:
        lineage = build_index()
        mb = lineage.db_size_bytes / (1024 * 1024)
        assert mb < 500, f"Banco knowledge.db excede 500 MB: {mb:.2f} MB"


@pytest.mark.qg_c11
class TestIndexerLineage:
    """Validações adicionais da linhagem de indexação."""

    def test_lineage_persists_after_build(self, mini_index_env: Dict[str, Path]) -> None:
        lineage = build_index()
        assert lineage.total_chunks > 0
        assert lineage.total_files > 0
        assert lineage.layers_found
        assert lineage.index_duration_sec >= 0
        assert mini_index_env["lineage_dir"].exists()


def _fetch_chunk_ids(db_path: Path) -> list[str]:
    conn: Any = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT chunk_id FROM knowledge_chunks ORDER BY chunk_id"
        ).fetchall()
        return [row["chunk_id"] for row in rows]
    finally:
        conn.close()
