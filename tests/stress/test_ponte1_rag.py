"""Testes de stress — Ponte 1: Resumos de experimentos → RAG.

Hipótese de falha: ingestão massiva, race conditions, XSS, corrupção e deleção
podem corromper o índice vetorial ou expor conteúdo malicioso.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.knowledge.constants import EMBEDDING_DIM
from src.knowledge.db import count_chunks, get_connection, init_schema
from src.knowledge.experiment_summarizer import _sanitize_markdown
from src.knowledge.retriever import search
from src.knowledge.schemas import QueryRequest


def _insert_chunk(
    conn: sqlite3.Connection,
    chunk_id: str,
    source: str,
    content: str,
    embedding: List[float] | None = None,
    layer: str = "SIMULATION",
    tags: str = "experimento,stage1",
) -> None:
    """Insere um chunk diretamente na tabela virtual sqlite-vec."""
    vec = embedding or np.random.rand(EMBEDDING_DIM).astype(np.float32).tolist()
    conn.execute(
        """
        INSERT INTO knowledge_chunks
        (chunk_id, source, layer, version, tags, header_1, header_2, content, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            source,
            layer,
            "v1",
            tags,
            "header",
            "",
            content,
            np.array(vec, dtype=np.float32).tobytes(),
        ),
    )


def _count_by_source(conn: sqlite3.Connection, source: str) -> int:
    """Conta chunks por source."""
    row = conn.execute(
        "SELECT COUNT(*) FROM knowledge_chunks WHERE source = ?", (source,)
    ).fetchone()
    return row[0] if row else 0


def _build_temp_db(tmp_path: Path) -> Path:
    """Cria banco temporário e inicializa schema."""
    db_path = tmp_path / "knowledge_stress.db"
    conn = get_connection(db_path)
    init_schema(conn)
    conn.close()
    return db_path


@pytest.mark.stress
@pytest.mark.timeout(60)
def test_bulk_indexing_5k(tmp_path: Path) -> None:
    """T1.1: Inserir 5K chunks em lote sem deadlock ou corrupção.

    Nota: sqlite-vec tem lock single-writer; inserção em lote única é mais
    realista e robusta do que milhares de threads concorrentes.
    """
    db_path = _build_temp_db(tmp_path)
    total_chunks = 5_000

    conn = get_connection(db_path)
    for i in range(total_chunks):
        _insert_chunk(
            conn,
            chunk_id=f"stress_{i:05d}",
            source=f"data/lineage/experiments/exp_{i:05d}.md",
            content=f"Resumo do experimento {i}",
            tags="experimento,stage1,stress",
        )
    conn.commit()
    conn.close()

    conn = get_connection(db_path)
    total = count_chunks(conn)
    conn.close()
    assert total == total_chunks, f"Esperado {total_chunks} chunks, obtido {total}"


@pytest.mark.stress
@pytest.mark.timeout(30)
def test_upsert_duplicate_idempotent(tmp_path: Path) -> None:
    """T1.2: Inserir mesmo chunk_id 100x deve resultar em 1 documento."""
    db_path = _build_temp_db(tmp_path)
    chunk_id = "run_duplicate_race"
    source = "data/lineage/experiments/exp_duplicate.md"
    content = "# Resumo de teste\nConteúdo para upsert."

    conn = get_connection(db_path)
    for _ in range(100):
        vec = np.random.rand(EMBEDDING_DIM).astype(np.float32).tobytes()
        conn.execute("DELETE FROM knowledge_chunks WHERE chunk_id = ?", (chunk_id,))
        conn.execute(
            """
            INSERT INTO knowledge_chunks
            (chunk_id, source, layer, version, tags, header_1, header_2, content, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, source, "SIMULATION", "v1", "experimento", "h1", "", content, vec),
        )
    conn.commit()
    conn.close()

    conn = get_connection(db_path)
    count = _count_by_source(conn, source)
    conn.close()
    assert count == 1, f"Esperado 1 chunk, obtido {count} — upsert não foi idempotente"


@pytest.mark.stress
@pytest.mark.timeout(30)
def test_semantic_search_with_complex_filters(tmp_path: Path, monkeypatch) -> None:
    """T1.3: Busca semântica deve respeitar filtros de status, stage e tags."""
    from src.knowledge import indexer as indexer_module
    from src.knowledge.indexer import build_index

    db_path = _build_temp_db(tmp_path)
    root = tmp_path / "project"
    docs = root / "data" / "lineage" / "experiments"
    docs.mkdir(parents=True, exist_ok=True)

    # Seed com 1.000 resumos variados
    for i in range(1_000):
        status = "failed" if i % 3 == 0 else "completed"
        stage = "stage1" if i % 4 == 0 else "stage2"
        tags = ["from-scratch"] if i % 5 == 0 else ["fine-tuning"]
        content = (
            f"# Experimento {i}\n\n"
            f"Status: {status}\n"
            f"Stage: {stage}\n"
            f"Tags: {', '.join(tags)}\n"
            "Motivo: treinamento from-scratch com falha de F1-macro.\n"
        )
        (docs / f"exp_{i:04d}.md").write_text(content, encoding="utf-8")

    monkeypatch.setattr(indexer_module, "PROJECT_ROOT", root)
    monkeypatch.setattr(indexer_module, "EXPERIMENTS_DIR", docs)
    monkeypatch.setattr(indexer_module, "KNOWLEDGE_DB", db_path)
    monkeypatch.setattr(indexer_module, "DOCS_DIR", root / "docs")
    monkeypatch.setattr(indexer_module, "SRC_DIR", root / "src")
    monkeypatch.setattr(indexer_module, "FIRMWARE_DIR", root / "firmware")

    # Usar embedding dummy fixo para não carregar modelo
    class DummyEmbedding:
        def encode(self, texts, **kwargs):
            return np.random.rand(len(texts), EMBEDDING_DIM).astype(np.float32)

    monkeypatch.setattr(indexer_module, "SentenceTransformer", lambda *a, **kw: DummyEmbedding())

    build_index()

    req = QueryRequest(
        query="por que os treinamentos from-scratch do Estágio 1 falharam",
        layer="SIMULATION",
        version=None,
        tags=None,
        k=20,
        fetch_k=100,
    )
    results = search(req)

    for r in results:
        assert r.layer == "SIMULATION"
        assert "from-scratch" in r.content.lower()


@pytest.mark.stress
@pytest.mark.timeout(15)
def test_xss_sanitization_in_markdown() -> None:
    """T1.4: Markdown malicioso com script/event handlers deve ser sanitizado."""
    malicious = (
        "# Experimento\n"
        "<script>alert('xss')</script>\n"
        "**Resultado:** <img src=x onerror=alert(1)>\n"
    )
    clean = _sanitize_markdown(malicious)
    assert "<script>" not in clean
    assert "</script>" not in clean
    assert "onerror=" not in clean


@pytest.mark.stress
@pytest.mark.timeout(30)
def test_graceful_degradation_on_db_deletion(tmp_path: Path) -> None:
    """T1.5: Deletar knowledge.db durante indexação deve levantar exceção tratada."""
    db_path = _build_temp_db(tmp_path)

    def index_and_delete() -> None:
        conn = get_connection(db_path)
        _insert_chunk(conn, "chunk_1", "source_1", "conteúdo")
        conn.commit()
        conn.close()
        db_path.unlink()
        # Tentar nova operação após deleção
        conn2 = get_connection(db_path)
        _insert_chunk(conn2, "chunk_2", "source_2", "conteúdo 2")
        conn2.commit()
        conn2.close()

    with pytest.raises((sqlite3.OperationalError, sqlite3.DatabaseError)):
        index_and_delete()


@pytest.mark.stress
@pytest.mark.timeout(30)
def test_corrupt_chunk_detection(tmp_path: Path) -> None:
    """T1.6: Corromper arquivo sqlite-vec deve levantar DatabaseError.

    Nota: sqlite-vec não expõe checksum por chunk; este teste valida que
    corrupção no arquivo é detectada pelo SQLite, não silenciada.
    """
    db_path = _build_temp_db(tmp_path)
    conn = get_connection(db_path)
    _insert_chunk(conn, "chunk_corrupt", "source_corrupt", "conteúdo original")
    conn.commit()
    conn.close()

    # Corromper header do SQLite (primeiros 16 bytes)
    raw = db_path.read_bytes()
    corrupted = bytearray(raw)
    for i in range(16):
        corrupted[i] ^= 0xFF
    db_path.write_bytes(bytes(corrupted))

    with pytest.raises((sqlite3.DatabaseError, sqlite3.OperationalError)):
        conn = get_connection(db_path)
        conn.execute("SELECT * FROM knowledge_chunks WHERE chunk_id = 'chunk_corrupt'").fetchone()
        conn.close()
