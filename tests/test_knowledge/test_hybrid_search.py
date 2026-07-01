from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pytest

from src.knowledge import cli as cli_module
from src.knowledge import indexer as indexer_module
from src.knowledge import retriever as retriever_module
from src.knowledge.db import get_connection, init_schema
from src.knowledge.hybrid_search import HybridSearcher
from src.knowledge.retriever import hybrid_search
from src.knowledge.schemas import QueryRequest, QueryResult

FIXED_EMBEDDING = np.ones(384, dtype=np.float32) * 0.1


def _insert_chunk(conn, chunk_id, source, content, tags="experimento"):
    conn.execute(
        """
        INSERT INTO knowledge_chunks
            (chunk_id, source, layer, version, tags, header_1, header_2, content, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            source,
            "SIMULATION",
            "v1",
            tags,
            "h1",
            "",
            content,
            np.array(FIXED_EMBEDDING, dtype=np.float32).tobytes(),
        ),
    )


def test_hybrid_search_rrf(tmp_path):
    db_path = tmp_path / "hybrid.db"
    conn = get_connection(db_path)
    init_schema(conn)

    _insert_chunk(conn, "c1", "run_001.md", "accuracy do run_001 foi 0.92")
    _insert_chunk(conn, "c2", "run_002.md", "resultado do run_002")
    conn.commit()
    conn.close()

    class FakeIndexer:
        def semantic_search(self, query, filters=None, top_k=10):
            return [{"id": "c1", "score": 0.9, "content": "accuracy do run_001 foi 0.92"}]

        def get_doc(self, doc_id):
            return {"id": doc_id, "content": f"doc {doc_id}"}

    searcher = HybridSearcher(str(db_path), FakeIndexer())
    searcher.index_document(doc_id="c1", content="accuracy do run_001 foi 0.92", run_id="c1")
    searcher.index_document(doc_id="c2", content="resultado do run_002", run_id="c2")
    results = searcher.search("run_001 accuracy", top_k=5)
    assert any(r["id"] == "c1" for r in results)


def test_hybrid_search_bm25_boosts_rrf(tmp_path):
    db_path = tmp_path / "hybrid.db"
    conn = get_connection(db_path)
    init_schema(conn)

    _insert_chunk(conn, "c1", "run_001.md", "accuracy do run_001 foi 0.92")
    _insert_chunk(conn, "c2", "run_002.md", "resultado do run_002")
    conn.commit()
    conn.close()

    class FakeIndexer:
        def semantic_search(self, query, filters=None, top_k=10):
            # Vector ranking alone would pick c2.
            return [
                {"id": "c2", "score": 0.95, "content": "resultado do run_002"},
                {"id": "c1", "score": 0.5, "content": "accuracy do run_001 foi 0.92"},
            ]

        def get_doc(self, doc_id):
            content_map = {
                "c1": "accuracy do run_001 foi 0.92",
                "c2": "resultado do run_002",
            }
            return {"id": doc_id, "content": content_map.get(doc_id, "")}

    searcher = HybridSearcher(str(db_path), FakeIndexer())
    searcher.index_document(doc_id="c1", content="accuracy do run_001 foi 0.92", run_id="c1")
    searcher.index_document(doc_id="c2", content="resultado do run_002", run_id="c2")

    results = searcher.search("run_001 accuracy", top_k=5)
    ids = [r["id"] for r in results]
    assert ids[0] == "c1"
    assert "c2" in ids


def _populate_hybrid_db(
    db_path: Path,
    test_embedding: Any,
    monkeypatch: pytest.MonkeyPatch,
    chunks: List[Tuple[str, str, str, str, str, List[str]]],
) -> None:
    """Popula banco temporário com chunks vetoriais e FTS5 para busca híbrida."""
    conn = get_connection(db_path)
    init_schema(conn)

    for chunk_id, source, layer, version, content, tags in chunks:
        emb = test_embedding.encode(f"{source}\n{content}")[0]
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
                version,
                json.dumps(tags),
                "",
                "",
                content,
                np.array(emb, dtype=np.float32).tobytes(),
            ),
        )

    conn.commit()
    conn.close()

    class _FakeAdapter:
        pass

    hybrid = HybridSearcher(str(db_path), _FakeAdapter())
    for chunk_id, source, layer, version, content, tags in chunks:
        hybrid.index_document(
            doc_id=chunk_id,
            content=content,
            run_id=source,
            stage=layer,
            status=",".join(tags),
        )

    monkeypatch.setattr(retriever_module, "KNOWLEDGE_DB", db_path)
    monkeypatch.setattr(retriever_module, "_get_model", lambda: test_embedding)


def test_hybrid_search_returns_query_results(
    tmp_path: Path,
    test_embedding: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A busca híbrida exposta no retriever deve retornar objetos QueryResult."""
    db_path = tmp_path / "hybrid_retriever.db"
    chunks = [
        (
            "c1",
            "docs/Camada-04-Modelagem-v1.1.md",
            "C04",
            "v1.1",
            "O QG5 exige F1-macro maior que 30 por cento.",
            ["ml"],
        ),
        (
            "c2",
            "docs/Camada-08-Firmware-v1.1.md",
            "C08",
            "v1.1",
            "O STM32 roda TFLM e CMSIS-NN no Cortex-M4.",
            ["firmware"],
        ),
    ]
    _populate_hybrid_db(db_path, test_embedding, monkeypatch, chunks)

    req = QueryRequest(query="QG5 threshold F1 macro", k=3)
    results = hybrid_search(req)

    assert results
    assert all(isinstance(r, QueryResult) for r in results)
    assert all(r.rank >= 1 for r in results)
    assert results[0].rank == 1


def test_cli_query_hybrid_flag_runs(
    mini_index_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O comando 'query --hybrid' deve executar sem erro com um índice populado."""
    monkeypatch.setattr(retriever_module, "KNOWLEDGE_DB", mini_index_env["db_path"])
    monkeypatch.setattr(cli_module, "KNOWLEDGE_DB", mini_index_env["db_path"])

    indexer_module.build_index()

    exit_code = cli_module.main(["query", "--hybrid", "STM32", "--k", "3"])
    assert exit_code == 0
