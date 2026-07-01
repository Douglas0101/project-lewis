"""Hybrid search: BM25 via FTS5 + vector cosine via sqlite-vec, fused by RRF."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


class VectorIndexer(Protocol):
    """Contrato mínimo para um indexador vetorial usado pelo HybridSearcher."""

    def semantic_search(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        ...

    def get_doc(self, doc_id: str) -> Dict[str, Any]:
        ...


class HybridSearcher:
    """Combina busca vetorial e BM25 com Reciprocal Rank Fusion."""

    def __init__(self, db_path: str | Path, vector_indexer: VectorIndexer, k: int = 60):
        self.db_path = str(db_path)
        self.vector_indexer = vector_indexer
        self.k = k
        self._ensure_fts5()

    def _ensure_fts5(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                    content, run_id UNINDEXED, stage UNINDEXED, status UNINDEXED,
                    tokenize="porter"
                )
                """
            )
        finally:
            conn.close()

    def index_document(
        self, doc_id: str, content: str, run_id: str = "", stage: str = "", status: str = ""
    ) -> None:
        """Indexa um documento na tabela FTS5.

        O ``doc_id`` é armazenado na coluna ``run_id`` para fusão RRF com
        resultados vetoriais, que normalmente representam execuções de
        experimentos. As colunas ``run_id``, ``stage`` e ``status`` são
        UNINDEXED, reduzindo o tamanho do índice FTS5.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO knowledge_fts (content, run_id, stage, status) VALUES (?, ?, ?, ?)",
                (content, doc_id, stage, status),
            )
            conn.commit()
        finally:
            conn.close()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Executa busca híbrida e retorna documentos ordenados por RRF."""
        vector_results = self.vector_indexer.semantic_search(query, top_k=top_k * 2)
        scores: Dict[str, float] = {}
        for rank, doc in enumerate(vector_results):
            doc_id = doc.get("id") or doc.get("chunk_id")
            if doc_id:
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.k + rank + 1)

        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT run_id, rank
                FROM knowledge_fts
                WHERE knowledge_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, top_k * 2),
            ).fetchall()
        finally:
            conn.close()

        for rank, (run_id, _) in enumerate(rows):
            if run_id:
                scores[run_id] = scores.get(run_id, 0.0) + 1.0 / (self.k + rank + 1)

        sorted_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)[:top_k]
        return [self.vector_indexer.get_doc(doc_id) for doc_id in sorted_ids]
