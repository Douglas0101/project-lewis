"""Retriever inteligente da Camada C11.

Responsabilidade: busca semântica com filtros 3D,
formatação para agentes e logging de audit trail.

Autor: Douglas Souza
Data: 2026-06-27
"""

from __future__ import annotations

import functools
import json
import time
from typing import Any, Dict, List, Optional

import sqlite_vec
from sentence_transformers import SentenceTransformer

from src.observability.metrics import LatencyTracker

from .constants import EMBEDDING_MODEL, KNOWLEDGE_DB, LOG_QUERIES
from .db import get_connection
from .schemas import QueryRequest, QueryResult
from .utils import format_context_for_agent


@functools.lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Factory do modelo de embeddings (singleton por processo)."""
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def _build_where_clause(req: QueryRequest) -> tuple[str, list]:
    """Constrói cláusula WHERE e parâmetros para filtros metadata SQL.

    Nota: sqlite-vec não permite operador ``LIKE`` em colunas de metadata
    dentro de uma query KNN. Filtros por tags são aplicados em Python após
    o KNN, mantendo a compatibilidade com a extensão.
    """
    conditions: List[str] = []
    params: List[str] = []
    if req.layer:
        conditions.append("layer = ?")
        params.append(req.layer)
    if req.version:
        conditions.append("version = ?")
        params.append(req.version)
    where = " AND ".join(conditions)
    return where, params


def search(req: QueryRequest) -> List[QueryResult]:
    """Executa busca semântica com filtros e retorna top-k."""
    with LatencyTracker("rag", "semantic_search"):
        model = _get_model()
        query_emb = model.encode([req.query], normalize_embeddings=True, show_progress_bar=False)[0]

        conn = get_connection()
        try:
            where_clause, where_params = _build_where_clause(req)

            sql = """
                SELECT
                    chunk_id,
                    source,
                    layer,
                    version,
                    tags,
                    header_1,
                    header_2,
                    content,
                    distance
                FROM knowledge_chunks
                WHERE embedding MATCH ?
            """
            params: List = [sqlite_vec.serialize_float32(query_emb)]
            if where_clause:
                sql += f" AND {where_clause}"
                params.extend(where_params)
            sql += " AND k = ? ORDER BY distance"
            params.append(req.fetch_k)

            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    required_tags = set(req.tags or [])
    results: List[QueryResult] = []
    for row in rows:
        if len(results) >= req.k:
            break
        tags = json.loads(row["tags"] or "[]")
        if required_tags and not required_tags.issubset(set(tags)):
            continue
        results.append(
            QueryResult(
                chunk_id=row["chunk_id"],
                source=row["source"],
                layer=row["layer"],
                version=row["version"],
                tags=tags,
                content=row["content"],
                score=1.0 - float(row["distance"]),
                rank=len(results) + 1,
            )
        )

    _log_query(req, len(results))
    return results


def _log_query(req: QueryRequest, result_count: int) -> None:
    """Registra query em audit trail JSONL."""
    LOG_QUERIES.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": req.query,
        "layer": req.layer,
        "version": req.version,
        "tags": req.tags,
        "k": req.k,
        "results_returned": result_count,
    }
    with open(LOG_QUERIES, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class _VectorIndexerAdapter:
    """Adapta o retriever vetorial existente para o HybridSearcher."""

    def __init__(self) -> None:
        self.model = _get_model()

    def semantic_search(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        query_emb = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[
            0
        ]
        conn = get_connection()
        try:
            rows = conn.execute(
                """
                SELECT chunk_id, source, layer, version, tags,
                       header_1, header_2, content, distance
                FROM knowledge_chunks
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                (sqlite_vec.serialize_float32(query_emb), top_k),
            ).fetchall()
            return [{"id": row["chunk_id"], **dict(row)} for row in rows]
        finally:
            conn.close()

    def get_doc(self, doc_id: str) -> Dict[str, Any]:
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM knowledge_chunks WHERE chunk_id = ?", (doc_id,)
            ).fetchone()
            return dict(row) if row else {"id": doc_id}
        finally:
            conn.close()


def hybrid_search(req: QueryRequest) -> List[QueryResult]:
    """Busca híbrida BM25 + vector com filtros 3D em Python."""
    from .hybrid_search import HybridSearcher

    adapter = _VectorIndexerAdapter()
    searcher = HybridSearcher(KNOWLEDGE_DB, adapter)
    rows = searcher.search(req.query, top_k=req.k * 4)
    results: List[QueryResult] = []
    required_tags = set(req.tags or [])
    for row in rows:
        doc = adapter.get_doc(row["id"])
        if req.layer and doc.get("layer") != req.layer:
            continue
        if req.version and doc.get("version") != req.version:
            continue
        raw_tags = doc.get("tags")
        tags: List[str] = []
        if isinstance(raw_tags, str):
            try:
                tags = json.loads(raw_tags)
            except json.JSONDecodeError:
                tags = []
        elif isinstance(raw_tags, list):
            tags = raw_tags
        if required_tags and not required_tags.issubset(set(tags)):
            continue
        if len(results) >= req.k:
            break
        results.append(
            QueryResult(
                chunk_id=doc.get("chunk_id") or doc.get("id") or "",
                source=doc.get("source", ""),
                layer=doc.get("layer", "GENERAL"),
                version=doc.get("version", "unversioned"),
                tags=tags,
                content=doc.get("content", ""),
                score=float(row.get("rrf_score", 1.0)),
                rank=len(results) + 1,
            )
        )
    return results


def get_context_for_agent(req: QueryRequest) -> str:
    """Busca e formata contexto para injeção em prompt de agente."""
    results = search(req)
    docs = [r.model_dump() for r in results]
    return format_context_for_agent(docs)
