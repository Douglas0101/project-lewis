"""Retriever inteligente da Camada C11.

Responsabilidade: busca semântica com filtros 3D,
formatação para agentes e logging de audit trail.

Autor: Douglas Souza
Data: 2026-06-27
"""

from __future__ import annotations

import json
import time
from typing import List

import sqlite_vec
from sentence_transformers import SentenceTransformer

from .constants import EMBEDDING_MODEL, LOG_QUERIES
from .db import get_connection
from .schemas import QueryRequest, QueryResult
from .utils import format_context_for_agent


def _get_model() -> SentenceTransformer:
    """Factory do modelo de embeddings."""
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
        results.append(QueryResult(
            chunk_id=row["chunk_id"],
            source=row["source"],
            layer=row["layer"],
            version=row["version"],
            tags=tags,
            content=row["content"],
            score=1.0 - float(row["distance"]),
            rank=len(results) + 1,
        ))

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


def get_context_for_agent(req: QueryRequest) -> str:
    """Busca e formata contexto para injeção em prompt de agente."""
    results = search(req)
    docs = [r.model_dump() for r in results]
    return format_context_for_agent(docs)
