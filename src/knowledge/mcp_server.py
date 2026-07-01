"""MCP Server stdio para integração Kimi Code / OpenCode.

Usa SDK oficial do Python (FastMCP).
Tools expostas:
  - search_docs(query, layer?, version?, tags?, k?)
  - list_layers()
  - get_doc_by_source(source, k?)
  - query_training_logs(question)
  - execute_validated_sql(sql, params?)

Autor: Douglas Souza
Data: 2026-06-27
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .retriever import search
from .schemas import QueryRequest
from .structured_query import NaturalQueryRequest, answer_question, execute_custom_sql

mcp = FastMCP("project-lewis-knowledge")


@mcp.tool()
def search_docs(
    query: str,
    layer: Optional[str] = None,
    version: Optional[str] = None,
    tags: Optional[List[str]] = None,
    k: int = 5,
) -> str:
    """Busca semântica na documentação do Project-Lewis."""
    req = QueryRequest(
        query=query,
        layer=layer,
        version=version,
        tags=tags,
        k=k,
        fetch_k=20,
    )
    results = search(req)
    blocks = []
    for r in results:
        blocks.append(
            f"[{r.rank}] {r.source} (Camada {r.layer}, {r.version}, score={r.score:.4f})\n"
            f"Tags: {', '.join(r.tags)}\n{r.content}"
        )
    return "\n---\n".join(blocks)


@mcp.tool()
def search_docs_hybrid(
    query: str,
    layer: Optional[str] = None,
    version: Optional[str] = None,
    tags: Optional[List[str]] = None,
    k: int = 5,
) -> str:
    """Busca híbrida (BM25 + vetorial) na documentação do Project-Lewis."""
    from .retriever import hybrid_search

    req = QueryRequest(query=query, layer=layer, version=version, tags=tags, k=k, fetch_k=k * 4)
    results = hybrid_search(req)
    blocks = []
    for r in results:
        blocks.append(
            f"[{r.rank}] {r.source} (Camada {r.layer}, {r.version})\n"
            f"Tags: {', '.join(r.tags)}\n{r.content}"
        )
    return "\n---\n".join(blocks)


@mcp.tool()
def list_layers() -> List[str]:
    """Lista camadas arquiteturais disponíveis."""
    return [
        "C01",
        "C02",
        "C03",
        "C04",
        "C05",
        "C06",
        "C07",
        "C08",
        "C09",
        "C10",
        "SDD",
        "PRD",
        "UNIFIED",
        "SIMULATION",
    ]


@mcp.tool()
def get_doc_by_source(source: str, k: int = 3) -> str:
    """Recupera chunks por caminho de arquivo."""
    req = QueryRequest(
        query=source,
        layer=None,
        version=None,
        tags=None,
        k=k,
        fetch_k=20,
    )
    results = [r for r in search(req) if r.source == source]
    return "\n---\n".join(f"[{r.rank}] {r.source}\n{r.content}" for r in results)


@mcp.tool()
def query_training_logs(question: str) -> str:
    """Consulta o banco estruturado de métricas via linguagem natural.

    Exemplos:
      - "últimas runs que falharam"
      - "alertas críticos"
      - "runs do estágio stage1"
      - "linha do tempo"
      - "métricas da run 147"
    """
    req = NaturalQueryRequest(question=question)
    result = answer_question(req)
    return json.dumps(
        {
            "sql": result.sql,
            "params": result.params,
            "columns": result.columns,
            "rows": result.rows,
            "row_count": result.row_count,
            "truncated": result.truncated,
            "source": result.source,
            "audit_id": result.audit_id,
        },
        ensure_ascii=False,
        default=str,
        indent=2,
    )


@mcp.tool()
def execute_validated_sql(sql: str, params: Optional[str] = None) -> str:
    """Executa SQL read-only no banco de métricas após validação de segurança.

    Parâmetros:
      - sql: query SELECT validada.
      - params: JSON com parâmetros (opcional).
    """
    parsed_params: Dict[str, Any] = {}
    if params:
        parsed_params = json.loads(params)
    result = execute_custom_sql(sql, parsed_params)
    return json.dumps(
        {
            "sql": result.sql,
            "params": result.params,
            "columns": result.columns,
            "rows": result.rows,
            "row_count": result.row_count,
            "truncated": result.truncated,
            "source": result.source,
            "audit_id": result.audit_id,
        },
        ensure_ascii=False,
        default=str,
        indent=2,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
