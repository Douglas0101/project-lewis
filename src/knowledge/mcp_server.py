"""MCP Server stdio para integração Kimi Code / OpenCode.

Usa SDK oficial do Python (FastMCP).
Tools expostas:
  - search_docs(query, layer?, version?, tags?, k?)
  - list_layers()
  - get_doc_by_source(source, k?)

Autor: Douglas Souza
Data: 2026-06-27
"""

from __future__ import annotations

from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from .retriever import search
from .schemas import QueryRequest

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
def list_layers() -> List[str]:
    """Lista camadas arquiteturais disponíveis."""
    return [
        "C01", "C02", "C03", "C04", "C05", "C06",
        "C07", "C08", "C09", "C10", "SDD", "PRD", "UNIFIED",
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
    return "\n---\n".join(
        f"[{r.rank}] {r.source}\n{r.content}" for r in results
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
