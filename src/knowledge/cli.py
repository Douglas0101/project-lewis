"""CLI interno da Camada C11.

Comandos:
  reindex    — Rebuild completo do índice sqlite-vec
  query      — Query via terminal
  status     — Status do índice (chunks, camadas, tamanho)

Autor: Douglas Souza
Data: 2026-06-27
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .constants import KNOWLEDGE_DB, LINEAGE_DIR
from .db import count_chunks
from .indexer import build_index
from .retriever import search
from .schemas import QueryRequest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.knowledge.cli",
        description="CLI da Camada C11 — Knowledge Layer",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("reindex", help="Rebuild completo do índice semântico")

    query_parser = subparsers.add_parser("query", help="Executa busca semântica")
    query_parser.add_argument("q", help="Pergunta ou keyword")
    query_parser.add_argument("--layer", "-l", default=None, help="Filtro camada")
    query_parser.add_argument("--version", "-v", default=None, help="Filtro versão")
    query_parser.add_argument("--k", "-k", type=int, default=5, help="Número de resultados")

    subparsers.add_parser("status", help="Exibe status do índice")

    return parser


def _cmd_reindex() -> int:
    print("Iniciando reindexação...")
    lineage = build_index()
    print(f"✅ {lineage.total_chunks} chunks indexados em {lineage.index_duration_sec:.2f}s")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    req = QueryRequest(
        query=args.q, layer=args.layer, version=args.version, tags=None, k=args.k, fetch_k=20
    )
    results = search(req)
    for r in results:
        print(f"\n[{r.rank}] {r.source} | {r.layer} | {r.version}")
        print(f"Tags: {', '.join(r.tags)}")
        print(f"{r.content[:300]}...")
    return 0


def _cmd_status() -> int:
    if not KNOWLEDGE_DB.exists():
        print("❌ Índice não encontrado. Execute: uv run python -m src.knowledge.cli reindex")
        return 1

    size_mb = KNOWLEDGE_DB.stat().st_size / 1024 / 1024
    lineage_files = sorted(LINEAGE_DIR.glob("index_*.json")) if LINEAGE_DIR.exists() else []

    print(f"📦 Banco de knowledge: {KNOWLEDGE_DB}")
    print(f"📊 Tamanho: {size_mb:.2f} MB")
    print(f"📄 Chunks: {count_chunks()}")
    print(f"📄 Registros de linhagem: {len(lineage_files)}")
    if lineage_files:
        latest = lineage_files[-1]
        print(f"🕐 Última indexação: {latest.name}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "reindex":
        return _cmd_reindex()
    if args.command == "query":
        return _cmd_query(args)
    if args.command == "status":
        return _cmd_status()
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
