"""Banco de dados SQLite + sqlite-vec para a Camada C11.

Autor: Douglas Souza
Data: 2026-06-27
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from .constants import EMBEDDING_DIM, KNOWLEDGE_DB


_SQL_CREATE_VEC_TABLE = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks USING vec0(
    chunk_id TEXT PRIMARY KEY,
    source TEXT,
    layer TEXT,
    version TEXT,
    tags TEXT,
    header_1 TEXT,
    header_2 TEXT,
    content TEXT,
    embedding float[{EMBEDDING_DIM}] distance_metric=cosine
);
"""

_SQL_CREATE_META_TABLE = """
CREATE TABLE IF NOT EXISTS knowledge_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Retorna conexão SQLite com extensão sqlite-vec carregada."""
    path = db_path or KNOWLEDGE_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn


def get_engine(db_path: Path | None = None) -> Engine:
    """Retorna engine SQLAlchemy apontando para o banco de knowledge."""
    path = db_path or KNOWLEDGE_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=False, future=True)


def init_schema(conn: sqlite3.Connection | None = None) -> None:
    """Cria tabela virtual sqlite-vec e tabela de metadados."""
    own_conn = conn is None
    target_conn = get_connection() if own_conn else conn
    assert target_conn is not None
    try:
        target_conn.execute(_SQL_CREATE_VEC_TABLE)
        target_conn.execute(_SQL_CREATE_META_TABLE)
        target_conn.commit()
    finally:
        if own_conn:
            target_conn.close()


def count_chunks(conn: sqlite3.Connection | None = None) -> int:
    """Retorna número de chunks indexados."""
    own_conn = conn is None
    target_conn = get_connection() if own_conn else conn
    assert target_conn is not None
    try:
        row = target_conn.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()
        return row[0] if row else 0
    finally:
        if own_conn:
            target_conn.close()


def clear_index(conn: sqlite3.Connection | None = None) -> None:
    """Remove todos os chunks do índice."""
    own_conn = conn is None
    target_conn = get_connection() if own_conn else conn
    assert target_conn is not None
    try:
        target_conn.execute("DELETE FROM knowledge_chunks")
        target_conn.commit()
    finally:
        if own_conn:
            target_conn.close()
