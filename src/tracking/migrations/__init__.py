"""Runner minimalista de migrations SQL para o tracking.

Executa arquivos `.sql` em ordem lexical uma única vez, registrando
a versão aplicada na tabela interna `_migrations`.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import List, Tuple

import sqlparse
from sqlalchemy import Engine, text
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)


_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS _migrations (
    filename TEXT PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _list_migration_files(migrations_dir: Path) -> List[Path]:
    """Lista arquivos `.sql` em ordem lexical."""
    return sorted(migrations_dir.glob("*.sql"))


def _already_applied(engine: Engine, filename: str) -> bool:
    """Verifica se uma migration já foi aplicada."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT 1 FROM _migrations WHERE filename = :filename"),
            {"filename": filename},
        ).fetchone()
        return result is not None


def _record_migration(engine: Engine, filename: str) -> None:
    """Registra migration aplicada."""
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO _migrations (filename) VALUES (:filename)"),
            {"filename": filename},
        )
        conn.commit()


def apply_migrations(engine: Engine, migrations_dir: Path | None = None) -> List[str]:
    """Aplica migrations pendentes e retorna nomes aplicados.

    Parameters
    ----------
    engine : Engine
        Engine SQLAlchemy do banco de tracking.
    migrations_dir : Path | None
        Diretório com arquivos `.sql`. Padrão: `src/tracking/migrations`.

    Returns
    -------
    list[str]
        Nomes das migrations aplicadas nesta execução.
    """
    if migrations_dir is None:
        migrations_dir = Path(__file__).resolve().parent

    applied: List[str] = []

    with engine.connect() as conn:
        conn.execute(text(_MIGRATIONS_TABLE))
        conn.commit()

    for sql_path in _list_migration_files(migrations_dir):
        filename = sql_path.name
        if _already_applied(engine, filename):
            logger.debug("Migration %s já aplicada", filename)
            continue

        sql = sql_path.read_text(encoding="utf-8")
        logger.info("Aplicando migration %s", filename)

        with engine.connect() as conn:
            for statement in sqlparse.split(sql):
                stripped = statement.strip()
                if not stripped:
                    continue
                try:
                    conn.execute(text(stripped))
                except OperationalError as exc:
                    # SQLite não suporta ADD COLUMN IF NOT EXISTS; torna idempotente
                    if isinstance(exc.orig, sqlite3.OperationalError):
                        if "duplicate column name" in str(exc.orig):
                            logger.debug("Coluna já existe, ignorando: %s", stripped)
                            continue
                    raise
            conn.commit()

        _record_migration(engine, filename)
        applied.append(filename)

    return applied


def get_applied_migrations(engine: Engine) -> List[Tuple[str, str]]:
    """Retorna migrations já aplicadas com timestamp."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT filename, applied_at FROM _migrations ORDER BY filename")
        ).fetchall()
        return [(row[0], row[1]) for row in rows]
