"""Configuração de banco de dados SQLite para tracking.

O banco padrão é ``data/lewis_metrics.db`` relativo à raiz do projeto.
A variável de ambiente ``LEWIS_TRACKING_DB`` pode sobrescrever o caminho.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.tracking.models import Base


def _project_root() -> Path:
    """Retorna a raiz do projeto a partir deste arquivo."""
    return Path(__file__).resolve().parents[2]


def get_db_path() -> Path:
    """Retorna o caminho absoluto do banco SQLite."""
    env_path = os.environ.get("LEWIS_TRACKING_DB")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return _project_root() / "data" / "lewis_metrics.db"


def get_engine(db_path: Path | None = None):
    """Cria engine SQLAlchemy apontando para o banco SQLite."""
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=False, future=True)


def get_sessionmaker(engine=None):
    """Retorna fábrica de sessões vinculada ao engine."""
    return sessionmaker(bind=engine or get_engine(), expire_on_commit=False)


def get_session(engine=None) -> Session:
    """Retorna uma sessão nova (não gerenciada por contexto)."""
    return get_sessionmaker(engine)()


def session_scope(engine=None) -> Generator[Session, None, None]:
    """Context manager para sessões SQLAlchemy com rollback automático."""
    session = get_session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_schema(engine=None) -> None:
    """Cria todas as tabelas se não existirem."""
    target = engine or get_engine()
    Base.metadata.create_all(target)
