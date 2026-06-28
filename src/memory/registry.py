"""ArtifactRegistry para registro de artefatos de runs.

Registra artefatos gerados por treinamentos/avaliações em banco SQLite,
com checksum SHA-256 e referência à run de origem.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from sqlalchemy import (
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
)


class Base(DeclarativeBase):
    """Base declarativa do SQLAlchemy 2.0."""


class Artifact(Base):
    """Artefato vinculado a uma run."""

    __tablename__ = "memory_artifact"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    extra: Mapped[str | None] = mapped_column(Text, nullable=True)


def _project_root() -> Path:
    """Retorna a raiz do projeto a partir deste arquivo."""
    return Path(__file__).resolve().parents[2]


def _get_db_path() -> Path:
    """Retorna o caminho padrão do banco SQLite."""
    env_path = os.environ.get("LEWIS_TRACKING_DB")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return _project_root() / "data" / "lewis_metrics.db"


def _get_engine(db_path: Path | None = None):
    """Cria engine SQLAlchemy apontando para o banco SQLite."""
    path = db_path or _get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=False, future=True)


@contextmanager
def _session_scope(engine=None) -> Generator:
    """Context manager para sessões SQLAlchemy com rollback automático."""
    Session = sessionmaker(bind=engine or _get_engine(), expire_on_commit=False)
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _sha256_file(path: Path) -> str:
    """Calcula SHA-256 de um arquivo em blocos."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def record_artifact(run_id: int, path: Path, artifact_type: str) -> int:
    """Registra um artefato no ArtifactRegistry.

    Args:
        run_id: Identificador da run de origem.
        path: Caminho do artefato no filesystem.
        artifact_type: Tipo do artefato (e.g. "model", "config", "report").

    Returns:
        Identificador do artefato registrado.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Artefato nao encontrado: {path}")

    checksum = _sha256_file(path)
    size_bytes = path.stat().st_size

    # Garante que a tabela artifact exista no mesmo banco do tracking.
    engine = _get_engine()
    Base.metadata.create_all(engine)

    with _session_scope(engine) as session:
        artifact = Artifact(
            run_id=run_id,
            artifact_type=artifact_type,
            path=str(path.resolve()),
            checksum=checksum,
            size_bytes=size_bytes,
        )
        session.add(artifact)
        session.flush()
        return int(artifact.id)
