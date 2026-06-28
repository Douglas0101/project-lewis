"""Registro de artefatos no banco de tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy.engine import Engine

from src.memory.checksums import sha256_file
from src.tracking.db import session_scope
from src.tracking.repositories import ArtifactRepository
from src.tracking.schemas import ArtifactCreate


def record_artifact(
    run_id: int,
    path: Path,
    artifact_type: str,
    name: Optional[str] = None,
    dvc_status: str = "untracked",
    engine: Optional[Engine] = None,
) -> int:
    """Registra um artefato no banco, evitando duplicatas por checksum.

    Parameters
    ----------
    run_id : int
        ID da run vinculada ao artefato.
    path : Path
        Caminho do arquivo no disco.
    artifact_type : str
        Tipo do artefato (ex.: "model", "config", "report").
    name : str, optional
        Nome descritivo; usa ``path.name`` quando omitido.
    dvc_status : str, optional
        Status de versionamento DVC (padrão: "untracked").
    engine : Engine, optional
        Engine SQLAlchemy para uso em testes; usa o padrão do projeto quando
        omitido.

    Returns
    -------
    int
        ID do artefato registrado ou ID do existente com mesmo checksum.

    Raises
    ------
    FileNotFoundError
        Se ``path`` não existir no disco.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    checksum = sha256_file(path)
    try:
        stored_path = str(path.relative_to(Path.cwd()))
    except ValueError:
        stored_path = str(path)
    with session_scope(engine=engine) as session:
        repo = ArtifactRepository(session)
        existing = repo.get_by_checksum(checksum)
        if existing:
            return existing.id
        art = repo.create(
            ArtifactCreate(
                run_id=run_id,
                name=name or path.name,
                path=stored_path,
                checksum=checksum,
                artifact_type=artifact_type,
                dvc_status=dvc_status,
            )
        )
        return art.id
