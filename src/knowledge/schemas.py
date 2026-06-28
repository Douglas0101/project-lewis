"""Schemas Pydantic v2 para a Camada C11.

Autor: Douglas Souza
Data: 2026-06-27
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class _BaseConfig(BaseModel):
    """Base com configuração comum."""

    model_config = {"from_attributes": True}


class DocumentMeta(_BaseConfig):
    """Metadados 3D de um chunk indexado."""

    source: str = Field(..., description="Caminho relativo do arquivo fonte")
    layer: str = Field(..., description="D1: Camada arquitetural (C01-C10, SDD, PRD, etc.)")
    version: str = Field(..., description="D2: Versão do documento (v1.1, v2.0, unversioned)")
    tags: List[str] = Field(default_factory=list, description="D3: Tags semânticas extraídas")
    filename: str = Field(..., description="Nome do arquivo")
    chunk_id: str = Field(..., description="Hash determinístico do chunk")
    header_1: Optional[str] = Field(None, description="H1 do markdown, se presente")
    header_2: Optional[str] = Field(None, description="H2 do markdown, se presente")

    @field_validator("layer")
    @classmethod
    def validate_layer(cls, v: str) -> str:
        allowed = {
            "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10",
            "SDD", "PRD", "UNIFIED", "ESPECIFICACAO", "SIMULATION", "DEBITO_TECNICO", "GENERAL",
        }
        if v not in allowed:
            raise ValueError(f"Layer '{v}' não é válida. Valores permitidos: {allowed}")
        return v


class QueryRequest(_BaseConfig):
    """Request de query para o retriever."""

    query: str = Field(..., min_length=1, max_length=4096)
    layer: Optional[str] = Field(None, description="Filtro D1")
    version: Optional[str] = Field(None, description="Filtro D2")
    tags: Optional[List[str]] = Field(None, description="Filtro D3 (AND lógico)")
    k: int = Field(5, ge=1, le=20)
    fetch_k: int = Field(20, ge=5, le=100)

    @field_validator("layer")
    @classmethod
    def validate_layer_optional(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {
            "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10",
            "SDD", "PRD", "UNIFIED", "ESPECIFICACAO", "SIMULATION", "DEBITO_TECNICO", "GENERAL",
        }
        if v not in allowed:
            raise ValueError(f"Layer '{v}' não é válida.")
        return v


class QueryResult(_BaseConfig):
    """Resultado de uma query."""

    chunk_id: str
    source: str
    layer: str
    version: str
    tags: List[str]
    content: str
    score: float = Field(..., description="Score de similaridade (cosine)")
    rank: int = Field(..., ge=1)


class IndexLineage(_BaseConfig):
    """Registro de linhagem de uma indexação."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_files: int
    total_chunks: int
    layers_found: List[str]
    versions_found: List[str]
    index_duration_sec: float
    db_size_bytes: int
    config_hash: str
