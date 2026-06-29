"""Camada C11 — Knowledge Layer (RAG + sqlite-vec + MCP).

Exporta os componentes fundamentais da camada de conhecimento do Project-Lewis.
"""

from src.knowledge.constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    CONFIG_PATH,
    DLQ_PATH,
    DOCS_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FIRMWARE_DIR,
    FORBIDDEN_EXTENSIONS,
    FORBIDDEN_PATH_PATTERNS,
    KNOWLEDGE_DB,
    LINEAGE_DIR,
    LOG_QUERIES,
    MAX_CHUNK_CHARS,
    PII_PATTERNS,
    PROJECT_ROOT,
    SRC_DIR,
    TAG_KEYWORDS,
)
from src.knowledge.schemas import DocumentMeta, IndexLineage, QueryRequest, QueryResult

__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "COLLECTION_NAME",
    "CONFIG_PATH",
    "DLQ_PATH",
    "DOCS_DIR",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "FIRMWARE_DIR",
    "FORBIDDEN_EXTENSIONS",
    "FORBIDDEN_PATH_PATTERNS",
    "KNOWLEDGE_DB",
    "LINEAGE_DIR",
    "LOG_QUERIES",
    "MAX_CHUNK_CHARS",
    "PII_PATTERNS",
    "PROJECT_ROOT",
    "SRC_DIR",
    "TAG_KEYWORDS",
    "DocumentMeta",
    "IndexLineage",
    "QueryRequest",
    "QueryResult",
]
