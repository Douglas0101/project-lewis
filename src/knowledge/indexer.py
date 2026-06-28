"""Indexador semântico da Camada C11.

Responsabilidade: carregar documentos markdown e código-fonte,
segmentar preservando hierarquia, extrair metadados 3D,
gerar embeddings e persistir no sqlite-vec.

Restrições:
- Não indexar arquivos fora de docs/, src/, firmware/src/.
- Rejeitar chunks contendo PII (LGPD).
- Determinístico: reindexação gera IDs idênticos.
- Nunca indexar dados brutos de ECG (.dat, .mat, .hea).

Autor: Douglas Souza
Data: 2026-06-27
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List

import sqlite_vec
from sentence_transformers import SentenceTransformer

from .constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONFIG_PATH,
    DOCS_DIR,
    EMBEDDING_MODEL,
    FIRMWARE_DIR,
    KNOWLEDGE_DB,
    LAYER_MAP,
    LINEAGE_DIR,
    MAX_CHUNK_CHARS,
    PROJECT_ROOT,
    SRC_DIR,
)
from .db import get_connection, init_schema
from .schemas import DocumentMeta, IndexLineage
from .utils import (
    compute_config_hash,
    detect_pii,
    deterministic_chunk_id,
    extract_tags_from_content,
    is_forbidden_path,
    parse_markdown_frontmatter,
    write_dlq,
)


class Chunk:
    """Chunk de documento com metadados e conteúdo."""

    def __init__(
        self,
        content: str,
        source: str,
        header_1: str | None = None,
        header_2: str | None = None,
    ):
        self.content = content
        self.source = source
        self.header_1 = header_1
        self.header_2 = header_2


def resolve_layer(file_path: Path) -> str:
    """Resolve camada arquitetural a partir do nome do arquivo."""
    name = file_path.name
    for prefix, layer in LAYER_MAP.items():
        if prefix in name:
            return layer
    return "GENERAL"


def resolve_version(file_path: Path) -> str:
    """Extrai versão do nome do arquivo (vX.Y)."""
    match = re.search(r"v(\d+\.\d+)", file_path.name)
    return f"v{match.group(1)}" if match else "unversioned"


def load_markdown_documents() -> List[Chunk]:
    """Carrega todos os arquivos markdown do projeto."""
    chunks: List[Chunk] = []
    md_paths = list(DOCS_DIR.rglob("*.md")) + list(PROJECT_ROOT.glob("*.md"))

    for md_path in md_paths:
        if not md_path.exists() or is_forbidden_path(md_path):
            continue
        raw = md_path.read_text(encoding="utf-8")
        parse_markdown_frontmatter(raw)  # descartado por simplicidade; pode enriquecer metadados
        # Split por headers
        parts = raw.split("\n# ")
        for i, part in enumerate(parts):
            if i == 0 and not part.strip().startswith("#"):
                header = None
                body = part
            else:
                lines = part.splitlines()
                header = lines[0].lstrip("# ").strip() if lines else None
                body = "\n".join(lines[1:])
            if len(body) > MAX_CHUNK_CHARS:
                sub_chunks = split_text_recursive(body, CHUNK_SIZE, CHUNK_OVERLAP)
                for sub in sub_chunks:
                    chunks.append(
                        Chunk(sub, md_path.relative_to(PROJECT_ROOT).as_posix(), header_1=header)
                    )
            else:
                chunks.append(
                    Chunk(body, md_path.relative_to(PROJECT_ROOT).as_posix(), header_1=header)
                )
    return chunks


def load_code_documents() -> List[Chunk]:
    """Carrega código-fonte Python e C como documentos."""
    chunks: List[Chunk] = []

    for py_path in SRC_DIR.rglob("*.py"):
        if "__pycache__" in str(py_path) or is_forbidden_path(py_path):
            continue
        content = py_path.read_text(encoding="utf-8")
        for sub in split_text_recursive(content, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(Chunk(sub, py_path.relative_to(PROJECT_ROOT).as_posix()))

    for ext in ("*.c", "*.h", "*.cpp"):
        for c_path in FIRMWARE_DIR.rglob(ext):
            if is_forbidden_path(c_path):
                continue
            content = c_path.read_text(encoding="utf-8")
            for sub in split_text_recursive(content, CHUNK_SIZE, CHUNK_OVERLAP):
                chunks.append(Chunk(sub, c_path.relative_to(PROJECT_ROOT).as_posix()))

    return chunks


def split_text_recursive(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Segmentador recursivo simples por parágrafos/frases/palavras."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    result: List[str] = []
    current = ""
    for sep in separators:
        parts = text.split(sep) if sep else list(text)
        for part in parts:
            if len(current) + len(part) + len(sep) > chunk_size and current:
                result.append(current.strip())
                current = current[-overlap:] if overlap > 0 else ""
            current += part + sep
        if current.strip():
            result.append(current.strip())
        if len(result) > 1:
            return result
    if not result and text.strip():
        result.append(text.strip())
    return result


def enrich_metadata(chunks: List[Chunk]) -> List[tuple[DocumentMeta, str]]:
    """Enriquece cada chunk com metadados 3D e rejeita PII."""
    enriched: List[tuple[DocumentMeta, str]] = []

    for chunk in chunks:
        source = chunk.source
        file_path = Path(source)
        content = chunk.content

        if is_forbidden_path(file_path):
            continue

        pii_matches = detect_pii(content)
        if pii_matches:
            write_dlq({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": source,
                "reason": "PII_DETECTED",
                "matches": pii_matches,
                "action": "REJECTED",
            })
            continue

        layer = resolve_layer(file_path)
        version = resolve_version(file_path)
        tags = extract_tags_from_content(content)
        chunk_id = deterministic_chunk_id(source, content)

        meta = DocumentMeta(
            source=source,
            layer=layer,
            version=version,
            tags=tags,
            filename=file_path.name,
            chunk_id=chunk_id,
            header_1=chunk.header_1,
            header_2=chunk.header_2,
        )
        enriched.append((meta, content))

    return enriched


def build_index() -> IndexLineage:
    """Pipeline completo de indexação. Retorna linhagem."""
    start_time = time.perf_counter()
    KNOWLEDGE_DB.parent.mkdir(parents=True, exist_ok=True)
    LINEAGE_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    md_chunks = load_markdown_documents()
    code_chunks = load_code_documents()
    all_chunks = md_chunks + code_chunks

    enriched = enrich_metadata(all_chunks)

    # Deduplicação determinística por chunk_id (mesmo conteúdo no mesmo source)
    unique_enriched: List[tuple[DocumentMeta, str]] = []
    seen_ids: set[str] = set()
    for meta, content in enriched:
        if meta.chunk_id in seen_ids:
            write_dlq({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": meta.source,
                "chunk_id": meta.chunk_id,
                "reason": "DUPLICATE_CHUNK_ID",
                "action": "REJECTED",
            })
            continue
        seen_ids.add(meta.chunk_id)
        unique_enriched.append((meta, content))

    conn = get_connection()
    init_schema(conn)
    conn.execute("DELETE FROM knowledge_chunks")

    contents = [f"{m.source}\n{c}" for m, c in unique_enriched]
    embeddings = model.encode(contents, normalize_embeddings=True, show_progress_bar=False)

    for i, ((meta, content), emb) in enumerate(zip(unique_enriched, embeddings)):
        conn.execute(
            """
            INSERT INTO knowledge_chunks
            (chunk_id, source, layer, version, tags, header_1, header_2, content, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meta.chunk_id,
                meta.source,
                meta.layer,
                meta.version,
                json.dumps(meta.tags),
                meta.header_1 or "",
                meta.header_2 or "",
                content,
                sqlite_vec.serialize_float32(emb),
            ),
        )
        if (i + 1) % 100 == 0:
            conn.commit()
    conn.commit()
    conn.close()

    duration = time.perf_counter() - start_time

    metas = [m for m, _ in unique_enriched]
    layers = sorted({m.layer for m in metas})
    versions = sorted({m.version for m in metas})
    db_size = KNOWLEDGE_DB.stat().st_size if KNOWLEDGE_DB.exists() else 0

    lineage = IndexLineage(
        total_files=len({m.source for m in metas}),
        total_chunks=len(metas),
        layers_found=layers,
        versions_found=versions,
        index_duration_sec=duration,
        db_size_bytes=db_size,
        config_hash=compute_config_hash(CONFIG_PATH),
    )

    lineage_path = LINEAGE_DIR / f"index_{lineage.timestamp.isoformat()}.json"
    lineage_path.write_text(lineage.model_dump_json(indent=2), encoding="utf-8")

    print(f"[indexer] {len(metas)} chunks indexados em {duration:.2f}s")
    print(f"[indexer] Camadas: {layers}")
    print(f"[indexer] Versões: {versions}")
    print(f"[indexer] Tamanho DB: {db_size / 1024 / 1024:.2f} MB")

    return lineage


if __name__ == "__main__":
    build_index()
