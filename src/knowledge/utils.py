"""Utilitários da Camada C11.

Autor: Douglas Souza
Data: 2026-06-27
"""

import hashlib
import json
import re
from pathlib import Path
from typing import List, Set

from .constants import (
    FORBIDDEN_EXTENSIONS,
    FORBIDDEN_PATH_PATTERNS,
    PII_PATTERNS,
    TAG_KEYWORDS,
)


def deterministic_chunk_id(source: str, content_prefix: str) -> str:
    """Gera ID determinístico de 16 chars para evitar duplicatas.

    Usa o conteúdo completo no hash para minimizar colisões entre chunks
    de um mesmo arquivo com prefixos iniciais idênticos.
    """
    payload = f"{source}:{content_prefix}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def extract_tags_from_content(content: str) -> List[str]:
    """Extrai tags semânticas por heurística de keywords."""
    content_lower = content.lower()
    tags: Set[str] = set()
    for tag, keywords in TAG_KEYWORDS.items():
        if any(kw.lower() in content_lower for kw in keywords):
            tags.add(tag)
    return sorted(tags)


def detect_pii(text: str) -> List[str]:
    """Detecta padrões de PII no texto. Retorna lista de matches."""
    matches: List[str] = []
    for pattern in PII_PATTERNS:
        matches.extend(re.findall(pattern, text, re.IGNORECASE))
    return matches


def is_forbidden_path(path: Path) -> bool:
    """Verifica se o caminho contém dados brutos de ECG ou diretórios proibidos."""
    text = path.as_posix().lower()
    if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
        return True
    return any(p.lower() in text for p in FORBIDDEN_PATH_PATTERNS)


def compute_config_hash(config_path: Path) -> str:
    """Hash SHA256 do arquivo de config para invalidação de cache."""
    if not config_path.exists():
        return "no-config"
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]


def format_context_for_agent(docs: List[dict]) -> str:
    """Formata documentos para injeção em prompt de agente."""
    blocks = []
    for i, doc in enumerate(docs, 1):
        tags = doc.get("tags", [])
        block = (
            f"[{i}] Fonte: {doc['source']} | "
            f"Camada: {doc['layer']} | Versão: {doc['version']} | "
            f"Tags: {', '.join(tags)}\n"
            f"{doc['content']}\n"
        )
        blocks.append(block)
    return "\n---\n".join(blocks)


def parse_markdown_frontmatter(content: str) -> dict:
    """Extrai YAML frontmatter de markdown, se presente."""
    if not content.startswith("---"):
        return {}
    try:
        _, frontmatter, _ = content.split("---", 2)
        meta = {}
        for line in frontmatter.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        return meta
    except ValueError:
        return {}


def write_dlq(record: dict) -> None:
    """Escreve registro rejeitado na DLQ."""
    from .constants import DLQ_PATH

    DLQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DLQ_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
