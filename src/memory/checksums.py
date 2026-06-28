"""Funções de checksum SHA-256 para arquivos e diretórios."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Retorna o hex digest SHA-256 de um arquivo."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_directory(path: Path, pattern: str = "**/*") -> str:
    """Retorna um checksum agregado de todos os arquivos em um diretório."""
    h = hashlib.sha256()
    for p in sorted(path.glob(pattern)):
        if p.is_file():
            h.update(f"{p.relative_to(path)}:{sha256_file(p)}\n".encode())
    return h.hexdigest()
