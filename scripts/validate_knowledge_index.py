"""Validação simples do índice da Camada C11.

Assume que o índice já foi construído (ex.: via ``make knowledge-index``).
Verifica:
  - Existência de ``data/knowledge.db``
  - Número de chunks > 0
  - Camadas presentes no índice
  - DLQ acessível

Uso:
    uv run python scripts/validate_knowledge_index.py
"""

from __future__ import annotations

import sys
from typing import Any, Set

from src.knowledge.constants import DLQ_PATH, KNOWLEDGE_DB
from src.knowledge.db import count_chunks, get_connection


def main() -> int:
    errors: list[str] = []

    # 1. Banco existe.
    if not KNOWLEDGE_DB.exists():
        errors.append(f"Banco não encontrado: {KNOWLEDGE_DB}")
        _report(errors)
        return 1

    # 2. Chunks > 0.
    try:
        total = count_chunks()
    except Exception as exc:  # pragma: no cover
        errors.append(f"Falha ao consultar banco: {exc}")
        _report(errors)
        return 1

    if total == 0:
        errors.append("Índice vazio: nenhum chunk encontrado.")

    # 3. Camadas presentes.
    conn: Any = get_connection()
    try:
        layers: Set[str] = {
            row[0] for row in conn.execute("SELECT DISTINCT layer FROM knowledge_chunks")
        }
    finally:
        conn.close()

    if not layers:
        errors.append("Nenhuma camada encontrada no índice.")

    # 4. DLQ acessível.
    if not DLQ_PATH.parent.exists():
        errors.append(f"Diretório da DLQ não existe: {DLQ_PATH.parent}")
    else:
        try:
            with open(DLQ_PATH, "a", encoding="utf-8"):
                pass
        except OSError as exc:
            errors.append(f"DLQ não acessível para escrita: {exc}")

    _report(errors, total=total, layers=layers)
    return 1 if errors else 0


def _report(errors: list[str], total: int = 0, layers: Set[str] | None = None) -> None:
    if errors:
        print("[C11] Validação FALHOU:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("[C11] Validação OK")
        print(f"  - Chunks indexados: {total}")
        print(f"  - Camadas presentes: {sorted(layers or set())}")
        print(f"  - DLQ: {DLQ_PATH}")


if __name__ == "__main__":
    sys.exit(main())
