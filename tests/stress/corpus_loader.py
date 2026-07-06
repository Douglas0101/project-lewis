"""Loader e validação do corpus adversarial para stress tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class AdversarialSample(BaseModel):
    """Schema de um exemplo adversarial do corpus."""

    id: str = Field(..., min_length=1)
    component: str = Field(..., pattern=r"^(structured_query|knowledge_retriever|mcp)$")
    category: str = Field(
        ...,
        pattern=r"^(sql_injection|prompt_injection|denial_of_service|boundary|semantic_trick)$",
    )
    input: str
    expected_behavior: str = Field(..., pattern=r"^(reject|sanitize|safe_result)$")
    description: str = Field(..., min_length=1)


CORPUS_PATH: Path = Path(__file__).with_name("adversarial_corpus.jsonl")


def load_corpus(
    component: str | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Carrega o corpus adversarial e filtra opcionalmente.

    Args:
        component: Filtro por componente (structured_query, knowledge_retriever, mcp).
        category: Filtro por categoria de ataque.

    Returns:
        Lista de dicionários validados contra AdversarialSample.

    Raises:
        FileNotFoundError: Se o arquivo JSONL não existir.
        ValueError: Se alguma linha violar o schema ou houver ID duplicado.
        ValueError: Se o arquivo estiver vazio.
    """
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus não encontrado: {CORPUS_PATH}")

    samples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with CORPUS_PATH.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido na linha {line_number}: {exc}") from exc

            try:
                sample = AdversarialSample.model_validate(data)
            except ValidationError as exc:
                raise ValueError(f"Schema inválido na linha {line_number}: {exc}") from exc

            if sample.id in seen_ids:
                raise ValueError(f"ID duplicado no corpus: {sample.id!r} (linha {line_number})")
            seen_ids.add(sample.id)

            if component is not None and sample.component != component:
                continue
            if category is not None and sample.category != category:
                continue

            samples.append(sample.model_dump())

    if not samples:
        raise ValueError("Corpus está vazio ou nenhum item corresponde aos filtros.")

    return samples
