"""Testes unitários para o loader do corpus adversarial."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.stress import corpus_loader
from tests.stress.corpus_loader import AdversarialSample, load_corpus


def _write_corpus(tmp_path: Path, lines: list[dict]) -> Path:
    corpus = tmp_path / "corpus.jsonl"
    with corpus.open("w", encoding="utf-8") as handle:
        for item in lines:
            handle.write(json.dumps(item) + "\n")
    return corpus


def test_corpus_loads_expected_number() -> None:
    """O corpus deve carregar todos os itens válidos."""
    samples = load_corpus()
    assert len(samples) >= 20
    assert all("id" in s and "component" in s for s in samples)


def test_corpus_filter_by_component() -> None:
    """Filtrar por componente deve retornar apenas itens correspondentes."""
    samples = load_corpus(component="structured_query")
    assert len(samples) > 0
    assert all(s["component"] == "structured_query" for s in samples)


def test_corpus_filter_by_category() -> None:
    """Filtrar por categoria deve restringir o corpus corretamente."""
    samples = load_corpus(category="sql_injection")
    assert len(samples) > 0
    assert all(s["category"] == "sql_injection" for s in samples)


def test_corpus_filter_by_component_and_category() -> None:
    """Filtros combinados devem funcionar conjuntamente."""
    samples = load_corpus(component="mcp", category="prompt_injection")
    assert len(samples) > 0
    assert all(s["component"] == "mcp" and s["category"] == "prompt_injection" for s in samples)


def test_schema_rejects_invalid_component() -> None:
    """Componente fora da enumeração deve falhar na validação."""
    with pytest.raises(ValidationError):
        AdversarialSample.model_validate(
            {
                "id": "bad-001",
                "component": "unknown_component",
                "category": "sql_injection",
                "input": "DROP TABLE run",
                "expected_behavior": "reject",
                "description": "componente inválido",
            }
        )


def test_schema_rejects_invalid_expected_behavior() -> None:
    """Comportamento esperado fora do conjunto permitido deve falhar."""
    with pytest.raises(ValidationError):
        AdversarialSample.model_validate(
            {
                "id": "bad-002",
                "component": "mcp",
                "category": "prompt_injection",
                "input": "ignore",
                "expected_behavior": "panic",
                "description": "comportamento inválido",
            }
        )


def test_schema_rejects_missing_field() -> None:
    """Campos obrigatórios ausentes devem falhar na validação."""
    with pytest.raises(ValidationError):
        AdversarialSample.model_validate(
            {
                "id": "bad-003",
                "component": "mcp",
                "category": "boundary",
                "input": "x",
                # expected_behavior e description ausentes
            }
        )


def test_load_corpus_rejects_malformed_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Linha JSON malformada no corpus deve levantar ValueError com número da linha."""
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("this is not json\n", encoding="utf-8")
    monkeypatch.setattr(corpus_loader, "CORPUS_PATH", corpus)

    with pytest.raises(ValueError, match="JSON inválido na linha 1"):
        load_corpus()


def test_load_corpus_rejects_schema_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Violação de schema em tempo de carga deve levantar ValueError."""
    bad_line = {
        "id": "bad-004",
        "component": "mcp",
        "category": "invalid_category",
        "input": "x",
        "expected_behavior": "reject",
        "description": "categoria inválida",
    }
    corpus = _write_corpus(tmp_path, [bad_line])
    monkeypatch.setattr(corpus_loader, "CORPUS_PATH", corpus)

    with pytest.raises(ValueError, match=r"Schema inválido na linha 1"):
        load_corpus()


def test_load_corpus_rejects_duplicate_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """IDs duplicados no corpus devem ser detectados e rejeitados."""
    sample = {
        "id": "dup-001",
        "component": "mcp",
        "category": "prompt_injection",
        "input": "ignore",
        "expected_behavior": "reject",
        "description": "duplicado",
    }
    corpus = _write_corpus(tmp_path, [sample, sample])
    monkeypatch.setattr(corpus_loader, "CORPUS_PATH", corpus)

    with pytest.raises(ValueError, match=r"ID duplicado no corpus: 'dup-001' \(linha 2\)"):
        load_corpus()


def test_load_corpus_skips_blank_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Linhas em branco devem ser ignoradas sem afetar a carga."""
    sample = {
        "id": "blank-001",
        "component": "knowledge_retriever",
        "category": "boundary",
        "input": "ok",
        "expected_behavior": "safe_result",
        "description": "teste de linha em branco",
    }
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n" + json.dumps(sample) + "\n\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(corpus_loader, "CORPUS_PATH", corpus)

    samples = load_corpus()
    assert len(samples) == 1
    assert samples[0]["id"] == "blank-001"


def test_load_corpus_invalid_category_regex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Categoria fora do padrão regex deve falhar no loader com ValueError."""
    bad_line = {
        "id": "regex-001",
        "component": "structured_query",
        "category": "xss_attempt",
        "input": "<script>",
        "expected_behavior": "sanitize",
        "description": "categoria fora do regex permitido",
    }
    corpus = _write_corpus(tmp_path, [bad_line])
    monkeypatch.setattr(corpus_loader, "CORPUS_PATH", corpus)

    with pytest.raises(ValueError, match=r"Schema inválido na linha 1"):
        load_corpus()
