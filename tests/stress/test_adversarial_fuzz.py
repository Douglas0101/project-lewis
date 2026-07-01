"""Fuzzing adversarial contra a Ponte 2 (NL → SQL).

Usa o corpus adversarial para garantir que entradas maliciosas sejam
rejeitadas, sanitizadas ou convertidas em respostas seguras, e que toda
tentativa de SQL customizado seja auditada.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.knowledge.structured_query as structured_query_module
from src.knowledge.structured_query import (
    NaturalQueryRequest,
    StructuredQueryResult,
    answer_question,
    execute_custom_sql,
)
from tests.stress.corpus_loader import load_corpus

pytestmark = pytest.mark.stress

REJECT = "reject"
SANITIZE = "sanitize"
SAFE_RESULT = "safe_result"
COMPONENT = "structured_query"
FUZZ_USER = "fuzz_user"

SCHEMA_HELP = "schema_help"
CATALOG = "catalog"

CORPUS = load_corpus(component=COMPONENT)


@pytest.fixture
def fuzz_audit_log(tmp_path: Path, monkeypatch) -> Path:
    """Redireciona o audit log para arquivo temporário isolado."""
    path = tmp_path / "adversarial_fuzz_audit.jsonl"
    monkeypatch.setattr(structured_query_module, "_AUDIT_LOG", path)
    return path


def _read_audit_entries(path: Path) -> list[dict]:
    """Carrega entradas JSONL do audit log, ignorando linhas malformadas."""
    if not path.exists():
        return []
    entries: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


@pytest.mark.parametrize("sample", CORPUS, ids=[s["id"] for s in CORPUS])
@pytest.mark.timeout(30)
def test_structured_query_adversarial_corpus(
    sample: dict,
    populated_tracking_db,
    fuzz_audit_log: Path,
) -> None:
    """Cada item do corpus adversarial deve ser tratado de forma segura."""
    engine, db_path, *_ = populated_tracking_db

    if sample["expected_behavior"] == REJECT:
        with pytest.raises(ValueError):
            execute_custom_sql(
                sample["input"],
                db_path=db_path,
                user_id=FUZZ_USER,
                roles=(),
            )

    else:
        result = answer_question(
            NaturalQueryRequest(question=sample["input"]),
            db_path=db_path,
            user_id=FUZZ_USER,
        )

        assert isinstance(result, StructuredQueryResult)

        if sample["expected_behavior"] == SANITIZE:
            assert result.source.startswith(CATALOG)
            assert sample["input"] not in result.sql
            assert all(
                keyword not in result.sql.lower()
                for keyword in structured_query_module._FORBIDDEN_KEYWORDS
            )

        elif sample["expected_behavior"] == SAFE_RESULT:
            assert result.source == SCHEMA_HELP or result.source.startswith(CATALOG)


@pytest.mark.timeout(30)
def test_audit_log_records_adversarial_custom_sql(
    populated_tracking_db,
    fuzz_audit_log: Path,
) -> None:
    """Toda chamada adversarial a execute_custom_sql deve gerar audit entry."""
    engine, db_path, *_ = populated_tracking_db

    reject_samples = [s for s in CORPUS if s["expected_behavior"] == REJECT]

    for sample in reject_samples:
        with pytest.raises(ValueError):
            execute_custom_sql(
                sample["input"],
                db_path=db_path,
                user_id=FUZZ_USER,
                roles=(),
            )

    entries = _read_audit_entries(fuzz_audit_log)
    assert len(entries) == len(reject_samples)
    for entry, sample in zip(entries, reject_samples):
        assert entry["source"] == "custom_sql_rejected"
        assert entry["user_id"] == FUZZ_USER
        assert entry["sql"] == sample["input"]
        assert entry.get("reason")
