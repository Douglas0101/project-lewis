"""Testes de stress — Ponte 2: Consulta Natural → SQL.

Hipótese de falha: LLM ou usuário malicioso pode gerar SQL destrutivo,
subqueries não autorizadas, UNION com tabelas externas ou extrair mais
linhas do que o permitido.
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

import pytest
import sqlparse

from src.knowledge.structured_query import (
    NaturalQueryRequest,
    answer_question,
    execute_custom_sql,
    validate_sql,
)


@pytest.fixture
def big_sql_db(tmp_path: Path):
    """Banco SQLite temporário com 50.000 runs para teste de MAX_ROWS."""
    db_path = tmp_path / "lewis_big.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE run (id INTEGER PRIMARY KEY, run_id TEXT, stage INTEGER)")
    conn.executemany(
        "INSERT INTO run (run_id, stage) VALUES (?, ?)",
        [(f"run_{i}", i % 4 + 1) for i in range(50_000)],
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.mark.stress
@pytest.mark.security
@pytest.mark.timeout(15)
def test_sql_injection_delete_blocked(populated_tracking_db) -> None:
    """T2.1: Pergunta de deleção deve ser bloqueada; nenhum DELETE gerado."""
    engine, db_path, *_ = populated_tracking_db
    req = NaturalQueryRequest(question="Delete todos os runs")
    result = answer_question(req, db_path=db_path)

    assert result.source == "schema_help" or "DELETE" not in result.sql.upper()
    assert "DELETE" not in result.sql.upper()


@pytest.mark.stress
@pytest.mark.security
@pytest.mark.timeout(15)
def test_sql_injection_union_blocked(big_sql_db) -> None:
    """T2.2: UNION SELECT deve ser rejeitado pela validação."""
    sql = "SELECT run_id FROM run UNION SELECT password FROM users"
    ok, reason = validate_sql(sql)
    assert not ok
    assert "UNION" in reason.upper() or "union" in reason.lower()


@pytest.mark.stress
@pytest.mark.security
@pytest.mark.timeout(15)
def test_sql_injection_drop_table_blocked(big_sql_db) -> None:
    """T2.3: String maliciosa '; DROP TABLE run; -- deve ser rejeitada."""
    sql = "'; DROP TABLE run; --"
    ok, reason = validate_sql(sql)
    assert not ok
    assert any(kw in reason.upper() for kw in ["DROP", "NÃO PERMITIDO"])

    # Verificar que a tabela ainda existe
    conn = sqlite3.connect(str(big_sql_db))
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='run'")
    assert cursor.fetchone() is not None
    conn.close()


@pytest.mark.stress
@pytest.mark.timeout(30)
def test_max_rows_enforcement(big_sql_db) -> None:
    """T2.4: Query que retornaria 50K deve ser limitada a MAX_ROWS (1.000)."""
    result = execute_custom_sql(
        "SELECT * FROM run ORDER BY id",
        max_rows=1000,
        db_path=big_sql_db,
    )
    assert result.row_count <= 1000
    assert result.truncated is True


@pytest.mark.stress
@pytest.mark.timeout(15)
def test_nonexistent_table_rejection(big_sql_db) -> None:
    """T2.5: Pergunta sobre tabela inexistente retorna schema_help."""
    req = NaturalQueryRequest(question="Quantos registros na tabela secrets?")
    result = answer_question(req, db_path=big_sql_db)
    # Fora do catálogo -> schema_help, sem executar SQL na tabela inexistente
    assert result.source == "schema_help"


@pytest.mark.stress
@pytest.mark.timeout(120)
def test_fuzzing_100_random_queries(big_sql_db) -> None:
    """T2.6: Fuzzing de 100 perguntas; ≥85% válidas, 0% malicioso executável."""
    templates = [
        "Quantos {noun} na tabela {table}?",
        "Qual o maior {noun} em {table}?",
        "Liste {table} onde {field} = {value}",
        "Média de {field} por {group} em {table}",
        "{verb} {table} com {condition}",
    ]
    nouns = ["runs", "experimentos", "alertas", "métricas", "usuários"]
    tables = ["run", "metric", "alert", "experiment"]
    fields = ["stage", "status", "severity", "value"]
    verbs = ["Delete", "Drop", "Update", "Insert", "Select", "Union", "Alter"]

    valid_count = 0
    malicious_count = 0
    handled_count = 0

    for _ in range(100):
        template = random.choice(templates)
        query = template.format(
            noun=random.choice(nouns),
            table=random.choice(tables + ["secrets", "users", "passwords"]),
            field=random.choice(fields),
            value=random.choice(["1", "'test'", "NULL", "; DROP TABLE run;"]),
            group=random.choice(fields),
            condition=random.choice(["status = 'failure'", "1=1", "'; --"]),
            verb=random.choice(verbs),
        )

        try:
            result = answer_question(NaturalQueryRequest(question=query), db_path=big_sql_db)
        except ValueError:
            # Bloqueio de segurança conta como proteção
            handled_count += 1
            continue

        if result.sql:
            try:
                sqlparse.parse(result.sql)
                valid_count += 1
            except Exception:
                pass

        sql_upper = result.sql.upper()
        if not result.source.startswith("schema_help") and any(
            kw in sql_upper for kw in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "UNION"]
        ):
            malicious_count += 1
        if result.source == "schema_help":
            handled_count += 1

    # Conta como "válido/protegido" tanto SQL gerado quanto resposta schema_help/bloqueio
    assert (
        valid_count + handled_count
    ) >= 85, f"Apenas {valid_count + handled_count}% foram tratados"
    assert malicious_count == 0, f"{malicious_count} queries maliciosas executadas"


@pytest.mark.stress
@pytest.mark.timeout(15)
def test_schema_mismatch_detection(big_sql_db) -> None:
    """T2.7: Schema alterado (coluna renomeada) — query ainda reflete tabela real."""
    conn = sqlite3.connect(str(big_sql_db))
    conn.execute("ALTER TABLE run ADD COLUMN model_name TEXT")
    conn.commit()
    conn.close()

    # Query válida sobre coluna existente
    result = execute_custom_sql(
        "SELECT model_name FROM run LIMIT 1",
        db_path=big_sql_db,
    )
    # coluna existe mas está vazia -> 1 linha com NULL
    assert result.row_count == 1
    assert result.rows[0]["model_name"] is None

    # Query sobre coluna inexistente deve falhar na execução
    with pytest.raises((sqlite3.OperationalError, ValueError)):
        execute_custom_sql(
            "SELECT nonexistent_column FROM run LIMIT 1",
            db_path=big_sql_db,
        )
