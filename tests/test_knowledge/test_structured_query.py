"""Testes da Ponte 2: consulta natural → SQL estruturado."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

import src.knowledge.structured_query as structured_query_module
from src.knowledge.structured_query import (
    NaturalQueryRequest,
    _match_catalog,
    answer_question,
    build_schema_context,
    execute_custom_sql,
    validate_sql,
)
from src.tracking.db import get_engine, init_schema
from src.tracking.models import Alert, Experiment, Metric, Run


@pytest.fixture
def audit_log_path(tmp_path, monkeypatch):
    """Redireciona o audit log para arquivo temporário isolado por teste."""
    path = tmp_path / "knowledge_structured_queries.jsonl"
    monkeypatch.setattr(structured_query_module, "_AUDIT_LOG", path)
    return path


@pytest.fixture
def query_db(tmp_path):
    """Banco temporário com dados para testar queries (dois usuários)."""
    db_path = tmp_path / "query_test.db"
    engine = get_engine(db_path)
    init_schema(engine)

    with Session(engine) as session:
        exp_a = Experiment(
            name="exp_query_a", stage="stage1", status="completed", owner_id="user_a"
        )
        exp_b = Experiment(
            name="exp_query_b", stage="stage2", status="completed", owner_id="user_b"
        )
        session.add_all([exp_a, exp_b])
        session.flush()

        run_ok = Run(
            experiment_id=exp_a.id,
            run_type="train",
            status="completed",
            owner_id="user_a",
            start_time=datetime(2026, 6, 30, 10, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 6, 30, 10, 1, 0, tzinfo=timezone.utc),
        )
        run_fail = Run(
            experiment_id=exp_a.id,
            run_type="test",
            status="failed",
            owner_id="user_a",
            start_time=datetime(2026, 6, 30, 11, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 6, 30, 11, 2, 0, tzinfo=timezone.utc),
        )
        run_fail_b = Run(
            experiment_id=exp_b.id,
            run_type="test",
            status="failed",
            owner_id="user_b",
            start_time=datetime(2026, 6, 30, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 6, 30, 12, 2, 0, tzinfo=timezone.utc),
        )
        session.add_all([run_ok, run_fail, run_fail_b])
        session.flush()

        session.add(Metric(run_id=run_ok.id, namespace="global", name="f1_macro", value=0.7))
        session.add(
            Alert(
                run_id=run_fail.id,
                severity="warning",
                category="performance_drop",
                message="falhou",
            )
        )
        session.commit()

        run_ok_id = run_ok.id
        run_fail_id = run_fail.id
        run_fail_b_id = run_fail_b.id

    yield engine, db_path, run_ok_id, run_fail_id, run_fail_b_id


def test_validate_sql_rejects_drop():
    """DROP TABLE deve ser rejeitado."""
    ok, reason = validate_sql("DROP TABLE run")
    assert not ok
    assert "drop" in reason.lower()


def test_validate_sql_rejects_delete():
    """DELETE deve ser rejeitado."""
    ok, reason = validate_sql("DELETE FROM run WHERE id = 1")
    assert not ok


def test_validate_sql_rejects_injection_attempt():
    """Tentativa clássica de SQLi deve ser rejeitada."""
    ok, reason = validate_sql("SELECT * FROM run; DROP TABLE run; --")
    assert not ok
    assert "drop" in reason.lower()


def test_validate_sql_accepts_select():
    """SELECT simples deve ser aceito."""
    ok, reason = validate_sql("SELECT * FROM run WHERE status = 'failed'")
    assert ok
    assert reason == ""


def test_match_catalog_last_failed_runs():
    """Deve reconhecer pergunta sobre runs falhadas."""
    matched = _match_catalog("últimas runs que falharam")
    assert matched is not None
    pattern, params = matched
    assert pattern.name == "last_failed_runs"
    assert not params


def test_match_catalog_metrics_for_run():
    """Deve reconhecer pergunta sobre métricas de uma run."""
    matched = _match_catalog("métricas da run 42")
    assert matched is not None
    pattern, params = matched
    assert pattern.name == "metrics_for_run"
    assert params["run_id"] == 42


def test_answer_question_timeline(query_db):
    """Timeline deve retornar todas as runs."""
    engine, db_path, *_ = query_db
    req = NaturalQueryRequest(question="linha do tempo")
    result = answer_question(req, db_path=db_path)
    assert result.source == "catalog:timeline"
    assert result.row_count == 3


def test_answer_question_failed_runs(query_db):
    """Query de runs falhadas deve retornar apenas a run failed do usuário."""
    engine, db_path, _, run_fail_id, _ = query_db
    req = NaturalQueryRequest(question="runs que falharam")
    result = answer_question(req, db_path=db_path, user_id="user_a")
    assert result.source == "catalog:last_failed_runs"
    assert result.row_count == 1
    assert result.rows[0]["run_id"] == run_fail_id


def test_answer_question_unknown_returns_schema(query_db):
    """Pergunta fora do catálogo deve retornar contexto de schema."""
    engine, db_path, *_ = query_db
    req = NaturalQueryRequest(question="qual a cor do cavalo branco de napoleão")
    result = answer_question(req, db_path=db_path)
    assert result.source == "schema_help"
    assert "schema_context" in result.rows[0]


def test_execute_custom_sql_readonly(query_db):
    """SQL customizado SELECT deve funcionar em modo read-only."""
    engine, db_path, run_ok_id, _, _ = query_db
    result = execute_custom_sql(
        "SELECT * FROM run WHERE id = :run_id",
        {"run_id": run_ok_id},
        db_path=db_path,
    )
    assert result.row_count == 1
    assert result.rows[0]["id"] == run_ok_id


def test_execute_custom_sql_rejects_malicious(query_db):
    """SQL customizado malicioso deve ser rejeitado."""
    with pytest.raises(ValueError):
        execute_custom_sql("DROP TABLE run")


def test_build_schema_context_includes_allowed_tables(query_db):
    """Schema context deve incluir tabelas permitidas."""
    engine, db_path, *_ = query_db
    context = build_schema_context(engine)
    assert "run" in context.tables
    assert "metric" in context.tables
    assert "alert" in context.tables


def test_answer_question_rls_filters_by_user(query_db):
    """RLS deve filtrar runs falhadas apenas do usuário solicitante."""
    engine, db_path, _, run_fail_id, run_fail_b_id = query_db
    req = NaturalQueryRequest(question="runs que falharam")

    result_a = answer_question(req, db_path=db_path, user_id="user_a")
    assert result_a.row_count == 1
    assert result_a.rows[0]["run_id"] == run_fail_id
    assert result_a.params.get("owner_id") == "user_a"

    result_b = answer_question(req, db_path=db_path, user_id="user_b")
    assert result_b.row_count == 1
    assert result_b.rows[0]["run_id"] == run_fail_b_id
    assert result_b.params.get("owner_id") == "user_b"


def test_answer_question_admin_bypass(query_db):
    """Admin deve ignorar RLS e ver runs de todos os usuários."""
    engine, db_path, _, run_fail_id, run_fail_b_id = query_db
    req = NaturalQueryRequest(question="runs que falharam")
    result = answer_question(req, db_path=db_path, user_id="admin", roles=["admin"])
    assert result.row_count == 2
    run_ids = {row["run_id"] for row in result.rows}
    assert run_ids == {run_fail_id, run_fail_b_id}
    assert "owner_id" not in result.params


def test_execute_custom_sql_rls(query_db):
    """RLS deve ser aplicado em SQL customizado com alias de tabela."""
    engine, db_path, run_ok_id, run_fail_id, run_fail_b_id = query_db
    sql = """
        SELECT r.id AS run_id, r.status
        FROM run r
        JOIN experiment e ON e.id = r.experiment_id
        WHERE r.status = 'failed'
        ORDER BY r.id
    """
    result = execute_custom_sql(sql, db_path=db_path, user_id="user_a")
    assert result.row_count == 1
    assert result.rows[0]["run_id"] == run_fail_id
    assert "r.owner_id = ?" in result.sql
    assert result.params.get("owner_id") == "user_a"


def test_answer_question_timeline_rls(query_db):
    """Timeline deve filtrar runs pelo owner_id."""
    engine, db_path, run_ok_id, run_fail_id, run_fail_b_id = query_db
    req = NaturalQueryRequest(question="linha do tempo")
    result = answer_question(req, db_path=db_path, user_id="user_a")
    assert result.source == "catalog:timeline"
    assert result.row_count == 2
    run_ids = {row["run_id"] for row in result.rows}
    assert run_ids == {run_ok_id, run_fail_id}
    assert result.params.get("owner_id") == "user_a"


def test_answer_question_timeline_health_pattern(query_db):
    """Deve reconhecer pergunta sobre status de saúde de forma case-insensitive."""
    engine, db_path, run_ok_id, run_fail_id, run_fail_b_id = query_db
    req = NaturalQueryRequest(question="runs healthy")
    result = answer_question(req, db_path=db_path, user_id="user_a")
    assert result.source == "catalog:timeline_health"
    assert result.row_count == 1
    assert result.rows[0]["run_id"] == run_ok_id


def test_answer_question_empty_user_id_bypasses_rls(query_db):
    """user_id vazio deve ignorar RLS e retornar todas as runs falhadas."""
    engine, db_path, _, run_fail_id, run_fail_b_id = query_db
    req = NaturalQueryRequest(question="runs que falharam")
    result = answer_question(req, db_path=db_path, user_id="")
    assert result.row_count == 2
    run_ids = {row["run_id"] for row in result.rows}
    assert run_ids == {run_fail_id, run_fail_b_id}


def test_answer_question_admin_bypass_exact_role(query_db):
    """Somente a role exata 'admin' (case-insensitive) deve bypassar RLS."""
    engine, db_path, _, run_fail_id, run_fail_b_id = query_db
    req = NaturalQueryRequest(question="runs que falharam")
    result = answer_question(req, db_path=db_path, user_id="user_a", roles=["superadmin"])
    assert result.row_count == 1
    assert result.rows[0]["run_id"] == run_fail_id


def test_execute_custom_sql_positional_binds(query_db):
    """SQL customizado com binds posicionais ? deve funcionar com RLS."""
    engine, db_path, run_ok_id, run_fail_id, run_fail_b_id = query_db
    sql = "SELECT id AS run_id, status FROM run WHERE status = ?"
    result = execute_custom_sql(sql, ("failed",), db_path=db_path, user_id="user_a")
    assert result.row_count == 1
    assert result.rows[0]["run_id"] == run_fail_id
    assert "owner_id = ?" in result.sql
    assert result.params.get("owner_id") == "user_a"


def test_execute_custom_sql_non_standard_alias(query_db):
    """RLS deve detectar alias não padrão após FROM."""
    engine, db_path, run_ok_id, run_fail_id, run_fail_b_id = query_db
    sql = """
        SELECT myrun.id AS run_id
        FROM run AS myrun
        JOIN experiment AS myexp ON myexp.id = myrun.experiment_id
        WHERE myrun.status = 'failed'
    """
    result = execute_custom_sql(sql, db_path=db_path, user_id="user_a")
    assert result.row_count == 1
    assert result.rows[0]["run_id"] == run_fail_id
    assert "myrun.owner_id = ?" in result.sql
    assert result.params.get("owner_id") == "user_a"


def test_execute_custom_sql_group_by_rls(query_db):
    """RLS deve injetar WHERE antes de GROUP BY em SQL customizado."""
    engine, db_path, run_ok_id, run_fail_id, run_fail_b_id = query_db
    sql = """
        SELECT status, COUNT(*) AS cnt
        FROM run
        GROUP BY status
    """
    result = execute_custom_sql(sql, db_path=db_path, user_id="user_a")
    assert result.row_count == 2  # completed, failed
    assert "WHERE run.owner_id = ? GROUP BY status" in result.sql
    assert result.params.get("owner_id") == "user_a"


def test_execute_custom_sql_max_rows_named_binds(query_db):
    """max_rows deve ser aplicado automaticamente em SQL com binds nomeados."""
    engine, db_path, run_ok_id, run_fail_id, run_fail_b_id = query_db
    result = execute_custom_sql(
        "SELECT id AS run_id FROM run ORDER BY id",
        {},
        max_rows=2,
        db_path=db_path,
    )
    assert result.row_count == 2
    assert result.truncated is True
    assert "LIMIT :max_rows" in result.sql


def test_execute_custom_sql_existing_limit_not_duplicated(query_db):
    """SQL que já possui LIMIT no nível top-level não deve receber outro LIMIT."""
    engine, db_path, run_ok_id, run_fail_id, run_fail_b_id = query_db
    result = execute_custom_sql(
        "SELECT id AS run_id FROM run ORDER BY id LIMIT 1",
        {},
        max_rows=1000,
        db_path=db_path,
    )
    assert result.row_count == 1
    assert result.truncated is False
    assert result.sql.count("LIMIT") == 1


def test_execute_custom_sql_rls_named_binds(query_db):
    """RLS deve funcionar com binds nomeados em SQL customizado."""
    engine, db_path, run_ok_id, run_fail_id, run_fail_b_id = query_db
    sql = """
        SELECT r.id AS run_id, r.status
        FROM run r
        JOIN experiment e ON e.id = r.experiment_id
        WHERE r.status = :status
        ORDER BY r.id
    """
    result = execute_custom_sql(
        sql,
        {"status": "failed"},
        db_path=db_path,
        user_id="user_a",
    )
    assert result.row_count == 1
    assert result.rows[0]["run_id"] == run_fail_id
    assert "r.owner_id = :owner_id" in result.sql
    assert result.params.get("owner_id") == "user_a"
    assert result.params.get("status") == "failed"


def test_audit_id_unique():
    """IDs de auditoria devem ser únicos mesmo gerados em sequência rápida."""
    from src.knowledge.structured_query import _audit_id

    ids = {_audit_id() for _ in range(1000)}
    assert len(ids) == 1000


def _load_audit_entries(audit_log_path: Path) -> list[dict]:
    """Carrega entradas JSONL do audit log."""
    if not audit_log_path.exists():
        return []
    with open(audit_log_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_audit_log_catalog_contains_user_id_sql_params(query_db, audit_log_path):
    """Audit log de consulta catalog deve conter user_id, SQL e parâmetros."""
    engine, db_path, run_ok_id, run_fail_id, run_fail_b_id = query_db
    req = NaturalQueryRequest(question="runs que falharam")
    result = answer_question(req, db_path=db_path, user_id="user_a")

    entries = _load_audit_entries(audit_log_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["user_id"] == "user_a"
    assert entry["audit_id"] == result.audit_id
    assert entry["source"] == "catalog"
    assert entry["sql"] is not None
    assert "owner_id" in entry["sql"]
    assert entry["params"]["owner_id"] == "user_a"
    assert "max_rows" in entry["params"]


def test_audit_log_rls_applied_sql(query_db, audit_log_path):
    """SQL final auditado deve conter o filtro RLS owner_id."""
    engine, db_path, run_ok_id, run_fail_id, run_fail_b_id = query_db
    req = NaturalQueryRequest(question="runs que falharam")
    result = answer_question(req, db_path=db_path, user_id="user_b")

    entry = _load_audit_entries(audit_log_path)[0]
    assert "e.owner_id = :owner_id" in entry["sql"] or "owner_id = ?" in entry["sql"]
    assert entry["params"]["owner_id"] == "user_b"
    assert entry["sql"] == result.sql


def test_audit_log_rejected_custom_sql(query_db, audit_log_path):
    """SQL customizado rejeitado deve ser auditado com source custom_sql_rejected."""
    engine, db_path, *_ = query_db
    with pytest.raises(ValueError):
        execute_custom_sql(
            "DROP TABLE run",
            db_path=db_path,
            user_id="user_a",
        )

    entries = _load_audit_entries(audit_log_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["user_id"] == "user_a"
    assert entry["source"] == "custom_sql_rejected"
    assert entry["sql"] == "DROP TABLE run"
    assert entry["reason"]


def test_audit_log_schema_help(query_db, audit_log_path):
    """Pergunta fora do catálogo deve gerar audit entry de schema_help."""
    engine, db_path, *_ = query_db
    req = NaturalQueryRequest(question="qual a cor do cavalo branco de napoleão")
    result = answer_question(req, db_path=db_path, user_id="user_a")

    entries = _load_audit_entries(audit_log_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["user_id"] == "user_a"
    assert entry["audit_id"] == result.audit_id
    assert entry["source"] == "schema_help"
    assert entry["sql"] is None
    assert entry["params"] == req.params
    assert entry["question"] == req.question
