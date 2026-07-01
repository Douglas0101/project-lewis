"""Testes da Ponte 2: consulta natural → SQL estruturado."""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
from sqlalchemy.orm import Session

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
def query_db(tmp_path):
    """Banco temporário com dados para testar queries."""
    db_path = tmp_path / "query_test.db"
    engine = get_engine(db_path)
    init_schema(engine)

    with Session(engine) as session:
        exp = Experiment(name="exp_query", stage="stage1", status="completed")
        session.add(exp)
        session.flush()

        run_ok = Run(
            experiment_id=exp.id,
            run_type="train",
            status="completed",
            start_time=datetime(2026, 6, 30, 10, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 6, 30, 10, 1, 0, tzinfo=timezone.utc),
        )
        run_fail = Run(
            experiment_id=exp.id,
            run_type="test",
            status="failed",
            start_time=datetime(2026, 6, 30, 11, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 6, 30, 11, 2, 0, tzinfo=timezone.utc),
        )
        session.add_all([run_ok, run_fail])
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

    yield engine, db_path, run_ok_id, run_fail_id


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
    assert result.row_count == 2


def test_answer_question_failed_runs(query_db):
    """Query de runs falhadas deve retornar apenas a run failed."""
    engine, db_path, _, run_fail_id = query_db
    req = NaturalQueryRequest(question="runs que falharam")
    result = answer_question(req, db_path=db_path)
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
    engine, db_path, run_ok_id, _ = query_db
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
