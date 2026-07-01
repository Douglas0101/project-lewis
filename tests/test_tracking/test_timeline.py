"""Testes da view experiment_timeline e API de timeline."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from src.tracking.db import get_engine, init_schema
from src.tracking.models import Alert, Experiment, Metric, Run
from src.tracking.timeline import ALLOWED_SORT_COLUMNS, TimelineFilters, get_timeline
from src.tracking.timeline_refresh import TimelineRefresher


@pytest.fixture
def timeline_db(tmp_path):
    """Banco SQLite temporário com schema e dados de teste."""
    db_path = tmp_path / "timeline_test.db"
    engine = get_engine(db_path)
    init_schema(engine)

    with Session(engine) as session:
        exp = Experiment(
            name="exp_test",
            stage="stage1",
            status="completed",
            extra={"baseline_f1_macro": 0.5},
        )
        session.add(exp)
        session.flush()

        run_ok = Run(
            experiment_id=exp.id,
            run_type="train",
            status="completed",
            start_time=datetime(2026, 6, 30, 10, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 6, 30, 10, 1, 0, tzinfo=timezone.utc),
        )
        run_bad = Run(
            experiment_id=exp.id,
            run_type="test",
            status="failed",
            start_time=datetime(2026, 6, 30, 11, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 6, 30, 11, 2, 0, tzinfo=timezone.utc),
        )
        session.add_all([run_ok, run_bad])
        session.flush()

        session.add(Metric(run_id=run_ok.id, namespace="global", name="f1_macro", value=0.65))
        session.add(Metric(run_id=run_ok.id, namespace="global", name="accuracy", value=0.9))
        session.add(Metric(run_id=run_bad.id, namespace="global", name="f1_macro", value=0.45))
        session.add(
            Alert(
                run_id=run_bad.id, severity="warning", category="performance_drop", message="falhou"
            )
        )
        session.commit()

        exp_id = exp.id
        run_ok_id = run_ok.id
        run_bad_id = run_bad.id

    TimelineRefresher(str(db_path)).refresh_sync()

    yield engine, exp_id, run_ok_id, run_bad_id


def test_timeline_returns_rows(timeline_db):
    """A view deve retornar runs agregadas corretamente."""
    engine, _, run_ok_id, run_bad_id = timeline_db
    rows, total = get_timeline(TimelineFilters(), engine)

    assert total == 2
    ok = next(r for r in rows if r.run_id == run_ok_id)
    bad = next(r for r in rows if r.run_id == run_bad_id)

    assert ok.final_f1_macro == pytest.approx(0.65)
    assert ok.health_status == "HEALTHY"
    assert bad.status == "failed"
    assert bad.health_status == "FAILED"
    assert bad.warning_alerts == 1


def test_timeline_filters_by_status(timeline_db):
    """Filtro por status deve funcionar."""
    engine, _, _, run_bad_id = timeline_db
    rows, total = get_timeline(TimelineFilters(status="failed"), engine)
    assert total == 1
    assert rows[0].run_id == run_bad_id


def test_timeline_filters_by_health(timeline_db):
    """Filtro por health_status deve funcionar."""
    engine, _, run_ok_id, _ = timeline_db
    rows, total = get_timeline(TimelineFilters(health_status="HEALTHY"), engine)
    assert total == 1
    assert rows[0].run_id == run_ok_id


def test_timeline_date_order_validation():
    """date_to anterior a date_from deve ser rejeitado."""
    with pytest.raises(ValueError):
        TimelineFilters(
            date_from=datetime(2026, 6, 30, 12, 0, 0),
            date_to=datetime(2026, 6, 30, 10, 0, 0),
        )


def test_timeline_invalid_sort_rejected(timeline_db):
    """Coluna de ordenação inválida deve ser rejeitada."""
    engine, *_ = timeline_db
    with pytest.raises(ValueError):
        get_timeline(TimelineFilters(sort_by="invalid_column"), engine)


def test_allowed_sort_columns_cover_view():
    """Colunas permitidas devem incluir as principais da view."""
    assert "final_f1_macro" in ALLOWED_SORT_COLUMNS
    assert "health_status" in ALLOWED_SORT_COLUMNS
    assert "run_start" in ALLOWED_SORT_COLUMNS
