"""Fixtures compartilhadas para testes de stress das 3 pontes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from src.knowledge.db import get_connection, init_schema as init_knowledge_schema
from src.tracking.db import get_engine, init_schema as init_tracking_schema
from src.tracking.models import Alert, Experiment, Metric, Run


@pytest.fixture
def knowledge_db(tmp_path: Path):
    """Banco SQLite temporário para knowledge (sqlite-vec)."""
    db_path = tmp_path / "knowledge_stress.db"
    conn = get_connection(db_path)
    init_knowledge_schema(conn)
    yield db_path, conn
    conn.close()


@pytest.fixture
def tracking_db(tmp_path: Path):
    """Banco SQLite temporário para tracking com schema e dados mínimos."""
    db_path = tmp_path / "tracking_stress.db"
    engine = get_engine(db_path)
    init_tracking_schema(engine)
    yield engine, db_path


@pytest.fixture
def populated_tracking_db(tracking_db):
    """Banco de tracking com experimento, runs, métricas e alertas."""
    engine, db_path = tracking_db

    with Session(engine) as session:
        exp = Experiment(
            name="exp_stress",
            stage="stage1",
            status="completed",
            extra={"baseline_f1_macro": 0.55, "baseline_accuracy": 0.8},
        )
        session.add(exp)
        session.flush()

        run_ok = Run(
            experiment_id=exp.id,
            run_type="train",
            status="completed",
            start_time=datetime(2026, 6, 30, 10, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 6, 30, 10, 5, 0, tzinfo=timezone.utc),
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

        session.add(Metric(run_id=run_ok.id, namespace="global", name="f1_macro", value=0.65))
        session.add(Metric(run_id=run_ok.id, namespace="global", name="accuracy", value=0.9))
        session.add(Metric(run_id=run_fail.id, namespace="global", name="f1_macro", value=0.45))
        session.add(
            Alert(
                run_id=run_fail.id,
                severity="critical",
                category="performance_drop",
                message="F1 abaixo do QG",
            )
        )
        session.commit()

        run_ok_id = run_ok.id
        run_fail_id = run_fail.id
        experiment_id = exp.id

    yield engine, db_path, experiment_id, run_ok_id, run_fail_id
