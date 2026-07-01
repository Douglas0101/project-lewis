"""Testes de integração entre tracking e outros subsistemas."""

from __future__ import annotations

from sqlalchemy import text

from src.tracking.db import get_engine, init_schema
from src.tracking.integrations import refresh_timeline_after_run


def test_refresh_timeline_after_run(tmp_path, monkeypatch):
    db_path = tmp_path / "hook.db"
    engine = get_engine(db_path)
    init_schema(engine)
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO experiment (name, stage, status, created_at) "
                "VALUES ('exp', 'stage1', 'completed', '2026-06-30 09:00:00')"
            )
        )
        exp_id = conn.execute(text("SELECT last_insert_rowid()")).scalar()
        conn.execute(
            text(
                "INSERT INTO run (experiment_id, run_type, status, start_time) "
                "VALUES (:eid, 'train', 'completed', '2026-06-30 10:00:00')"
            ),
            {"eid": exp_id},
        )
        conn.commit()

    monkeypatch.setattr("src.tracking.integrations.get_db_path", lambda: db_path)
    refresh_timeline_after_run()

    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM experiment_timeline_materialized")
        ).scalar()
        assert total == 1
