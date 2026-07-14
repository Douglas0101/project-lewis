"""Testes do TimelineRefresher."""

# pyright: reportArgumentType=false

from __future__ import annotations

import asyncio

from sqlalchemy import text

from src.tracking.db import get_engine, init_schema
from src.tracking.timeline_refresh import TimelineRefresher


def _seed_one_run(engine):
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


def test_refresh_sync_inserts_data(tmp_path):
    db_path = tmp_path / "ref.db"
    engine = get_engine(db_path)
    init_schema(engine)
    _seed_one_run(engine)
    refresher = TimelineRefresher(str(db_path))
    refresher.refresh_sync()
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM experiment_timeline_materialized")).scalar()
        assert total == 1


def test_refresh_loop_runs(tmp_path):
    db_path = tmp_path / "loop.db"
    engine = get_engine(db_path)
    init_schema(engine)
    refresher = TimelineRefresher(str(db_path), interval_seconds=0.1)

    async def runner():
        task = asyncio.create_task(refresher.refresh_loop())
        await asyncio.sleep(0.25)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(runner())
    assert refresher.last_refresh is not None
