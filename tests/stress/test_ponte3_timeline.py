"""Testes de stress — Ponte 3: Linha do Tempo Consolidada.

Hipótese de falha: volume extremo, joins vazios, datas invertidas,
colunas inexistentes e exclusões podem tornar a view lenta, inconsistente
ou vulnerável a injeção.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import text

from src.tracking.db import get_engine, init_schema
from src.tracking.timeline import TimelineFilters, get_timeline
from src.tracking.timeline_refresh import TimelineRefresher


def _seed_timeline_db(db_path: Path, n_runs: int) -> None:
    """Popula banco temporário com N runs, métricas e alertas via SQL puro."""
    engine = get_engine(db_path)
    init_schema(engine)

    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO experiment (name, stage, status, created_at) "
                "VALUES (:name, :stage, :status, :created_at)"
            ),
            {
                "name": "exp_stress",
                "stage": "stage1",
                "status": "completed",
                "created_at": "2026-06-30 10:00:00",
            },
        )
        exp_id = conn.execute(text("SELECT last_insert_rowid()")).scalar()

        runs = [
            {
                "experiment_id": exp_id,
                "run_type": "train" if i % 2 == 0 else "test",
                "status": "completed" if i % 10 != 0 else "failed",
                "start_time": "2026-06-30 10:00:00",
                "end_time": "2026-06-30 10:01:00",
            }
            for i in range(n_runs)
        ]
        conn.execute(
            text(
                "INSERT INTO run (experiment_id, run_type, status, start_time, end_time) "
                "VALUES (:experiment_id, :run_type, :status, :start_time, :end_time)"
            ),
            runs,
        )
        last_run_id = conn.execute(text("SELECT last_insert_rowid()")).scalar()
        first_run_id = last_run_id - n_runs + 1

        metrics = [
            {
                "run_id": first_run_id + i,
                "namespace": "global",
                "name": "f1_macro",
                "value": 0.5 + (i % 100) / 1000.0,
                "step": None,
                "recorded_at": "2026-06-30 10:00:00",
            }
            for i in range(n_runs)
        ]
        conn.execute(
            text(
                "INSERT INTO metric (run_id, namespace, name, value, step, recorded_at) "
                "VALUES (:run_id, :namespace, :name, :value, :step, :recorded_at)"
            ),
            metrics,
        )

        alerts = [
            {
                "run_id": first_run_id + i,
                "severity": "warning",
                "category": "performance_drop",
                "message": "stress",
                "recorded_at": "2026-06-30 10:00:00",
                "resolved": False,
            }
            for i in range(0, n_runs, 20)
        ]
        if alerts:
            conn.execute(
                text(
                    "INSERT INTO alert "
                    "(run_id, severity, category, message, recorded_at, resolved) "
                    "VALUES (:run_id, :severity, :category, :message, :recorded_at, :resolved)"
                ),
                alerts,
            )
        conn.commit()


@pytest.mark.stress
@pytest.mark.timeout(60)
def test_timeline_performance_100k_records(tmp_path: Path) -> None:
    """T3.1: 100K runs com métricas e alertas; query na materializada < 3s."""
    db_path = tmp_path / "lewis_100k.db"
    _seed_timeline_db(db_path, 100_000)

    refresher = TimelineRefresher(str(db_path))
    refresher.refresh_sync()

    engine = get_engine(db_path)
    start = time.perf_counter()
    rows, total = get_timeline(TimelineFilters(limit=1000), engine)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, f"Query demorou {elapsed:.2f}s"
    assert len(rows) == 1000
    assert total == 100_000


@pytest.mark.stress
@pytest.mark.timeout(15)
def test_timeline_left_join_empty(tmp_path: Path) -> None:
    """T3.2: Run sem métricas deve aparecer com NULLs, sem quebrar a materializada."""
    db_path = tmp_path / "lewis_empty.db"
    engine = get_engine(db_path)
    init_schema(engine)

    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO experiment (name, stage, status, created_at) "
                "VALUES (:name, :stage, :status, :created_at)"
            ),
            {
                "name": "exp_empty",
                "stage": "stage1",
                "status": "completed",
                "created_at": "2026-06-30 10:00:00",
            },
        )
        exp_id = conn.execute(text("SELECT last_insert_rowid()")).scalar()
        conn.execute(
            text(
                "INSERT INTO run (experiment_id, run_type, status, start_time) "
                "VALUES (:experiment_id, :run_type, :status, :start_time)"
            ),
            {
                "experiment_id": exp_id,
                "run_type": "train",
                "status": "completed",
                "start_time": "2026-06-30 10:00:00",
            },
        )
        conn.commit()

    TimelineRefresher(str(db_path)).refresh_sync()

    rows, total = get_timeline(TimelineFilters(), engine)
    assert total == 1
    assert rows[0].final_f1_macro is None
    assert rows[0].health_status is not None


@pytest.mark.stress
@pytest.mark.timeout(30)
def test_timeline_aggregation_10k_alerts(tmp_path: Path) -> None:
    """T3.3: Run com 10.000 alertas; COUNT deve ser exato."""
    db_path = tmp_path / "lewis_alerts.db"
    engine = get_engine(db_path)
    init_schema(engine)

    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO experiment (name, stage, status, created_at) "
                "VALUES (:name, :stage, :status, :created_at)"
            ),
            {
                "name": "exp_alerts",
                "stage": "stage1",
                "status": "completed",
                "created_at": "2026-06-30 10:00:00",
            },
        )
        exp_id = conn.execute(text("SELECT last_insert_rowid()")).scalar()
        conn.execute(
            text(
                "INSERT INTO run (experiment_id, run_type, status, start_time) "
                "VALUES (:experiment_id, :run_type, :status, :start_time)"
            ),
            {
                "experiment_id": exp_id,
                "run_type": "train",
                "status": "completed",
                "start_time": "2026-06-30 10:00:00",
            },
        )
        run_id = conn.execute(text("SELECT last_insert_rowid()")).scalar()
        alerts = [
            {
                "run_id": run_id,
                "severity": "critical",
                "category": "stress",
                "message": f"alert {i}",
                "recorded_at": "2026-06-30 10:00:00",
                "resolved": False,
            }
            for i in range(10_000)
        ]
        conn.execute(
            text(
                "INSERT INTO alert (run_id, severity, category, message, recorded_at, resolved) "
                "VALUES (:run_id, :severity, :category, :message, :recorded_at, :resolved)"
            ),
            alerts,
        )
        conn.commit()

    TimelineRefresher(str(db_path)).refresh_sync()

    rows, total = get_timeline(TimelineFilters(), engine)
    assert total == 1
    assert rows[0].critical_alerts == 10_000


@pytest.mark.stress
@pytest.mark.timeout(15)
def test_timeline_inverted_date_filter() -> None:
    """T3.4: date_to anterior a date_from deve ser rejeitado pelo Pydantic."""
    with pytest.raises(ValidationError):
        TimelineFilters(
            date_from=datetime(2026, 6, 30, 12, 0, 0, tzinfo=timezone.utc),
            date_to=datetime(2026, 6, 30, 10, 0, 0, tzinfo=timezone.utc),
        )


@pytest.mark.stress
@pytest.mark.security
@pytest.mark.timeout(15)
def test_timeline_invalid_sort_column(tmp_path: Path) -> None:
    """T3.5: Ordenação por coluna inexistente/injetada deve ser rejeitada."""
    db_path = tmp_path / "lewis_sort.db"
    engine = get_engine(db_path)
    init_schema(engine)

    with pytest.raises(ValueError):
        invalid_sort = "injected_column; DROP TABLE run; --"  # type: ignore[assignment]
        get_timeline(
            TimelineFilters(sort_by=invalid_sort),  # type: ignore[arg-type]
            engine,
        )


@pytest.mark.stress
@pytest.mark.timeout(15)
def test_timeline_cascade_delete_consistency(tmp_path: Path) -> None:
    """T3.6: Deletar run deve refletir na materializada após refresh."""
    db_path = tmp_path / "lewis_cascade.db"
    engine = get_engine(db_path)
    init_schema(engine)

    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO experiment (name, stage, status, created_at) "
                "VALUES (:name, :stage, :status, :created_at)"
            ),
            {
                "name": "exp_cascade",
                "stage": "stage1",
                "status": "completed",
                "created_at": "2026-06-30 10:00:00",
            },
        )
        exp_id = conn.execute(text("SELECT last_insert_rowid()")).scalar()
        conn.execute(
            text(
                "INSERT INTO run (experiment_id, run_type, status, start_time) "
                "VALUES (:experiment_id, :run_type, :status, :start_time)"
            ),
            {
                "experiment_id": exp_id,
                "run_type": "train",
                "status": "completed",
                "start_time": "2026-06-30 10:00:00",
            },
        )
        run_id = conn.execute(text("SELECT last_insert_rowid()")).scalar()
        conn.execute(
            text(
                "INSERT INTO metric (run_id, namespace, name, value, recorded_at) "
                "VALUES (:run_id, 'global', 'accuracy', 0.95, :recorded_at)"
            ),
            {"run_id": run_id, "recorded_at": "2026-06-30 10:00:00"},
        )
        conn.commit()

    TimelineRefresher(str(db_path)).refresh_sync()

    rows_before, _ = get_timeline(TimelineFilters(), engine)
    assert len(rows_before) == 1

    with engine.connect() as conn:
        conn.execute(text("DELETE FROM run WHERE id = :run_id"), {"run_id": run_id})
        conn.commit()

    TimelineRefresher(str(db_path)).refresh_sync()

    rows_after, _ = get_timeline(TimelineFilters(), engine)
    assert len(rows_after) == 0


@pytest.mark.stress
@pytest.mark.timeout(15)
def test_timeline_compare_two_experiments(tmp_path: Path) -> None:
    """T3.7: Comparar duas runs via API de timeline e calcular delta."""
    db_path = tmp_path / "lewis_compare.db"
    engine = get_engine(db_path)
    init_schema(engine)

    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO experiment (name, stage, status, created_at) "
                "VALUES (:name, :stage, :status, :created_at)"
            ),
            {
                "name": "exp_compare",
                "stage": "stage1",
                "status": "completed",
                "created_at": "2026-06-30 10:00:00",
            },
        )
        exp_id = conn.execute(text("SELECT last_insert_rowid()")).scalar()
        for rid, acc in [("run_a", 0.95), ("run_b", 0.85)]:
            conn.execute(
                text(
                    "INSERT INTO run (experiment_id, run_type, status, start_time) "
                    "VALUES (:experiment_id, :run_type, :status, :start_time)"
                ),
                {
                    "experiment_id": exp_id,
                    "run_type": "train",
                    "status": "completed",
                    "start_time": "2026-06-30 10:00:00",
                },
            )
            run_id = conn.execute(text("SELECT last_insert_rowid()")).scalar()
            conn.execute(
                text(
                    "INSERT INTO metric (run_id, namespace, name, value, recorded_at) "
                    "VALUES (:run_id, 'global', 'accuracy', :acc, :recorded_at)"
                ),
                {"run_id": run_id, "acc": acc, "recorded_at": "2026-06-30 10:00:00"},
            )
        conn.commit()

    TimelineRefresher(str(db_path)).refresh_sync()

    rows, _ = get_timeline(TimelineFilters(sort_by="final_accuracy", sort_desc=True), engine)
    assert len(rows) == 2
    assert rows[0].final_accuracy is not None
    assert rows[1].final_accuracy is not None
    assert rows[0].final_accuracy == pytest.approx(0.95)
    assert rows[1].final_accuracy == pytest.approx(0.85)
    assert rows[0].final_accuracy - rows[1].final_accuracy == pytest.approx(0.10, abs=0.01)


@pytest.mark.stress
@pytest.mark.timeout(180)
def test_timeline_performance_1m_records(tmp_path: Path) -> None:
    """T3.8: 1M runs com métricas e alertas; query na materializada < 3s."""
    db_path = tmp_path / "lewis_1m.db"
    _seed_timeline_db(db_path, 1_000_000)

    refresher = TimelineRefresher(str(db_path))
    refresher.refresh_sync()

    engine = get_engine(db_path)
    start = time.perf_counter()
    rows, total = get_timeline(TimelineFilters(limit=1000), engine)
    elapsed = time.perf_counter() - start

    assert elapsed < 3.0, f"Query demorou {elapsed:.2f}s"
    assert len(rows) == 1000
    assert total == 1_000_000
