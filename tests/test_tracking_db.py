"""Testes unitários do banco de tracking."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.tracking.db import init_schema
from src.tracking.repositories import (
    AlertRepository,
    ExperimentRepository,
    MetricRepository,
    RunRepository,
)
from src.tracking.schemas import (
    AlertCreate,
    ExperimentCreate,
    MetricCreate,
    RunCreate,
)


@pytest.fixture
def session() -> Session:
    """Sessão em banco SQLite in-memory."""
    engine = create_engine("sqlite:///:memory:", future=True)
    init_schema(engine)
    with Session(engine) as s:
        yield s
        s.rollback()


def test_create_experiment(session: Session) -> None:
    repo = ExperimentRepository(session)
    exp = repo.create(
        ExperimentCreate(name="exp_test", stage="stage1", config_path="config.yaml")
    )
    assert exp.id is not None
    assert exp.name == "exp_test"
    assert exp.stage == "stage1"


def test_list_experiments_filter_stage(session: Session) -> None:
    repo = ExperimentRepository(session)
    repo.create(ExperimentCreate(name="e1", stage="stage1"))
    repo.create(ExperimentCreate(name="e2", stage="stage2"))
    stage1 = repo.list(stage="stage1")
    assert len(stage1) == 1
    assert stage1[0].stage == "stage1"


def test_create_run_and_metrics(session: Session) -> None:
    exp_repo = ExperimentRepository(session)
    exp = exp_repo.create(ExperimentCreate(name="exp", stage="finetune"))

    run_repo = RunRepository(session)
    run = run_repo.create(
        RunCreate(experiment_id=exp.id, run_type="test", fold_idx=2)
    )
    assert run.experiment_id == exp.id
    assert run.fold_idx == 2

    metric_repo = MetricRepository(session)
    metrics = metric_repo.create_many(
        [
            MetricCreate(run_id=run.id, name="F1_macro", value=0.85),
            MetricCreate(run_id=run.id, name="Acc", value=0.92, namespace="global"),
        ]
    )
    assert len(metrics) == 2
    assert metrics[0].value == pytest.approx(0.85)


def test_get_best_metric(session: Session) -> None:
    exp_repo = ExperimentRepository(session)
    exp = exp_repo.create(ExperimentCreate(name="exp", stage="stage1"))
    run_repo = RunRepository(session)
    run = run_repo.create(RunCreate(experiment_id=exp.id, run_type="test"))

    MetricRepository(session).create_many(
        [
            MetricCreate(run_id=run.id, name="F1_macro", value=0.75),
            MetricCreate(run_id=run.id, name="F1_macro", value=0.91),
        ]
    )

    best = MetricRepository(session).get_best(exp.id, "F1_macro", maximize=True)
    assert best is not None
    assert best.value == pytest.approx(0.91)


def test_run_complete(session: Session) -> None:
    exp_repo = ExperimentRepository(session)
    exp = exp_repo.create(ExperimentCreate(name="exp", stage="stage1"))
    run_repo = RunRepository(session)
    run = run_repo.create(RunCreate(experiment_id=exp.id, run_type="test"))

    updated = run_repo.complete(run.id)
    assert updated is not None
    assert updated.status == "completed"
    assert updated.end_time is not None


def test_alert_create_and_resolve(session: Session) -> None:
    exp_repo = ExperimentRepository(session)
    exp = exp_repo.create(ExperimentCreate(name="exp", stage="stage1"))

    alert_repo = AlertRepository(session)
    alert = alert_repo.create(
        AlertCreate(
            experiment_id=exp.id,
            severity="warning",
            category="qg_failure",
            message="QG5 não satisfeito",
        )
    )
    assert alert.id is not None
    assert alert.resolved is False

    unresolved = alert_repo.count_unresolved(exp.id)
    assert unresolved == 1

    resolved = alert_repo.resolve(alert.id)
    assert resolved is not None
    assert resolved.resolved is True
    assert alert_repo.count_unresolved(exp.id) == 0


def test_cascade_delete_experiment_removes_runs(session: Session) -> None:
    exp_repo = ExperimentRepository(session)
    exp = exp_repo.create(ExperimentCreate(name="exp", stage="stage1"))
    run_repo = RunRepository(session)
    run = run_repo.create(RunCreate(experiment_id=exp.id, run_type="test"))

    exp_repo.delete(exp.id)
    assert run_repo.get(run.id) is None
