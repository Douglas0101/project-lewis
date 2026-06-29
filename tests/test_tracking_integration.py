"""Testes de integração do tracking com avaliação AAMI."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.models.evaluate import evaluate_aami
from src.tracking.db import init_schema
from src.tracking.integrations import (
    record_evaluation_metrics,
    record_fold_results,
    record_summary_metrics,
    record_two_stage_results,
)
from src.tracking.schemas import RunCreate
from src.tracking.repositories import AlertRepository, MetricRepository, RunRepository


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    init_schema(engine)
    with Session(engine) as s:
        yield s
        s.rollback()


def test_record_evaluation_metrics(session: Session) -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 2])
    result = evaluate_aami(y_true, y_pred, class_names=["N", "S", "V"], verbose=False)

    run_repo = RunRepository(session)
    run = run_repo.create(
        RunCreate(experiment_id=1, run_type="test")
    )
    # Criar experimento manualmente para FK
    from src.tracking.models import Experiment

    exp = Experiment(name="integration", stage="stage1")
    session.add(exp)
    session.flush()
    run.experiment_id = exp.id
    session.flush()

    metrics = record_evaluation_metrics(session, run.id, result)
    assert len(metrics) > 0

    global_metrics = MetricRepository(session).list_by_run(run.id, namespace="global")
    assert any(m.name == "F1_macro" for m in global_metrics)


def test_record_fold_results(session: Session) -> None:
    from src.tracking.models import Experiment

    exp = Experiment(name="integration", stage="stage1")
    session.add(exp)
    session.flush()

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    result = evaluate_aami(y_true, y_pred, class_names=["N", "Anormal"], verbose=False)

    run_id = record_fold_results(
        experiment_id=exp.id,
        fold_idx=0,
        eval_result=result,
        stage_label="stage1",
        session=session,
    )

    run = RunRepository(session).get(run_id)
    assert run is not None
    assert run.fold_idx == 0

    if not result["passes_qg5"]:
        alerts = AlertRepository(session).list(experiment_id=exp.id)
        assert any(a.category == "qg_failure" for a in alerts)


def test_record_summary_metrics(session: Session) -> None:
    from src.tracking.models import Experiment

    exp = Experiment(name="integration", stage="stage1")
    session.add(exp)
    session.flush()

    summary = {
        "best_fold": 2,
        "mean_metrics": {"F1_macro": 0.80, "Acc": 0.90},
        "std_metrics": {"F1_macro": 0.05, "Acc": 0.03},
        "passes_qg5": False,
    }
    record_summary_metrics(
        experiment_id=exp.id,
        summary=summary,
        stage_label="stage1",
        session=session,
    )

    runs = RunRepository(session).list_by_experiment(exp.id, run_type="summary")
    assert len(runs) == 1
    metrics = MetricRepository(session).list_by_run(runs[0].id)
    assert any(m.name == "stage1_mean_F1_macro" for m in metrics)

    alerts = AlertRepository(session).list(experiment_id=exp.id)
    assert any(a.category == "qg_failure" for a in alerts)


def test_record_two_stage_results(session: Session) -> None:
    from src.tracking.models import Experiment

    exp = Experiment(name="integration", stage="two_stage")
    session.add(exp)
    session.flush()

    results = {
        "stage1": evaluate_aami(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
            class_names=["N", "Anormal"],
            verbose=False,
        ),
        "stage2": evaluate_aami(
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            class_names=["S", "V", "F"],
            verbose=False,
        ),
        "integrated": evaluate_aami(
            np.array([0, 1, 2, 3]),
            np.array([0, 1, 2, 3]),
            class_names=["N", "S", "V", "F"],
            verbose=False,
        ),
    }
    record_two_stage_results(experiment_id=exp.id, results=results, session=session)

    runs = RunRepository(session).list_by_experiment(exp.id)
    assert len(runs) == 3
