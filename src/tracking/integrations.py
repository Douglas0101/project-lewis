"""Integração entre tracking e pipelines de treinamento/avaliação.

Fornece funções de alto nível para registrar experimentos de GroupKFold,
avaliações AAMI e alertas sem poluir os scripts de negócio.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, cast

from sqlalchemy.orm import Session

from src.tracking._git import git_commit_short
from src.tracking.db import get_session, init_schema
from src.tracking.repositories import (
    AlertRepository,
    ExperimentRepository,
    MetricRepository,
    RunRepository,
    record_alert_on_qg_failure,
    record_evaluation_metrics,
)
from src.tracking.schemas import (
    AlertCreate,
    ExperimentCreate,
    ExperimentStage,
    ExperimentUpdate,
    MetricCreate,
    RunCreate,
    RunType,
    Status,
)

LOGGER = logging.getLogger("lewis.tracking.integrations")


@contextmanager
def _managed_session(session: Optional[Session] = None) -> Generator[Session, None, None]:
    """Fornece sessão existente ou cria/fecha nova."""
    if session is not None:
        yield session
        return
    new_session = get_session()
    try:
        yield new_session
        new_session.commit()
    except Exception:
        new_session.rollback()
        raise
    finally:
        new_session.close()


def ensure_schema() -> None:
    """Garante que o banco e tabelas existem."""
    init_schema()


def start_tracking_experiment(
    name: str,
    stage: str,
    config_path: Optional[Path] = None,
    description: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    session: Optional[Session] = None,
) -> int:
    """Inicia experimento e retorna ID.

    Chamado pelos scripts de treinamento antes do GroupKFold.
    """
    if session is None:
        ensure_schema()
    with _managed_session(session) as sess:
        git_commit = git_commit_short()

        exp = ExperimentRepository(sess).create(
            ExperimentCreate(
                name=name,
                stage=cast(ExperimentStage, stage),
                config_path=str(config_path) if config_path else None,
                git_commit=git_commit,
                description=description,
                extra=extra or {},
            )
        )
        LOGGER.info("Tracking experimento id=%d | %s | stage=%s", exp.id, name, stage)
        return int(exp.id)


def finish_tracking_experiment(
    experiment_id: int,
    status: str = "completed",
    session: Optional[Session] = None,
) -> None:
    """Finaliza experimento."""
    with _managed_session(session) as sess:
        ExperimentRepository(sess).update(
            experiment_id,
            ExperimentUpdate(status=cast(Status, status)),
        )


def start_tracking_run(
    experiment_id: int,
    run_type: str,
    fold_idx: Optional[int] = None,
    artifact_dir: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
    session: Optional[Session] = None,
) -> int:
    """Inicia run e retorna ID."""
    with _managed_session(session) as sess:
        run = RunRepository(sess).create(
            RunCreate(
                experiment_id=experiment_id,
                run_type=cast(RunType, run_type),
                fold_idx=fold_idx,
                artifact_dir=str(artifact_dir) if artifact_dir else None,
                extra=extra or {},
            )
        )
        return int(run.id)


def finish_tracking_run(
    run_id: int,
    status: str = "completed",
    eval_result: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[int] = None,
    stage_label: str = "",
    session: Optional[Session] = None,
) -> None:
    """Finaliza run, registra métricas e alertas de QG."""
    with _managed_session(session) as sess:
        RunRepository(sess).complete(run_id, status)
        if eval_result:
            record_evaluation_metrics(sess, run_id, eval_result)
            if experiment_id:
                record_alert_on_qg_failure(
                    sess, experiment_id, run_id, eval_result, stage_label
                )


def record_fold_results(
    experiment_id: int,
    fold_idx: int,
    eval_result: Dict[str, Any],
    artifact_dir: Optional[Path] = None,
    stage_label: str = "",
    session: Optional[Session] = None,
) -> int:
    """Registra uma run de fold completa com métricas e alertas.

    Parameters
    ----------
    experiment_id : int
    fold_idx : int
    eval_result : dict
        Resultado de ``evaluate_aami``.
    artifact_dir : Path, optional
    stage_label : str
        Prefixo para alertas (ex.: "stage1", "stage2").
    session : Session, optional

    Returns
    -------
    int
        ID da run criada.
    """
    with _managed_session(session) as sess:
        run_id = start_tracking_run(
            experiment_id=experiment_id,
            run_type="test",
            fold_idx=fold_idx,
            artifact_dir=artifact_dir,
            session=sess,
        )
        finish_tracking_run(
            run_id=run_id,
            status="completed",
            eval_result=eval_result,
            experiment_id=experiment_id,
            stage_label=stage_label,
            session=sess,
        )
        return run_id


def record_summary_metrics(
    experiment_id: int,
    summary: Dict[str, Any],
    stage_label: str = "",
    session: Optional[Session] = None,
) -> None:
    """Registra métricas agregadas de um experimento como run de tipo ``summary``."""
    with _managed_session(session) as sess:
        run_id = start_tracking_run(
            experiment_id=experiment_id,
            run_type="summary",
            extra={
                "best_fold": summary.get("best_fold"),
                "passes_qg5": summary.get("passes_qg5"),
            },
            session=sess,
        )

        metrics: List[MetricCreate] = []
        mean_metrics = summary.get("mean_metrics", {})
        for name, value in mean_metrics.items():
            if isinstance(value, (int, float)):
                metrics.append(
                    MetricCreate(
                        run_id=run_id,
                        namespace="global",
                        name=f"{stage_label}_mean_{name}".strip("_"),
                        value=float(value),
                    )
                )

        std_metrics = summary.get("std_metrics", {})
        for name, value in std_metrics.items():
            if isinstance(value, (int, float)):
                metrics.append(
                    MetricCreate(
                        run_id=run_id,
                        namespace="global",
                        name=f"{stage_label}_std_{name}".strip("_"),
                        value=float(value),
                    )
                )

        if metrics:
            MetricRepository(sess).create_many(metrics)
            if not summary.get("passes_qg5"):
                AlertRepository(sess).create(
                    AlertCreate(
                        run_id=run_id,
                        experiment_id=experiment_id,
                        severity="warning",
                        category="qg_failure",
                        message=f"{stage_label} mean QG5 não satisfeito".strip(),
                    )
                )

        finish_tracking_run(run_id=run_id, status="completed", session=sess)


def record_two_stage_results(
    experiment_id: int,
    results: Dict[str, Dict[str, Any]],
    session: Optional[Session] = None,
) -> None:
    """Registra resultados do pipeline two-stage (stage1, stage2, integrated)."""
    with _managed_session(session) as sess:
        for stage_name in ("stage1", "stage2", "integrated"):
            eval_result = results.get(stage_name)
            if not eval_result:
                continue
            run_id = start_tracking_run(
                experiment_id=experiment_id,
                run_type="test",
                extra={"stage": stage_name},
                session=sess,
            )
            finish_tracking_run(
                run_id=run_id,
                status="completed",
                eval_result=eval_result,
                experiment_id=experiment_id,
                stage_label=stage_name,
                session=sess,
            )
