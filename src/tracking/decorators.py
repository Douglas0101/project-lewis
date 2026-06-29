"""Decoradores e helpers para instrumentação não-invasiva.

Fornece ``log_experiment`` e ``log_run`` para envolver funções de treinamento e
avaliação, registrando automaticamente experimentos, runs e alertas.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, Optional, cast

from sqlalchemy.orm import Session

from src.tracking._git import git_commit_short as _git_commit
from src.tracking.db import get_session
from src.tracking.models import Experiment, Run
from src.tracking.repositories import (
    ExperimentRepository,
    RunRepository,
    record_alert_on_qg_failure,
    record_evaluation_metrics,
)
from src.tracking.schemas import (
    ExperimentCreate,
    ExperimentStage,
    ExperimentUpdate,
    RunCreate,
    RunType,
    Status,
)

LOGGER = logging.getLogger("lewis.tracking.decorators")


def start_experiment(
    session: Session,
    name: str,
    stage: str,
    config_path: Optional[str] = None,
    description: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Experiment:
    """Cria experimento e retorna a entidade."""
    repo = ExperimentRepository(session)
    return repo.create(
        ExperimentCreate(
            name=name,
            stage=cast(ExperimentStage, stage),
            config_path=config_path,
            git_commit=_git_commit(),
            description=description,
            extra=extra or {},
        )
    )


def start_run(
    session: Session,
    experiment_id: int,
    run_type: str,
    fold_idx: Optional[int] = None,
    artifact_dir: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Run:
    """Cria run e retorna a entidade."""
    repo = RunRepository(session)
    return repo.create(
        RunCreate(
            experiment_id=experiment_id,
            run_type=cast(RunType, run_type),
            fold_idx=fold_idx,
            artifact_dir=artifact_dir,
            extra=extra or {},
        )
    )


def finish_run(session: Session, run_id: int, status: str = "completed") -> None:
    """Finaliza run com status e timestamp."""
    RunRepository(session).complete(run_id, status)


def finish_experiment(session: Session, experiment_id: int, status: str = "completed") -> None:
    """Finaliza experimento com status."""
    ExperimentRepository(session).update(
        experiment_id,
        ExperimentUpdate(status=cast(Status, status)),
    )


def log_experiment(
    stage: str,
    name: Optional[str] = None,
    config_path: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable:
    """Decorador que cria experimento antes da função e finaliza após.

    A função decorada recebe o kwarg ``experiment_id`` e pode retornar um dict
    com ``status`` e ``extra``.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            session = get_session()
            try:
                exp = start_experiment(
                    session=session,
                    name=name or func.__name__,
                    stage=stage,
                    config_path=config_path,
                    description=description,
                )
                kwargs["experiment_id"] = exp.id
                result = func(*args, **kwargs)
                status = "completed"
                if isinstance(result, dict):
                    status = result.get("status", "completed")
                ExperimentRepository(session).update(
                    exp.id,
                    ExperimentUpdate(status=cast(Status, status)),
                )
                session.commit()
                return result
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        return wrapper

    return decorator


def log_run(
    run_type: str,
    experiment_id_arg: str = "experiment_id",
    fold_idx_arg: Optional[str] = None,
) -> Callable:
    """Decorador que cria run, executa função e finaliza run.

    Útil para envolver ``evaluate_fold`` ou funções de inferência.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            session = get_session()
            try:
                experiment_id = kwargs.get(experiment_id_arg)
                fold_idx = kwargs.get(fold_idx_arg) if fold_idx_arg else None
                run = start_run(
                    session=session,
                    experiment_id=experiment_id,
                    run_type=run_type,
                    fold_idx=fold_idx,
                )
                kwargs["run_id"] = run.id
                result = func(*args, **kwargs)
                status = "completed"
                if isinstance(result, dict):
                    status = result.get("status", "completed")
                RunRepository(session).complete(run.id, status)

                # Se resultado for avaliação AAMI, registra métricas e alertas.
                if isinstance(result, dict) and "global" in result:
                    record_evaluation_metrics(session, run.id, result)
                    record_alert_on_qg_failure(session, experiment_id, run.id, result)

                session.commit()
                return result
            except Exception:
                session.rollback()
                LOGGER.exception("log_run falhou para %s", func.__name__)
                raise
            finally:
                session.close()

        return wrapper

    return decorator
