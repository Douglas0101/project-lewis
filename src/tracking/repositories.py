"""Repositórios CRUD para entidades de tracking.

Cada repositório recebe uma sessão SQLAlchemy e trabalha com schemas pydantic
para entrada/saída, mantendo o ORM isolado.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, cast

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session, joinedload

from src.tracking.models import Alert, Experiment, HardwareSnapshot, Metric, Prediction, Run
from src.tracking.schemas import (
    AlertCreate,
    AlertUpdate,
    ExperimentCreate,
    ExperimentUpdate,
    HardwareSnapshotCreate,
    MetricCreate,
    PredictionCreate,
    RunCreate,
    RunUpdate,
    Status,
)


class ExperimentRepository:
    """CRUD de experimentos."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, data: ExperimentCreate) -> Experiment:
        entity = Experiment(**data.model_dump(exclude_unset=True))
        self.session.add(entity)
        self.session.flush()
        return entity

    def get(self, experiment_id: int) -> Optional[Experiment]:
        return self.session.get(Experiment, experiment_id)

    def get_with_runs(self, experiment_id: int) -> Optional[Experiment]:
        stmt = (
            select(Experiment)
            .where(Experiment.id == experiment_id)
            .options(joinedload(Experiment.runs))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def list(
        self,
        stage: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Experiment]:
        stmt = select(Experiment).order_by(desc(Experiment.created_at))
        if stage:
            stmt = stmt.where(Experiment.stage == stage)
        if status:
            stmt = stmt.where(Experiment.status == status)
        stmt = stmt.limit(limit).offset(offset)
        return self.session.execute(stmt).scalars().all()

    def update(self, experiment_id: int, data: ExperimentUpdate) -> Optional[Experiment]:
        entity = self.get(experiment_id)
        if not entity:
            return None
        for key, value in data.model_dump(exclude_unset=True).items():
            setattr(entity, key, value)
        self.session.flush()
        return entity

    def delete(self, experiment_id: int) -> bool:
        entity = self.get(experiment_id)
        if not entity:
            return False
        self.session.delete(entity)
        self.session.flush()
        return True

    def count(self) -> int:
        return self.session.execute(select(func.count(Experiment.id))).scalar() or 0


class RunRepository:
    """CRUD de runs."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, data: RunCreate) -> Run:
        entity = Run(**data.model_dump(exclude_unset=True))
        self.session.add(entity)
        self.session.flush()
        return entity

    def get(self, run_id: int) -> Optional[Run]:
        return self.session.get(Run, run_id)

    def list_by_experiment(
        self, experiment_id: int, run_type: Optional[str] = None
    ) -> Sequence[Run]:
        stmt = select(Run).where(Run.experiment_id == experiment_id).order_by(Run.start_time)
        if run_type:
            stmt = stmt.where(Run.run_type == run_type)
        return self.session.execute(stmt).scalars().all()

    def update(self, run_id: int, data: RunUpdate) -> Optional[Run]:
        entity = self.get(run_id)
        if not entity:
            return None
        for key, value in data.model_dump(exclude_unset=True).items():
            setattr(entity, key, value)
        self.session.flush()
        return entity

    def complete(self, run_id: int, status: str = "completed") -> Optional[Run]:
        return self.update(
            run_id,
            RunUpdate(status=cast(Status, status), end_time=datetime.now(timezone.utc)),
        )


class MetricRepository:
    """CRUD de métricas com inserção em lote."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, data: MetricCreate) -> Metric:
        entity = Metric(**data.model_dump(exclude_unset=True))
        self.session.add(entity)
        self.session.flush()
        return entity

    def create_many(self, items: Sequence[MetricCreate]) -> List[Metric]:
        entities = [Metric(**item.model_dump(exclude_unset=True)) for item in items]
        self.session.add_all(entities)
        self.session.flush()
        return entities

    def get(self, metric_id: int) -> Optional[Metric]:
        return self.session.get(Metric, metric_id)

    def list_by_run(
        self,
        run_id: int,
        namespace: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Sequence[Metric]:
        stmt = select(Metric).where(Metric.run_id == run_id).order_by(Metric.recorded_at)
        if namespace:
            stmt = stmt.where(Metric.namespace == namespace)
        if name:
            stmt = stmt.where(Metric.name == name)
        return self.session.execute(stmt).scalars().all()

    def get_best(
        self,
        experiment_id: int,
        metric_name: str,
        namespace: str = "global",
        maximize: bool = True,
    ) -> Optional[Metric]:
        stmt = (
            select(Metric)
            .join(Run)
            .where(
                Run.experiment_id == experiment_id,
                Metric.name == metric_name,
                Metric.namespace == namespace,
            )
        )
        if maximize:
            stmt = stmt.order_by(desc(Metric.value))
        else:
            stmt = stmt.order_by(Metric.value)
        stmt = stmt.limit(1)
        return self.session.execute(stmt).scalars().first()


class PredictionRepository:
    """CRUD de predições."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, data: PredictionCreate) -> Prediction:
        entity = Prediction(**data.model_dump(exclude_unset=True))
        self.session.add(entity)
        self.session.flush()
        return entity

    def create_many(self, items: Sequence[PredictionCreate]) -> List[Prediction]:
        entities = [Prediction(**item.model_dump(exclude_unset=True)) for item in items]
        self.session.add_all(entities)
        self.session.flush()
        return entities

    def list_by_run(self, run_id: int, limit: int = 1000) -> Sequence[Prediction]:
        stmt = (
            select(Prediction)
            .where(Prediction.run_id == run_id)
            .order_by(Prediction.recorded_at)
            .limit(limit)
        )
        return self.session.execute(stmt).scalars().all()


class AlertRepository:
    """CRUD de alertas."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, data: AlertCreate) -> Alert:
        entity = Alert(**data.model_dump(exclude_unset=True))
        self.session.add(entity)
        self.session.flush()
        return entity

    def get(self, alert_id: int) -> Optional[Alert]:
        return self.session.get(Alert, alert_id)

    def list(
        self,
        experiment_id: Optional[int] = None,
        run_id: Optional[int] = None,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        unresolved_only: bool = False,
        limit: int = 100,
    ) -> Sequence[Alert]:
        stmt = select(Alert).order_by(desc(Alert.recorded_at))
        if experiment_id:
            stmt = stmt.where(Alert.experiment_id == experiment_id)
        if run_id:
            stmt = stmt.where(Alert.run_id == run_id)
        if severity:
            stmt = stmt.where(Alert.severity == severity)
        if category:
            stmt = stmt.where(Alert.category == category)
        if unresolved_only:
            stmt = stmt.where(Alert.resolved.is_(False))
        stmt = stmt.limit(limit)
        return self.session.execute(stmt).scalars().all()

    def resolve(self, alert_id: int) -> Optional[Alert]:
        entity = self.get(alert_id)
        if not entity:
            return None
        update = AlertUpdate(resolved=True, resolved_at=datetime.now(timezone.utc))
        for key, value in update.model_dump(exclude_unset=True).items():
            setattr(entity, key, value)
        self.session.flush()
        return entity

    def count_unresolved(self, experiment_id: Optional[int] = None) -> int:
        stmt = select(func.count(Alert.id)).where(Alert.resolved.is_(False))
        if experiment_id:
            stmt = stmt.where(Alert.experiment_id == experiment_id)
        return self.session.execute(stmt).scalar() or 0


class HardwareSnapshotRepository:
    """CRUD de snapshots de hardware."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, data: HardwareSnapshotCreate) -> HardwareSnapshot:
        entity = HardwareSnapshot(**data.model_dump(exclude_unset=True))
        self.session.add(entity)
        self.session.flush()
        return entity

    def create_many(self, items: Sequence[HardwareSnapshotCreate]) -> List[HardwareSnapshot]:
        entities = [
            HardwareSnapshot(**item.model_dump(exclude_unset=True)) for item in items
        ]
        self.session.add_all(entities)
        self.session.flush()
        return entities

    def list_by_run(self, run_id: int) -> Sequence[HardwareSnapshot]:
        stmt = (
            select(HardwareSnapshot)
            .where(HardwareSnapshot.run_id == run_id)
            .order_by(HardwareSnapshot.recorded_at)
        )
        return self.session.execute(stmt).scalars().all()


def record_evaluation_metrics(
    session: Session,
    run_id: int,
    eval_result: Dict[str, Any],
    prefix: str = "",
) -> List[Metric]:
    """Converte resultado de ``evaluate_aami`` em métricas no banco.

    Parameters
    ----------
    session : Session
    run_id : int
    eval_result : dict
        Saída de ``evaluate_aami`` com ``global`` e ``per_class``.
    prefix : str
        Prefixo opcional para o nome da métrica (ex.: ``stage1_``).

    Returns
    -------
    list[Metric]
    """
    metrics: List[MetricCreate] = []

    global_metrics = eval_result.get("global", {})
    for name, value in global_metrics.items():
        if isinstance(value, (int, float)):
            metrics.append(
                MetricCreate(
                    run_id=run_id,
                    namespace="global",
                    name=f"{prefix}{name}",
                    value=float(value),
                )
            )

    per_class = eval_result.get("per_class", {})
    for cls_name, cls_metrics in per_class.items():
        for name, value in cls_metrics.items():
            if isinstance(value, (int, float)):
                metrics.append(
                    MetricCreate(
                        run_id=run_id,
                        namespace="per_class",
                        class_name=cls_name,
                        name=f"{prefix}{name}",
                        value=float(value),
                    )
                )

    passes_qg = eval_result.get("passes_qg5")
    if isinstance(passes_qg, bool):
        metrics.append(
            MetricCreate(
                run_id=run_id,
                namespace="global",
                name=f"{prefix}passes_qg5",
                value=float(passes_qg),
            )
        )

    return MetricRepository(session).create_many(metrics)


def record_alert_on_qg_failure(
    session: Session,
    experiment_id: int,
    run_id: int,
    eval_result: Dict[str, Any],
    stage_label: str = "",
) -> Optional[Alert]:
    """Registra alerta se ``passes_qg5`` for False."""
    if eval_result.get("passes_qg5"):
        return None
    label = f"{stage_label} ".strip()
    return AlertRepository(session).create(
        AlertCreate(
            run_id=run_id,
            experiment_id=experiment_id,
            severity="warning",
            category="qg_failure",
            message=f"{label}QG5 não satisfeito",
            metric_name=f"{label}passes_qg5".strip().replace(" ", "_"),
            metric_value=0.0,
        )
    )
