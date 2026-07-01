"""API de consulta para a linha do tempo consolidada de experimentos.

A view `experiment_timeline` é criada pela migration 001 e une
`experiment`, `run`, `metric` e `alert` em uma única linha por run.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Engine, text

from src.observability.metrics import LatencyTracker
from src.tracking.db import get_engine

logger = logging.getLogger(__name__)

HealthStatus = Literal["HEALTHY", "UNSTABLE", "REGRESSION", "FAILED"]
SortBy = Literal[
    "created_at",
    "run_start",
    "finished_at",
    "duration_seconds",
    "final_accuracy",
    "final_loss",
    "final_f1_macro",
    "final_auc_roc",
    "f1_macro_delta",
    "critical_alerts",
]

ALLOWED_SORT_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "experiment_id",
    "experiment_name",
    "stage",
    "model_name",
    "run_type",
    "status",
    "run_start",
    "finished_at",
    "duration_seconds",
    "final_accuracy",
    "final_loss",
    "final_f1_macro",
    "final_auc_roc",
    "baseline_accuracy",
    "baseline_loss",
    "baseline_f1_macro",
    "baseline_auc_roc",
    "f1_macro_delta",
    "critical_alerts",
    "warning_alerts",
    "info_alerts",
    "health_status",
)


class TimelineFilters(BaseModel):
    """Filtros validados para consulta à timeline."""

    stage: Optional[str] = Field(default=None, max_length=32)
    status: Optional[Literal["running", "completed", "failed"]] = None
    health_status: Optional[HealthStatus] = None
    run_type: Optional[str] = Field(default=None, max_length=32)
    model_name: Optional[str] = Field(default=None, max_length=255)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_f1_macro: Optional[float] = None
    max_critical_alerts: Optional[int] = Field(default=None, ge=0)
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    sort_by: SortBy = "run_start"
    sort_desc: bool = True

    @field_validator("date_to")
    @classmethod
    def _date_order(cls, date_to: Optional[datetime], info: Any) -> Optional[datetime]:
        data = info.data
        date_from = data.get("date_from")
        if date_to is not None and date_from is not None and date_to < date_from:
            raise ValueError("date_to deve ser posterior ou igual a date_from")
        return date_to


class TimelineRow(BaseModel):
    """Uma linha da view experiment_timeline."""

    model_config = {"from_attributes": True}

    run_id: int
    experiment_id: int
    experiment_name: str
    stage: str
    model_name: str
    run_type: str
    status: str
    run_start: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    final_accuracy: Optional[float] = None
    final_loss: Optional[float] = None
    final_f1_macro: Optional[float] = None
    final_auc_roc: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    baseline_loss: Optional[float] = None
    baseline_f1_macro: Optional[float] = None
    baseline_auc_roc: Optional[float] = None
    f1_macro_delta: Optional[float] = None
    critical_alerts: int
    warning_alerts: int
    info_alerts: int
    health_status: str


def _build_where(filters: TimelineFilters) -> Tuple[str, Dict[str, Any]]:
    """Constrói cláusula WHERE parametrizada e dict de parâmetros."""
    conditions: List[str] = []
    params: Dict[str, Any] = {}

    if filters.stage is not None:
        conditions.append("stage = :stage")
        params["stage"] = filters.stage

    if filters.status is not None:
        conditions.append("status = :status")
        params["status"] = filters.status

    if filters.health_status is not None:
        conditions.append("health_status = :health_status")
        params["health_status"] = filters.health_status

    if filters.run_type is not None:
        conditions.append("run_type = :run_type")
        params["run_type"] = filters.run_type

    if filters.model_name is not None:
        conditions.append("model_name = :model_name")
        params["model_name"] = filters.model_name

    if filters.date_from is not None:
        conditions.append("run_start >= :date_from")
        params["date_from"] = filters.date_from.isoformat()

    if filters.date_to is not None:
        conditions.append("run_start <= :date_to")
        params["date_to"] = filters.date_to.isoformat()

    if filters.min_f1_macro is not None:
        conditions.append("final_f1_macro >= :min_f1_macro")
        params["min_f1_macro"] = filters.min_f1_macro

    if filters.max_critical_alerts is not None:
        conditions.append("critical_alerts <= :max_critical_alerts")
        params["max_critical_alerts"] = filters.max_critical_alerts

    where = " AND ".join(conditions)
    return where, params


def get_timeline(
    filters: TimelineFilters,
    engine: Engine | None = None,
) -> Tuple[List[TimelineRow], int]:
    """Consulta a timeline com filtros e retorna linhas + contagem total.

    Parameters
    ----------
    filters : TimelineFilters
        Filtros validados pela Pydantic.
    engine : Engine | None
        Engine SQLAlchemy opcional.

    Returns
    -------
    tuple[list[TimelineRow], int]
        Linhas da página atual e total de registros sem paginação.
    """
    target = engine or get_engine()
    where, params = _build_where(filters)

    sort_column = filters.sort_by
    if sort_column not in ALLOWED_SORT_COLUMNS:
        raise ValueError(f"Coluna de ordenação inválida: {sort_column}")

    direction = "DESC" if filters.sort_desc else "ASC"

    where_clause = f"WHERE {where}" if where else ""

    count_sql = f"SELECT COUNT(*) FROM experiment_timeline {where_clause}"  # nosec B608
    rows_sql = f"""
        SELECT * FROM experiment_timeline
        {where_clause}
        ORDER BY {sort_column} {direction}, run_id DESC
        LIMIT :limit OFFSET :offset
        """  # nosec B608

    params["limit"] = filters.limit
    params["offset"] = filters.offset

    with LatencyTracker("timeline", "get_timeline"):
        with target.connect() as conn:
            total = conn.execute(text(count_sql), params).scalar() or 0
            rows = conn.execute(text(rows_sql), params).mappings().all()

    return [TimelineRow(**dict(row)) for row in rows], int(total)
