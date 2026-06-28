"""Pydantic schemas para validação e serialização do tracking.

Mantém separação entre ORM (models.py) e contratos de API/CLI.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator

ExperimentStage = Literal[
    "pretrain", "stage1", "stage2", "finetune", "two_stage", "inference"
]
RunType = Literal["train", "val", "test", "summary", "inference"]
Severity = Literal["info", "warning", "critical"]
AlertCategory = Literal[
    "performance_drop", "qg_failure", "resource_fault", "anomaly"
]
Status = Literal["running", "completed", "failed"]


class _Base(BaseModel):
    """Base com configuração comum."""

    model_config = {"from_attributes": True}


class ExperimentCreate(_Base):
    """Dados para criação de experimento."""

    name: str = Field(..., min_length=1, max_length=255)
    stage: ExperimentStage
    config_path: Optional[str] = Field(None, max_length=512)
    git_commit: Optional[str] = Field(None, max_length=40)
    description: Optional[str] = None
    status: Status = "running"
    extra: Optional[Dict[str, Any]] = None


class ExperimentUpdate(_Base):
    """Atualização parcial de experimento."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    status: Optional[Status] = None
    description: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class ExperimentOut(_Base):
    """Representação de experimento."""

    id: int
    name: str
    stage: str
    config_path: Optional[str]
    git_commit: Optional[str]
    description: Optional[str]
    created_at: datetime
    status: str
    extra: Optional[Dict[str, Any]]


class RunCreate(_Base):
    """Dados para criação de run."""

    experiment_id: int
    run_type: RunType
    fold_idx: Optional[int] = None
    artifact_dir: Optional[str] = Field(default=None, max_length=512)
    status: Status = "running"
    extra: Optional[Dict[str, Any]] = None


class RunUpdate(_Base):
    """Atualização parcial de run."""

    status: Optional[Status] = None
    end_time: Optional[datetime] = None
    artifact_dir: Optional[str] = Field(default=None, max_length=512)
    extra: Optional[Dict[str, Any]] = None


class RunOut(_Base):
    """Representação de run."""

    id: int
    experiment_id: int
    fold_idx: Optional[int]
    run_type: str
    artifact_dir: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    status: str
    extra: Optional[Dict[str, Any]]


class MetricCreate(_Base):
    """Dados para criação de métrica."""

    run_id: int
    name: str = Field(..., min_length=1, max_length=64)
    value: float
    namespace: str = Field(default="global", max_length=32)
    class_name: Optional[str] = Field(default=None, max_length=8)
    step: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None

    @field_validator("namespace")
    @classmethod
    def _namespace_ok(cls, v: str) -> str:
        if v not in {"global", "per_class", "history"}:
            raise ValueError("namespace deve ser global, per_class ou history")
        return v


class MetricOut(_Base):
    """Representação de métrica."""

    id: int
    run_id: int
    namespace: str
    name: str
    value: float
    class_name: Optional[str]
    step: Optional[int]
    recorded_at: datetime
    extra: Optional[Dict[str, Any]]


class PredictionCreate(_Base):
    """Dados para criação de predição."""

    run_id: int
    predicted_label: int
    sample_id: Optional[str] = Field(default=None, max_length=255)
    true_label: Optional[int] = None
    confidence: Optional[float] = None
    stage: str = "integrated"
    extra: Optional[Dict[str, Any]] = None


class PredictionOut(_Base):
    """Representação de predição."""

    id: int
    run_id: int
    sample_id: Optional[str]
    true_label: Optional[int]
    predicted_label: int
    confidence: Optional[float]
    stage: str
    recorded_at: datetime
    extra: Optional[Dict[str, Any]]


class AlertCreate(_Base):
    """Dados para criação de alerta."""

    run_id: Optional[int] = None
    experiment_id: Optional[int] = None
    severity: Severity = "warning"
    category: AlertCategory
    message: str
    metric_name: Optional[str] = Field(default=None, max_length=64)
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None


class AlertUpdate(_Base):
    """Atualização de alerta (resolução)."""

    resolved: bool
    resolved_at: Optional[datetime] = None


class AlertOut(_Base):
    """Representação de alerta."""

    id: int
    run_id: Optional[int]
    experiment_id: Optional[int]
    severity: str
    category: str
    message: str
    metric_name: Optional[str]
    metric_value: Optional[float]
    threshold: Optional[float]
    recorded_at: datetime
    resolved: bool
    resolved_at: Optional[datetime]
    extra: Optional[Dict[str, Any]]


class HardwareSnapshotCreate(_Base):
    """Dados para criação de snapshot de hardware."""

    run_id: int
    cpu_percent: Optional[float] = None
    ram_used_gb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None


class HardwareSnapshotOut(_Base):
    """Representação de snapshot de hardware."""

    id: int
    run_id: int
    cpu_percent: Optional[float]
    ram_used_gb: Optional[float]
    gpu_utilization_percent: Optional[float]
    gpu_memory_used_mb: Optional[float]
    recorded_at: datetime
    extra: Optional[Dict[str, Any]]


class ExperimentSummary(_Base):
    """Resumo agregado de experimento com melhores métricas."""

    experiment: Any
    best_metric: Optional[str] = None
    best_value: Optional[float] = None
    n_runs: int
    n_alerts: int
