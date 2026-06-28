"""Modelos SQLAlchemy para o banco de tracking.

Tabelas:
- experiment: agrupa runs de um treinamento/avaliação.
- run: instância de execução (fold, avaliação, inferência).
- metric: valor numérico de métrica (global ou por classe).
- prediction: predição individual (opcional, para auditoria).
- alert: quedas, falhas de QG e anomalias.
- hardware_snapshot: uso de recursos computacionais.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarativa do SQLAlchemy 2.0."""


class Experiment(Base):
    """Agrupa runs de um experimento de treinamento/avaliação."""

    __tablename__ = "experiment"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    stage: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="pretrain, stage1, stage2, finetune, two_stage, inference",
    )
    config_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    git_commit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    status: Mapped[str] = mapped_column(
        String(16), default="running", comment="running, completed, failed"
    )
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    runs: Mapped[list["Run"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


class Run(Base):
    """Uma execução dentro de um experimento."""

    __tablename__ = "run"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiment.id", ondelete="CASCADE"), nullable=False
    )
    fold_idx: Mapped[int | None] = mapped_column(Integer, nullable=True)
    run_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="train, val, test, inference",
    )
    artifact_dir: Mapped[str | None] = mapped_column(String(512), nullable=True)
    start_time: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    end_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), default="running", comment="running, completed, failed"
    )
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    experiment: Mapped["Experiment"] = relationship(back_populates="runs")
    metrics: Mapped[list["Metric"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    alerts: Mapped[list["Alert"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    hardware_snapshots: Mapped[list["HardwareSnapshot"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Metric(Base):
    """Métrica numérica vinculada a uma run."""

    __tablename__ = "metric"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("run.id", ondelete="CASCADE"), nullable=False
    )
    namespace: Mapped[str] = mapped_column(
        String(32), default="global", comment="global, per_class, history"
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    class_name: Mapped[str | None] = mapped_column(String(8), nullable=True)
    step: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="epoch")
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    run: Mapped["Run"] = relationship(back_populates="metrics")


class Prediction(Base):
    """Predição individual para auditoria (uso controlado por amostragem)."""

    __tablename__ = "prediction"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("run.id", ondelete="CASCADE"), nullable=False
    )
    sample_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    true_label: Mapped[int | None] = mapped_column(Integer, nullable=True)
    predicted_label: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    stage: Mapped[str] = mapped_column(
        String(32), default="integrated", comment="stage1, stage2, integrated"
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    run: Mapped["Run"] = relationship(back_populates="predictions")


class Alert(Base):
    """Alerta de queda de performance, falha de QG ou anomalia."""

    __tablename__ = "alert"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int | None] = mapped_column(
        ForeignKey("run.id", ondelete="CASCADE"), nullable=True
    )
    experiment_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiment.id", ondelete="CASCADE"), nullable=True
    )
    severity: Mapped[str] = mapped_column(
        String(16), default="warning", comment="info, warning, critical"
    )
    category: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="performance_drop, qg_failure, resource_fault, anomaly",
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    metric_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    run: Mapped["Run | None"] = relationship(back_populates="alerts")


class HardwareSnapshot(Base):
    """Snapshot de uso de recursos durante uma run."""

    __tablename__ = "hardware_snapshot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("run.id", ondelete="CASCADE"), nullable=False
    )
    cpu_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    ram_used_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    gpu_utilization_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    gpu_memory_used_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    run: Mapped["Run"] = relationship(back_populates="hardware_snapshots")
