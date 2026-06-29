"""Sistema interno de tracking de experimentos, métricas e alertas.

Exporta componentes principais para integração com scripts de treinamento e
avaliação do Project-Lewis.
"""

from __future__ import annotations

from src.tracking.db import get_db_path, get_engine, get_session, init_schema
from src.tracking.decorators import log_experiment, log_run
from src.tracking.repositories import (
    AlertRepository,
    ExperimentRepository,
    HardwareSnapshotRepository,
    MetricRepository,
    PredictionRepository,
    RunRepository,
)

__all__ = [
    "get_db_path",
    "get_engine",
    "get_session",
    "init_schema",
    "log_experiment",
    "log_run",
    "ExperimentRepository",
    "RunRepository",
    "MetricRepository",
    "PredictionRepository",
    "AlertRepository",
    "HardwareSnapshotRepository",
]
