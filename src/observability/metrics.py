"""Métricas Prometheus e utilitário de latência para a Camada C11+."""

from __future__ import annotations

import time
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram

REQUEST_LATENCY = Histogram(
    "lewis_request_duration_seconds",
    "Latency por endpoint/ponte",
    ["component", "method"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
)

REQUEST_COUNT = Counter(
    "lewis_requests_total",
    "Total de requisicoes",
    ["component", "status"],
)

ACTIVE_RUNS = Gauge("lewis_active_runs", "Runs em execucao")


class LatencyTracker:
    """Context manager que registra latência e contador de requisição."""

    def __init__(self, component: str, method: str):
        self.component = component
        self.method = method
        self.start: Optional[float] = None

    def __enter__(self) -> "LatencyTracker":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start is None:
            return
        latency = time.perf_counter() - self.start
        REQUEST_LATENCY.labels(component=self.component, method=self.method).observe(latency)
        status = "error" if exc_type else "success"
        REQUEST_COUNT.labels(component=self.component, status=status).inc()
