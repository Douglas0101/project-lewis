"""API mínima de observabilidade: /health e /metrics."""

from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from src.observability import metrics  # noqa: F401
from src.tracking.db import get_db_path

app = FastAPI(title="Project-Lewis Observability")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health")
def health():
    return {"status": "ok", "db_path": str(get_db_path())}
