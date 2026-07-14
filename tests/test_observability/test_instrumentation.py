"""Testes de instrumentação de latência das pontes (P0.3).

Verifica que ``LatencyTracker`` registra métricas nas chamadas de
``retriever.search``, ``structured_query.answer_question``,
``structured_query.execute_custom_sql`` e ``timeline.get_timeline``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any, List, Tuple
from unittest.mock import patch

import numpy as np
import pytest
import sqlite_vec

from src.knowledge import db as db_module
from src.knowledge import retriever as retriever_module
from src.knowledge import structured_query as structured_query_module
from src.knowledge.schemas import QueryRequest
from src.knowledge.structured_query import NaturalQueryRequest, answer_question, execute_custom_sql
from src.tracking.db import get_engine
from src.tracking.db import init_schema as init_tracking_schema
from src.tracking.models import Alert, Experiment, Metric, Run
from src.tracking.timeline import TimelineFilters, get_timeline


class _FakeSentenceTransformer:
    """Modelo de embedding determinístico leve para testes."""

    dim: int = 384

    def encode(
        self,
        sentences: str | List[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings: List[np.ndarray] = []
        for text in sentences:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in re.findall(r"\w+", text.lower()):
                idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dim
                vec[idx] += 1.0
            if normalize_embeddings:
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
            embeddings.append(vec)

        return np.array(embeddings, dtype=np.float32)


@pytest.fixture
def _fake_model() -> _FakeSentenceTransformer:
    return _FakeSentenceTransformer()


@pytest.fixture(autouse=True)
def _isolated_knowledge_log_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redireciona audit trails de knowledge para tmp_path (evita mutar logs do repo)."""
    monkeypatch.setattr(
        retriever_module,
        "LOG_QUERIES",
        tmp_path / "knowledge_queries.jsonl",
    )
    monkeypatch.setattr(
        structured_query_module,
        "_AUDIT_LOG",
        tmp_path / "knowledge_structured_queries.jsonl",
    )


@pytest.fixture
def knowledge_db(tmp_path: Path) -> Generator[Tuple[Any, Path], None, None]:
    """Banco SQLite-vec temporário com schema inicializado."""
    db_path = tmp_path / "knowledge.db"
    conn = db_module.get_connection(db_path)
    db_module.init_schema(conn)
    yield conn, db_path
    conn.close()


@pytest.fixture
def populated_knowledge_db(
    knowledge_db: Tuple[Any, Path],
    _fake_model: _FakeSentenceTransformer,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[Path, None, None]:
    """Banco knowledge populado com um chunk e retriever monkeypatched."""
    conn, db_path = knowledge_db
    source = "docs/Camada-04-Modelagem-v1.1.md"
    content = "O QG5 exige F1-macro maior que 30 por cento."
    emb = _fake_model.encode(f"{source}\n{content}")[0]
    conn.execute(
        """
        INSERT INTO knowledge_chunks
        (chunk_id, source, layer, version, tags, header_1, header_2, content, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "chunk-1",
            source,
            "C04",
            "v1.1",
            json.dumps(["ml"]),
            "",
            "",
            content,
            sqlite_vec.serialize_float32(emb),
        ),
    )
    conn.commit()
    conn.close()

    from src.knowledge import retriever as retriever_module

    monkeypatch.setattr(retriever_module, "_get_model", lambda: _fake_model)
    monkeypatch.setattr(db_module, "KNOWLEDGE_DB", db_path)
    yield db_path


@pytest.fixture
def populated_tracking_db(tmp_path: Path):
    """Banco de tracking com experimento, runs, métricas e alertas."""
    db_path = tmp_path / "tracking_stress.db"
    engine = get_engine(db_path)
    init_tracking_schema(engine)

    from sqlalchemy.orm import Session

    with Session(engine) as session:
        exp = Experiment(
            name="exp_stress",
            stage="stage1",
            status="completed",
            extra={"baseline_f1_macro": 0.55, "baseline_accuracy": 0.8},
        )
        session.add(exp)
        session.flush()

        run_ok = Run(
            experiment_id=exp.id,
            run_type="train",
            status="completed",
        )
        run_fail = Run(
            experiment_id=exp.id,
            run_type="test",
            status="failed",
        )
        session.add_all([run_ok, run_fail])
        session.flush()

        session.add(Metric(run_id=run_ok.id, namespace="global", name="f1_macro", value=0.65))
        session.add(Metric(run_id=run_ok.id, namespace="global", name="accuracy", value=0.9))
        session.add(Metric(run_id=run_fail.id, namespace="global", name="f1_macro", value=0.45))
        session.add(
            Alert(
                run_id=run_fail.id,
                severity="critical",
                category="performance_drop",
                message="F1 abaixo do QG",
            )
        )
        session.commit()

    yield engine, db_path


def test_retriever_instrumented(populated_knowledge_db: Path) -> None:
    from src.knowledge import retriever as retriever_module

    with patch("src.observability.metrics.REQUEST_LATENCY") as mock_hist:
        with patch("src.observability.metrics.REQUEST_COUNT") as mock_cnt:
            retriever_module.search(
                QueryRequest(
                    query="F1-macro QG5", k=1, fetch_k=5, layer=None, version=None, tags=None
                )
            )

    mock_hist.labels.return_value.observe.assert_called()
    mock_cnt.labels.return_value.inc.assert_called()

    labels = mock_hist.labels.call_args
    assert labels.kwargs["component"] == "rag"
    assert labels.kwargs["method"] == "semantic_search"


def test_answer_question_instrumented(populated_tracking_db: Tuple[Any, Path]) -> None:
    engine, db_path = populated_tracking_db
    with patch("src.observability.metrics.REQUEST_LATENCY") as mock_hist:
        with patch("src.observability.metrics.REQUEST_COUNT") as mock_cnt:
            answer_question(NaturalQueryRequest(question="timeline"), db_path=db_path)

    mock_hist.labels.return_value.observe.assert_called()
    mock_cnt.labels.return_value.inc.assert_called()

    labels = mock_hist.labels.call_args
    assert labels.kwargs["component"] == "sql"
    assert labels.kwargs["method"] == "answer_question"


def test_execute_custom_sql_instrumented(populated_tracking_db: Tuple[Any, Path]) -> None:
    engine, db_path = populated_tracking_db
    with patch("src.observability.metrics.REQUEST_LATENCY") as mock_hist:
        with patch("src.observability.metrics.REQUEST_COUNT") as mock_cnt:
            execute_custom_sql("SELECT * FROM run LIMIT 1", db_path=db_path)

    mock_hist.labels.return_value.observe.assert_called()
    mock_cnt.labels.return_value.inc.assert_called()

    labels = mock_hist.labels.call_args
    assert labels.kwargs["component"] == "sql"
    assert labels.kwargs["method"] == "execute_custom_sql"


def test_timeline_instrumented(populated_tracking_db: Tuple[Any, Path]) -> None:
    engine, db_path = populated_tracking_db
    with patch("src.observability.metrics.REQUEST_LATENCY") as mock_hist:
        with patch("src.observability.metrics.REQUEST_COUNT") as mock_cnt:
            get_timeline(TimelineFilters(), engine=engine)

    mock_hist.labels.return_value.observe.assert_called()
    mock_cnt.labels.return_value.inc.assert_called()

    labels = mock_hist.labels.call_args
    assert labels.kwargs["component"] == "timeline"
    assert labels.kwargs["method"] == "get_timeline"
