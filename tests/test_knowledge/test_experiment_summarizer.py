"""Testes do gerador de resumos de experimentos para RAG."""

# pyright: reportArgumentType=false

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from src.knowledge.experiment_summarizer import (
    _build_metrics_table,
    _build_recommendations,
    _extract_baseline,
    _sanitize_markdown,
    generate_experiment_summary,
)
from src.tracking.db import get_engine, init_schema
from src.tracking.models import Alert, Artifact, Experiment, Metric, Run


@pytest.fixture
def summary_db(tmp_path):
    """Banco temporário com experimento, run, métricas e alerta."""
    db_path = tmp_path / "summary_test.db"
    engine = get_engine(db_path)
    init_schema(engine)

    with Session(engine) as session:
        exp = Experiment(
            name="exp_summary",
            stage="stage1",
            status="completed",
            extra={"baseline_f1_macro": 0.55, "baseline_accuracy": 0.8},
            config_path="config/test.yaml",
            git_commit="abc123",
        )
        session.add(exp)
        session.flush()

        run = Run(
            experiment_id=exp.id,
            run_type="train",
            status="failed",
            start_time=datetime(2026, 6, 30, 10, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 6, 30, 10, 5, 0, tzinfo=timezone.utc),
            artifact_dir="experiments/exp_summary",
        )
        session.add(run)
        session.flush()

        session.add(Metric(run_id=run.id, namespace="global", name="f1_macro", value=0.52))
        session.add(Metric(run_id=run.id, namespace="global", name="accuracy", value=0.78))
        session.add(
            Alert(
                run_id=run.id,
                severity="warning",
                category="performance_drop",
                message="F1 abaixo do esperado",
            )
        )
        session.add(
            Artifact(
                run_id=run.id, artifact_type="model", path="models/model.keras", checksum="deadbeef"
            )
        )
        session.commit()

        exp_id = exp.id
        run_id = run.id

    yield engine, exp_id, run_id, db_path


def test_extract_baseline_from_extra():
    """Baseline deve ser extraído de dict extra."""
    extra = {"baseline_f1_macro": "0.6", "baseline_accuracy": 0.8, "foo": "bar"}
    baselines = _extract_baseline(extra)
    assert baselines["f1_macro"] == pytest.approx(0.6)
    assert baselines["accuracy"] == pytest.approx(0.8)
    assert baselines["loss"] is None


def test_build_metrics_table_with_delta():
    """Tabela de métricas deve calcular delta vs baseline."""
    metrics = [
        Metric(run_id=1, namespace="global", name="f1_macro", value=0.6),
        Metric(run_id=1, namespace="global", name="accuracy", value=0.85),
    ]
    baselines = {"f1_macro": 0.55, "accuracy": 0.8}
    table = _build_metrics_table(metrics, baselines)
    f1_row = next(r for r in table if r["name"] == "f1_macro")
    assert f1_row["value"] == 0.6
    assert f1_row["baseline"] == 0.55
    assert f1_row["delta"] == pytest.approx(0.05)


def test_build_recommendations_for_failed_run():
    """Recomendações para run falhada devem incluir investigação."""
    run = Run(status="failed")
    alerts = [Alert(severity="warning", category="performance_drop", message="falhou")]
    recs = _build_recommendations(run, [], alerts, [])
    assert any("Investigar" in r for r in recs)
    assert any("arquitetura" in r.lower() for r in recs)


def test_sanitize_markdown_removes_script():
    """Sanitização deve remover tags script maliciosas."""
    dirty = "Resumo <script>alert('xss')</script> de teste"
    clean = _sanitize_markdown(dirty)
    assert "<script>" not in clean
    assert "</script>" not in clean


def test_generate_experiment_summary_creates_markdown(summary_db, tmp_path):
    """Deve gerar arquivo markdown com metadados de indexação."""
    engine, exp_id, run_id, _ = summary_db
    output_dir = tmp_path / "summaries"

    # Sobrescreve engine padrão injetando caminho via variável de ambiente
    import os

    old_env = os.environ.get("LEWIS_TRACKING_DB")
    os.environ["LEWIS_TRACKING_DB"] = str(tmp_path / "summary_test.db")
    try:
        path = generate_experiment_summary(exp_id, run_id, output_dir=output_dir)
    finally:
        if old_env is None:
            os.environ.pop("LEWIS_TRACKING_DB", None)
        else:
            os.environ["LEWIS_TRACKING_DB"] = old_env

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Experimento: exp_summary" in content
    assert "stage1" in content
    assert "FAILED" in content
    assert "deadbeef" in content
    assert "Metadados para Indexação" in content


def test_generate_experiment_summary_invalid_experiment(tmp_path):
    """Experimento inexistente deve levantar ValueError."""
    import os

    db_path = tmp_path / "empty.db"
    from src.tracking.db import init_schema

    init_schema(get_engine(db_path))

    old_env = os.environ.get("LEWIS_TRACKING_DB")
    os.environ["LEWIS_TRACKING_DB"] = str(db_path)
    try:
        with pytest.raises(ValueError):
            generate_experiment_summary(9999)
    finally:
        if old_env is None:
            os.environ.pop("LEWIS_TRACKING_DB", None)
        else:
            os.environ["LEWIS_TRACKING_DB"] = old_env
