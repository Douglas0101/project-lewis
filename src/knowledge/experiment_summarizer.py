"""Gera resumos de experimentos a partir do banco de métricas e os persiste como markdown.

A saída é armazenada em ``data/lineage/experiments/<experiment_name>.md`` para
posterior indexação no RAG (Camada C11).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import bleach
from jinja2 import Template
from sqlalchemy.orm import Session, joinedload

from src.tracking.db import get_engine, get_session
from src.tracking.models import Alert, Artifact, Experiment, Metric, Run

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SUMMARIES_DIR = _PROJECT_ROOT / "data" / "lineage" / "experiments"
_TEMPLATE_PATH = _PROJECT_ROOT / "src" / "knowledge" / "templates" / "experiment_summary.md"

_ALLOWED_HTML_TAGS: set[str] = set()
_METRIC_NAMES_OF_INTEREST = {
    "accuracy",
    "loss",
    "f1_macro",
    "F1_macro",
    "auc_roc",
    "precision",
    "recall",
    "passes_qg5",
}


def _load_template() -> Template:
    """Carrega template Jinja2 do resumo."""
    raw = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return Template(raw)


def _get_or_create_experiment(session: Session, experiment_id: int) -> Optional[Experiment]:
    """Busca experimento com relacionamentos úteis."""
    return (
        session.query(Experiment)
        .options(joinedload(Experiment.runs))
        .filter(Experiment.id == experiment_id)
        .first()
    )


def _get_target_run(experiment: Experiment, run_id: Optional[int] = None) -> Optional[Run]:
    """Seleciona a run mais representativa do experimento.

    Preferência: run explícita > run de treino > run de teste > primeira run.
    """
    runs: Sequence[Run] = sorted(
        experiment.runs, key=lambda r: r.start_time or datetime.min.replace(tzinfo=timezone.utc)
    )
    if run_id is not None:
        for run in runs:
            if run.id == run_id:
                return run
    for run in runs:
        if run.run_type == "train":
            return run
    for run in runs:
        if run.run_type == "test":
            return run
    return runs[0] if runs else None


def _metrics_for_run(session: Session, run_id: int) -> Sequence[Metric]:
    """Retorna métricas globais e por classe da run."""
    return (
        session.query(Metric)
        .filter(Metric.run_id == run_id)
        .filter(Metric.namespace.in_(["global", "per_class"]))
        .all()
    )


def _history_metrics(session: Session, run_id: int) -> Sequence[Metric]:
    """Retorna métricas de histórico (épocas) da run."""
    return (
        session.query(Metric)
        .filter(Metric.run_id == run_id)
        .filter(Metric.namespace == "history")
        .order_by(Metric.step)
        .all()
    )


def _alerts_for_run(session: Session, run_id: int) -> Sequence[Alert]:
    """Retorna alertas da run."""
    return session.query(Alert).filter(Alert.run_id == run_id).order_by(Alert.recorded_at).all()


def _artifacts_for_run(session: Session, run_id: int) -> Sequence[Artifact]:
    """Retorna artefatos da run."""
    return (
        session.query(Artifact)
        .filter(Artifact.run_id == run_id)
        .order_by(Artifact.recorded_at)
        .all()
    )


def _extract_baseline(extra: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Extrai baselines do campo JSON extra."""
    extra = extra or {}
    return {
        "accuracy": _to_float(extra.get("baseline_accuracy")),
        "loss": _to_float(extra.get("baseline_loss")),
        "f1_macro": _to_float(extra.get("baseline_f1_macro")),
        "auc_roc": _to_float(extra.get("baseline_auc_roc")),
    }


def _to_float(value: Any) -> Optional[float]:
    """Converte valor para float ou retorna None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_metrics_table(
    metrics: Sequence[Metric], baselines: Dict[str, Optional[float]]
) -> List[Dict[str, Any]]:
    """Constrói tabela de métricas com baselines e deltas."""
    latest: Dict[str, float] = {}
    for metric in metrics:
        name = metric.name.lower()
        if name in {n.lower() for n in _METRIC_NAMES_OF_INTEREST}:
            # Preferir namespace global; manter maior valor se houver duplicatas
            if name not in latest or metric.namespace == "global":
                latest[name] = metric.value

    rows: List[Dict[str, Any]] = []
    for name in ["accuracy", "loss", "f1_macro", "auc_roc", "precision", "recall"]:
        value = latest.get(name)
        baseline = baselines.get(name)
        delta = None
        if value is not None and baseline is not None:
            delta = round(value - baseline, 6)
        rows.append(
            {
                "name": name,
                "value": round(value, 6) if value is not None else "—",
                "baseline": round(baseline, 6) if baseline is not None else "—",
                "delta": round(delta, 6) if delta is not None else "—",
            }
        )
    return rows


def _format_duration(run: Run) -> Optional[float]:
    """Calcula duração em segundos, normalizando timezones."""
    if run.start_time is None or run.end_time is None:
        return None
    start = run.start_time
    end = run.end_time
    if start.tzinfo is None and end.tzinfo is not None:
        start = start.replace(tzinfo=end.tzinfo)
    elif end.tzinfo is None and start.tzinfo is not None:
        end = end.replace(tzinfo=start.tzinfo)
    return round((end - start).total_seconds(), 2)


def _build_recommendations(
    run: Run,
    metrics_table: List[Dict[str, Any]],
    alerts: Sequence[Alert],
    history: Sequence[Metric],
) -> List[str]:
    """Gera recomendações automáticas baseadas nos dados."""
    recs: List[str] = []

    if run.status == "failed":
        recs.append("Investigar causa raiz da falha antes de novo treinamento.")
        if alerts:
            categories = {a.category for a in alerts}
            if "performance_drop" in categories:
                recs.append("Revisar arquitetura/hiperparâmetros: val_F1_macro estagnou.")
            if "qg_failure" in categories:
                recs.append("Ajustar thresholds ou estratégia de treino para atingir QGs.")
        return recs

    # Verificar overfit/underfit via histórico
    if history:
        losses = [m.value for m in history if m.name == "loss" and m.step is not None]
        val_losses = [m.value for m in history if m.name == "val_loss" and m.step is not None]
        if losses and val_losses and len(losses) == len(val_losses):
            gap_start = val_losses[0] - losses[0]
            gap_end = val_losses[-1] - losses[-1]
            if gap_end > gap_start * 1.5:
                recs.append("Sinais de overfit: aumentar regularização ou early stopping.")

    # Recomendações baseadas em baseline
    f1_row = next((r for r in metrics_table if r["name"] == "f1_macro"), None)
    if f1_row and isinstance(f1_row["delta"], float):
        if f1_row["delta"] < 0:
            recs.append("Regressão em F1-macro: revisar dados ou modelo.")
        elif f1_row["delta"] > 0:
            recs.append("Melhoria em F1-macro: considerar este experimento como novo baseline.")

    if not recs:
        recs.append("Nenhuma recomendação crítica. Monitorar próximas runs.")

    return recs


def _build_executive_summary(
    experiment: Experiment,
    run: Run,
    metrics_table: List[Dict[str, Any]],
    alerts: Sequence[Alert],
) -> str:
    """Gera parágrafo de resumo executivo."""
    f1_row = next((r for r in metrics_table if r["name"] == "f1_macro"), None)
    f1_value = f1_row["value"] if f1_row and isinstance(f1_row["value"], float) else None

    parts = [
        f"Experimento `{experiment.name}` no estágio `{experiment.stage}`",
        f"terminou com status `{run.status}`",
    ]
    if f1_value is not None:
        parts.append(f"e F1-macro final {f1_value:.4f}")
    if alerts:
        severities = [a.severity for a in alerts]
        parts.append(f"({len(alerts)} alerta(s), sendo {severities.count('critical')} crítico(s))")
    parts.append(".")

    if run.status == "failed":
        parts.append(" A execução falhou e requer investigação.")
    elif alerts:
        parts.append(" Há alertas pendentes que merecem atenção.")
    else:
        parts.append(" Execução dentro dos parâmetros esperados.")

    return "".join(parts)


def _sanitize_markdown(content: str) -> str:
    """Remove scripts e conteúdo perigoso do markdown antes de indexar."""
    return bleach.clean(content, tags=_ALLOWED_HTML_TAGS, strip=True)


def generate_experiment_summary(
    experiment_id: int,
    run_id: Optional[int] = None,
    output_dir: Path | None = None,
    extra_tags: Optional[List[str]] = None,
) -> Path:
    """Gera markdown de resumo para um experimento/run e persiste em disco.

    Parameters
    ----------
    experiment_id : int
        ID do experimento no banco de tracking.
    run_id : int | None
        ID específico da run. Se None, usa a run de treino ou a mais recente.
    output_dir : Path | None
        Diretório de saída. Padrão: ``data/lineage/experiments``.
    extra_tags : list[str] | None
        Tags adicionais para indexação.

    Returns
    -------
    Path
        Caminho do arquivo markdown gerado.
    """
    output_dir = output_dir or _SUMMARIES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = get_engine()
    with get_session(engine) as session:
        experiment = _get_or_create_experiment(session, experiment_id)
        if experiment is None:
            raise ValueError(f"Experimento {experiment_id} não encontrado")

        run = _get_target_run(experiment, run_id)
        if run is None:
            raise ValueError(f"Experimento {experiment_id} não possui runs")

        metrics = _metrics_for_run(session, run.id)
        history = _history_metrics(session, run.id)
        alerts = _alerts_for_run(session, run.id)
        artifacts = _artifacts_for_run(session, run.id)

    baselines = _extract_baseline(experiment.extra)
    baselines.update(_extract_baseline(run.extra))

    metrics_table = _build_metrics_table(metrics, baselines)
    recommendations = _build_recommendations(run, metrics_table, alerts, history)
    executive_summary = _build_executive_summary(experiment, run, metrics_table, alerts)

    # Determina health_status com base nas mesmas regras da view
    health_status = "HEALTHY"
    if run.status == "failed":
        health_status = "FAILED"
    elif any(a.severity == "critical" for a in alerts):
        health_status = "UNSTABLE"
    else:
        f1_row = next((r for r in metrics_table if r["name"] == "f1_macro"), None)
        if (
            f1_row is not None
            and isinstance(f1_row["value"], float)
            and isinstance(f1_row["delta"], float)
        ):
            if f1_row["delta"] < 0:
                health_status = "REGRESSION"

    tags = {experiment.stage, run.status, run.run_type, health_status}
    tags.update(extra_tags or [])

    template = _load_template()
    rendered = template.render(
        experiment_name=experiment.name,
        experiment_id=experiment.id,
        run_id=run.id,
        stage=experiment.stage,
        status=run.status,
        run_type=run.run_type,
        start_time=run.start_time.isoformat() if run.start_time else "—",
        end_time=run.end_time.isoformat() if run.end_time else "—",
        duration_seconds=_format_duration(run),
        artifact_dir=run.artifact_dir or "—",
        git_commit=experiment.git_commit or "—",
        config_path=experiment.config_path or "—",
        executive_summary=executive_summary,
        metrics_table=metrics_table,
        alerts=alerts,
        artifacts=artifacts,
        recommendations=recommendations,
        health_status=health_status,
        tags=sorted(tags),
    )

    sanitized = _sanitize_markdown(rendered)

    output_path = output_dir / f"{experiment.name}.md"
    output_path.write_text(sanitized, encoding="utf-8")
    logger.info("Resumo do experimento %s salvo em %s", experiment.name, output_path)
    return output_path


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Uso: python -m src.knowledge.experiment_summarizer <experiment_id> [run_id]")
        raise SystemExit(1)

    eid = int(sys.argv[1])
    rid = int(sys.argv[2]) if len(sys.argv) > 2 else None
    path = generate_experiment_summary(eid, rid)
    print(path)
