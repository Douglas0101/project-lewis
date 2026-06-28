"""Geradores de relatórios a partir do banco de tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.tracking.models import Experiment, Metric, Run
from src.tracking.repositories import AlertRepository, ExperimentRepository, MetricRepository
from src.tracking.schemas import ExperimentSummary


def _best_metric_for_experiment(
    session: Session, experiment_id: int
) -> tuple[Optional[str], Optional[float]]:
    """Retorna (nome, valor) da melhor métrica global F1_macro do experimento."""
    metric = MetricRepository(session).get_best(experiment_id, "F1_macro", maximize=True)
    if metric:
        return metric.name, metric.value
    return None, None


def summarize_experiment(session: Session, experiment: Experiment) -> ExperimentSummary:
    """Cria resumo agregado de um experimento."""
    best_name, best_value = _best_metric_for_experiment(session, experiment.id)
    n_runs = len(experiment.runs)
    n_alerts = AlertRepository(session).count_unresolved(experiment.id)
    return ExperimentSummary(
        experiment=experiment,
        best_metric=best_name,
        best_value=best_value,
        n_runs=n_runs,
        n_alerts=n_alerts,
    )


def list_summaries(
    session: Session,
    stage: Optional[str] = None,
    limit: int = 50,
) -> List[ExperimentSummary]:
    """Lista resumos de experimentos."""
    experiments = ExperimentRepository(session).list(stage=stage, limit=limit)
    return [summarize_experiment(session, exp) for exp in experiments]


def experiment_markdown(session: Session, experiment_id: int) -> str:
    """Gera relatório Markdown detalhado de um experimento."""
    exp = ExperimentRepository(session).get_with_runs(experiment_id)
    if not exp:
        return "# Experimento não encontrado\n"

    lines: List[str] = [f"# Experimento: {exp.name}\n"]
    lines.append(f"- **ID:** {exp.id}\n")
    lines.append(f"- **Stage:** {exp.stage}\n")
    lines.append(f"- **Status:** {exp.status}\n")
    lines.append(f"- **Criado em:** {exp.created_at.isoformat()}\n")
    lines.append(f"- **Config:** {exp.config_path or 'N/A'}\n")
    lines.append(f"- **Git commit:** {exp.git_commit or 'N/A'}\n")
    if exp.description:
        lines.append(f"- **Descrição:** {exp.description}\n")
    lines.append(f"\n## Runs ({len(exp.runs)})\n")
    lines.append("| ID | Tipo | Fold | Início | Status |\n")
    lines.append("|----|------|------|--------|--------|\n")
    for run in sorted(exp.runs, key=lambda r: r.start_time):
        fold = run.fold_idx if run.fold_idx is not None else "-"
        lines.append(
            f"| {run.id} | {run.run_type} | {fold} | "
            f"{run.start_time.isoformat()} | {run.status} |\n"
        )

    metrics = (
        session.query(Metric)
        .join(Run)
        .filter(Run.experiment_id == experiment_id, Metric.namespace == "global")
        .order_by(Metric.name)
        .all()
    )
    if metrics:
        lines.append("\n## Métricas globais\n")
        lines.append("| Run | Métrica | Valor |\n")
        lines.append("|-----|---------|-------|\n")
        for m in metrics:
            lines.append(f"| {m.run_id} | {m.name} | {m.value:.4f} |\n")

    alerts = AlertRepository(session).list(experiment_id=experiment_id, limit=100)
    if alerts:
        lines.append(f"\n## Alertas ({len(alerts)})\n")
        lines.append("| ID | Severidade | Categoria | Mensagem | Resolvido |\n")
        lines.append("|----|------------|-----------|----------|-----------|\n")
        for alert in alerts:
            resolved = "Sim" if alert.resolved else "Não"
            lines.append(
                f"| {alert.id} | {alert.severity} | {alert.category} | "
                f"{alert.message} | {resolved} |\n"
            )

    return "".join(lines)


def experiment_json(session: Session, experiment_id: int) -> Dict[str, Any]:
    """Exporta experimento, runs, métricas e alertas como dict."""
    exp = ExperimentRepository(session).get_with_runs(experiment_id)
    if not exp:
        return {}

    runs_data: List[Dict[str, Any]] = []
    for run in exp.runs:
        metrics = [
            {"name": m.name, "namespace": m.namespace, "value": m.value}
            for m in MetricRepository(session).list_by_run(run.id)
        ]
        runs_data.append(
            {
                "id": run.id,
                "run_type": run.run_type,
                "fold_idx": run.fold_idx,
                "status": run.status,
                "start_time": run.start_time.isoformat() if run.start_time else None,
                "end_time": run.end_time.isoformat() if run.end_time else None,
                "metrics": metrics,
            }
        )

    alerts = [
        {
            "id": a.id,
            "severity": a.severity,
            "category": a.category,
            "message": a.message,
            "resolved": a.resolved,
        }
        for a in AlertRepository(session).list(experiment_id=experiment_id, limit=1000)
    ]

    return {
        "experiment": {
            "id": exp.id,
            "name": exp.name,
            "stage": exp.stage,
            "status": exp.status,
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "config_path": exp.config_path,
            "git_commit": exp.git_commit,
            "description": exp.description,
        },
        "runs": runs_data,
        "alerts": alerts,
    }


def save_experiment_report(
    session: Session,
    experiment_id: int,
    output_dir: Path,
    fmt: str = "markdown",
) -> Path:
    """Salva relatório de experimento em disco.

    Raises
    ------
    ValueError
        Se o experimento não existir.
    """
    if ExperimentRepository(session).get(experiment_id) is None:
        raise ValueError(f"Experimento {experiment_id} não encontrado")

    output_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path = output_dir / f"experiment_{experiment_id}.json"
        path.write_text(
            json.dumps(experiment_json(session, experiment_id), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        path = output_dir / f"experiment_{experiment_id}.md"
        path.write_text(experiment_markdown(session, experiment_id), encoding="utf-8")
    return path
