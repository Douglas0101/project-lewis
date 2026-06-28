"""CLI para consulta e exportação do banco de tracking."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tracking.db import get_session, init_schema
from src.tracking.reporting import list_summaries, save_experiment_report
from src.tracking.repositories import AlertRepository, ExperimentRepository


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.tracking.cli",
        description="CLI de tracking de experimentos e métricas do Project-Lewis",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Cria o banco e tabelas")
    init_parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Caminho alternativo para o SQLite",
    )

    list_parser = subparsers.add_parser("list-experiments", help="Lista experimentos")
    list_parser.add_argument("--stage", type=str, default=None)
    list_parser.add_argument("--status", type=str, default=None)
    list_parser.add_argument("--limit", type=int, default=50)

    show_parser = subparsers.add_parser("show-experiment", help="Mostra detalhes")
    show_parser.add_argument("experiment_id", type=int)

    compare_parser = subparsers.add_parser("compare", help="Compara experimentos por uma métrica")
    compare_parser.add_argument("--stage", type=str, default=None)
    compare_parser.add_argument("--metric", type=str, default="F1_macro")
    compare_parser.add_argument("--limit", type=int, default=50)

    export_parser = subparsers.add_parser("export", help="Exporta relatório")
    export_parser.add_argument("--experiment-id", type=int, required=True)
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/tracking"),
    )
    export_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")

    alerts_parser = subparsers.add_parser("alerts", help="Lista alertas")
    alerts_parser.add_argument("--experiment-id", type=int, default=None)
    alerts_parser.add_argument("--severity", type=str, default=None)
    alerts_parser.add_argument("--category", type=str, default=None)
    alerts_parser.add_argument("--unresolved", action="store_true")
    alerts_parser.add_argument("--limit", type=int, default=50)

    return parser


def _cmd_init(args: argparse.Namespace) -> int:
    engine = None
    if args.db_path:
        from src.tracking.db import get_engine

        engine = get_engine(args.db_path)
    init_schema(engine)
    print("Banco inicializado.")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    session = get_session()
    try:
        experiments = ExperimentRepository(session).list(
            stage=args.stage,
            status=args.status,
            limit=args.limit,
        )
        if not experiments:
            print("Nenhum experimento encontrado.")
            return 0
        print(f"{'ID':<6} {'Stage':<12} {'Status':<10} {'Nome':<40} {'Criado'}")
        print("-" * 90)
        for exp in experiments:
            created = exp.created_at.isoformat() if exp.created_at else "-"
            print(f"{exp.id:<6} {exp.stage:<12} {exp.status:<10} " f"{exp.name:<40} {created}")
        return 0
    finally:
        session.close()


def _cmd_show(args: argparse.Namespace) -> int:
    session = get_session()
    try:
        from src.tracking.reporting import experiment_markdown

        print(experiment_markdown(session, args.experiment_id))
        return 0
    finally:
        session.close()


def _cmd_compare(args: argparse.Namespace) -> int:
    session = get_session()
    try:
        summaries = list_summaries(session, stage=args.stage, limit=args.limit)
        print(f"{'ID':<6} {'Nome':<40} {'Métrica':<20} {'Valor':<10} {'Alertas'}")
        print("-" * 90)
        for summary in summaries:
            metric = summary.best_metric or "-"
            value = f"{summary.best_value:.4f}" if summary.best_value is not None else "-"
            print(
                f"{summary.experiment.id:<6} {summary.experiment.name:<40} "
                f"{metric:<20} {value:<10} {summary.n_alerts}"
            )
        return 0
    finally:
        session.close()


def _cmd_export(args: argparse.Namespace) -> int:
    session = get_session()
    try:
        path = save_experiment_report(
            session,
            args.experiment_id,
            args.output_dir,
            fmt=args.format,
        )
        print(f"Relatório salvo em: {path}")
        return 0
    except ValueError as exc:
        print(str(exc))
        return 1
    finally:
        session.close()


def _cmd_alerts(args: argparse.Namespace) -> int:
    session = get_session()
    try:
        alerts = AlertRepository(session).list(
            experiment_id=args.experiment_id,
            severity=args.severity,
            category=args.category,
            unresolved_only=args.unresolved,
            limit=args.limit,
        )
        if not alerts:
            print("Nenhum alerta encontrado.")
            return 0
        print(f"{'ID':<6} {'Run':<6} {'Sev':<10} {'Categoria':<18} {'Mensagem':<50}")
        print("-" * 100)
        for alert in alerts:
            run_id = alert.run_id if alert.run_id is not None else "-"
            msg = (alert.message or "")[:50]
            print(f"{alert.id:<6} {run_id:<6} {alert.severity:<10} " f"{alert.category:<18} {msg}")
        return 0
    finally:
        session.close()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "init": _cmd_init,
        "list-experiments": _cmd_list,
        "show-experiment": _cmd_show,
        "compare": _cmd_compare,
        "export": _cmd_export,
        "alerts": _cmd_alerts,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
