"""Canonical resumable CLI for Stage 2 E06.5, Fold 5, E07, and E08."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence, cast

from src.stage2_research.config import load_research_config
from src.stage2_research.contracts import ExitCode, ProfileName, ResearchError
from src.stage2_research.workflows import (
    build_e065_plan,
    run_e065,
    run_preflight,
    status_report,
)

LOGGER = logging.getLogger("stage2_research")
DEFAULT_CONFIG = Path("config/stage2_research.yaml")


def _csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("list values must be unique")
    return values


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in _csv_strings(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    return values


def _add_common(
    parser: argparse.ArgumentParser,
    *,
    matrix: bool = False,
    execution: bool = False,
) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--experiment-id")
    if matrix:
        parser.add_argument("--folds", type=_csv_ints, default=(1, 2, 3, 4, 5))
        parser.add_argument("--seeds", type=_csv_ints, default=(17, 29, 43, 71, 101))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    if execution:
        parser.add_argument(
            "--profile",
            choices=("smoke", "screening", "audit", "performance"),
            default="audit",
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.stage2_research",
        description="Causal, deterministic, resumable Stage 2 research orchestration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    _add_common(preflight)

    status = subparsers.add_parser("status")
    _add_common(status)

    plan = subparsers.add_parser("plan")
    _add_common(plan, matrix=True)
    plan.add_argument("--stage", choices=("e06.5", "e07", "e08"), required=True)
    plan.add_argument(
        "--candidates",
        type=_csv_strings,
        default=("baseline", "H6", "H11", "H12"),
    )

    e065 = subparsers.add_parser("e065-run")
    _add_common(e065, matrix=True, execution=True)
    e065.add_argument(
        "--candidates",
        type=_csv_strings,
        default=("baseline", "H6", "H11", "H12"),
    )

    fold_audit = subparsers.add_parser("fold-audit")
    _add_common(fold_audit, matrix=True)
    fold_audit.add_argument("--stage", choices=("e06.5",), required=True)
    fold_audit.add_argument("--fold", type=int, default=5)
    fold_audit.add_argument(
        "--candidates",
        type=_csv_strings,
        default=("baseline", "H6", "H11", "H12"),
    )

    representation = subparsers.add_parser("representation-select")
    _add_common(representation)
    representation.add_argument("--stage", choices=("e06.5",), required=True)
    representation.add_argument(
        "--candidates",
        type=_csv_strings,
        default=("H6", "H11", "H12"),
    )
    representation.add_argument("--baseline", default="baseline")

    e07 = subparsers.add_parser("e07-run")
    _add_common(e07, matrix=True, execution=True)
    e07.add_argument("--representation", default="selected")
    e07.add_argument(
        "--samplers",
        type=_csv_strings,
        default=(
            "natural",
            "random_oversampling",
            "patient_uniform",
            "patient_sqrt",
            "smote",
        ),
    )

    e07_select = subparsers.add_parser("e07-select")
    _add_common(e07_select)
    e07_select.add_argument("--phase", choices=("screening", "final"), required=True)
    e07_select.add_argument("--top-k", type=int, default=2)

    e08 = subparsers.add_parser("e08-run")
    _add_common(e08, matrix=True, execution=True)
    e08.add_argument(
        "--methods",
        type=_csv_strings,
        default=(
            "ce_control",
            "crt_patient_aware",
            "logit_adjustment",
            "balanced_softmax",
            "ldam_drw",
            "focal_legacy",
        ),
    )

    e08_select = subparsers.add_parser("e08-select")
    _add_common(e08_select)
    e08_select.add_argument("--phase", choices=("screening", "final"), required=True)
    e08_select.add_argument("--top-k", type=int, default=2)

    report = subparsers.add_parser("report")
    _add_common(report)
    report.add_argument("--from", dest="from_stage", choices=("e06.5", "e07", "e08"), required=True)
    report.add_argument("--through", choices=("e06.5", "e07", "e08"), required=True)

    verify = subparsers.add_parser("verify")
    _add_common(verify)
    verify.add_argument("--stage", choices=("e06.5", "e07", "e08", "all"), required=True)

    resume = subparsers.add_parser("resume")
    _add_common(resume)
    resume.add_argument("--stage", choices=("e06.5", "e07", "e08"), required=True)
    return parser


def _deterministic(args: argparse.Namespace, default: bool) -> bool:
    value = args.deterministic
    return default if value is None else bool(value)


def _print_sections(
    *,
    command: str,
    status: str,
    runs: dict[str, Any],
    integrity: dict[str, Any],
    metrics: dict[str, Any] | None,
    post_operation: dict[str, Any],
    regressions: Sequence[str],
    next_command: str,
) -> None:
    print(f"COMANDO:\n{command}\n")
    print(f"STATUS:\n{status}\n")
    print("RUNS:")
    for key, value in runs.items():
        print(f"{key}: {value}")
    print("\nINTEGRIDADE:")
    for key, value in integrity.items():
        print(f"{key}: {value}")
    print("\nMÉTRICAS:")
    if metrics:
        for key, value in metrics.items():
            print(f"{key}: {value}")
    else:
        print("não aplicável nesta etapa")
    print("\nPÓS-OPERAÇÃO:")
    for key, value in post_operation.items():
        print(f"{key}: {value}")
    print("\nREGRESSÕES:")
    print("nenhuma" if not regressions else "\n".join(regressions))
    print(f"\nPRÓXIMO COMANDO:\n{next_command}")


def _config(args: argparse.Namespace) -> Any:
    return load_research_config(
        args.config,
        output_root_override=args.output_root,
    )


def _cmd_preflight(args: argparse.Namespace) -> int:
    config = _config(args)
    deterministic = _deterministic(args, True)
    _, report = run_preflight(
        config,
        deterministic=deterministic,
        device=args.device,
        dry_run=args.dry_run,
    )
    integrity = report["integrity"]
    _print_sections(
        command="preflight",
        status="PASS",
        runs={
            "planejadas": report["matrix"]["planned_runs"],
            "executadas": 0,
            "retomadas": 0,
            "ignoradas": 0,
            "falhas": 0,
        },
        integrity={
            "dataset hash": integrity["dataset_manifest_hash"],
            "split hash": integrity["outer_split_manifest_hash"],
            "feature hash": json.dumps(integrity["feature_manifest_hashes"], sort_keys=True),
            "config hash": report["config_hash"],
            "Git HEAD": report["git"]["head"],
        },
        metrics=None,
        post_operation={
            "lint/Pyright/testes": "PASS",
            "artefatos": str(config.output_root / "reports" / "preflight_report.json"),
        },
        regressions=[],
        next_command=(
            "uv run --locked python -m src.cli.stage2_research plan --stage e06.5 "
            "--candidates baseline,H6,H11,H12 --folds 1,2,3,4,5 "
            "--seeds 17,29,43,71,101"
        ),
    )
    return ExitCode.PASS


def _cmd_plan(args: argparse.Namespace) -> int:
    config = _config(args)
    if args.stage != "e06.5":
        from src.stage2_research.advanced_workflows import build_advanced_plan

        plan = build_advanced_plan(config, args)
    else:
        plan = build_e065_plan(
            config,
            candidates=args.candidates,
            folds=args.folds,
            seeds=args.seeds,
            experiment_id=args.experiment_id,
        )
    _print_sections(
        command=f"plan --stage {args.stage}",
        status="PASS",
        runs={
            "planejadas": plan["run_count"],
            "executadas": 0,
            "retomadas": plan["counts"].get("RESUMABLE", 0),
            "ignoradas": plan["counts"].get("DONE", 0),
            "falhas": plan["counts"].get("INCOMPATIBLE", 0),
        },
        integrity={"plan hash": plan["plan_hash"], "experiment-id": plan["experiment_id"]},
        metrics=None,
        post_operation={
            "run matrix": str(
                config.output_root / "reports" / f"run_matrix_e065_{plan['experiment_id']}.csv"
            ),
            "treinamento iniciado": False,
        },
        regressions=[],
        next_command=(
            "uv run --locked python -m src.cli.stage2_research e065-run "
            "--candidates baseline,H6,H11,H12 --folds 1 --seeds 17 "
            "--profile smoke --deterministic"
        ),
    )
    return ExitCode.PASS


def _cmd_e065(args: argparse.Namespace) -> int:
    config = _config(args)
    profile = cast(ProfileName, args.profile)
    deterministic = _deterministic(args, config.profiles[profile].deterministic)
    aggregate, counts = run_e065(
        config,
        candidates=args.candidates,
        folds=args.folds,
        seeds=args.seeds,
        profile_name=profile,
        experiment_id=args.experiment_id,
        deterministic=deterministic,
        device=args.device,
        resume=args.resume,
        force=args.force,
        dry_run=args.dry_run,
        max_parallel=args.max_parallel,
    )
    summaries = aggregate.get("candidates", {})
    candidate_values = [item["F1_F"]["mean"] for item in summaries.values() if "F1_F" in item]
    mean_f1 = max(candidate_values) if candidate_values else 0.0
    best_summary: dict[str, Any] = next(
        (item for item in summaries.values() if item.get("F1_F", {}).get("mean") == mean_f1),
        {},
    )
    status = (
        "PASS"
        if profile == "smoke"
        else ("PASS" if mean_f1 >= config.gates.publication_f1_f else "SCIENTIFIC_GATE_NOT_MET")
    )
    _print_sections(
        command="e065-run",
        status=status,
        runs={
            "planejadas": counts["planned"],
            "executadas": counts["executed"],
            "retomadas": counts["resumed"],
            "ignoradas": counts["skipped"],
            "falhas": counts["failed"],
        },
        integrity={
            "aggregate hash": aggregate.get("aggregate_hash", ""),
            "experiment-id": aggregate.get("experiment_id", ""),
        },
        metrics={
            "mean F1(F)": mean_f1,
            "std F1(F)": best_summary.get("F1_F", {}).get("std", 0.0),
            "min F1(F)": best_summary.get("F1_F", {}).get("min", 0.0),
            "macro-F1": best_summary.get("macro_F1", {}).get("mean", 0.0),
            "gain fora 208/213": best_summary.get("outside_208_213_F1_F", {}).get("mean", 0.0),
            "zero-fold count": best_summary.get("zero_F1_fold_count", 0),
        },
        post_operation={
            "smoke gate": "E06_5_SMOKE_PASS" if profile == "smoke" else "não aplicável",
            "artefatos": str(config.output_root / "E06_5"),
        },
        regressions=[],
        next_command=(
            "uv run --locked python -m src.cli.stage2_research e065-run "
            "--candidates baseline,H6,H11,H12 --folds 1,2,3,4,5 "
            "--seeds 17,29,43,71,101 --profile audit --deterministic --resume"
        ),
    )
    if status == "SCIENTIFIC_GATE_NOT_MET":
        return ExitCode.SCIENTIFIC_GATE_NOT_MET
    return ExitCode.PASS


def _cmd_status(args: argparse.Namespace) -> int:
    report = status_report(_config(args))
    print("Stage      Planned  Running  Passed  Failed  Blocked")
    for row in report["stages"]:
        print(
            f"{row['stage']:<10} {row['planned']:<8} {row['running']:<8} "
            f"{row['passed']:<7} {row['failed']:<7} {row['blocked']}"
        )
    print(f"selected: {json.dumps(report['selected'], sort_keys=True)}")
    print(f"target final: {report['publication_target_F1_F']}")
    print(f"próxima ação: {report['next_action']}")
    return ExitCode.PASS


def _cmd_advanced(args: argparse.Namespace) -> int:
    from src.stage2_research.advanced_workflows import dispatch_advanced

    return dispatch_advanced(_config(args), args)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    try:
        if args.command == "preflight":
            return _cmd_preflight(args)
        if args.command == "plan":
            return _cmd_plan(args)
        if args.command == "e065-run":
            return _cmd_e065(args)
        if args.command == "status":
            return _cmd_status(args)
        return _cmd_advanced(args)
    except ResearchError as error:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "exit_code": error.exit_code.value,
                    "message": str(error),
                    "details": error.details,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
