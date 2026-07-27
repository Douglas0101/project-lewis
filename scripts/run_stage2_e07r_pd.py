#!/usr/bin/env python3
"""Canonical CLI for E06.5-PD and conditional E07-PD execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.stage2_research.e07r_integrity import E07RIntegrityError
from src.stage2_research.pd_workflows import run_e065_pd, run_e07_pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("e065-pd", "e07-pd"):
        child = subparsers.add_parser(command)
        child.add_argument(
            "--run-id",
            required=True,
            help="immutable preflight evidence identifier",
        )
        child.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result: dict[str, Any]
        if args.command == "e065-pd":
            result = run_e065_pd(
                PROJECT_ROOT,
                run_id=str(args.run_id),
                dry_run=bool(args.dry_run),
            )
        else:
            result = run_e07_pd(
                PROJECT_ROOT,
                run_id=str(args.run_id),
                dry_run=bool(args.dry_run),
            )
    except (E07RIntegrityError, FileExistsError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 10
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
