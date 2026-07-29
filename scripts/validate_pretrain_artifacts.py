"""Validate pretrain run artifacts (FASE 3/4).

Checks that a pretrain run directory contains the expected artifacts in a
parseable shape. Used by ``scripts/pretrain_wrapper.py`` and by the
``pretrain-validate`` make target.

Usage:
    python scripts/validate_pretrain_artifacts.py [run_dir]
        (default: newest experiments/*_pretrain_chapman)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger("lewis.camada04.validate")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

MIN_MODEL_BYTES = 1024
REQUIRED_CONFIG_KEYS = ("name", "input_shape", "total_params")
REQUIRED_METRICS_KEYS = ("final_val_loss",)

BASE_REQUIRED = ("backbone_pretrained.keras", "config.json", "metrics.json")
STRICT_EXTRA = (
    "provenance.json",
    "history.json",
    "metrics_per_class.json",
    "run_status.json",
    "qg4_result.json",
)


def newest_run_dir(experiments_dir: Path = EXPERIMENTS_DIR) -> Path | None:
    """Return the newest ``*_pretrain_chapman`` run directory, if any."""
    candidates = sorted(
        (p for p in experiments_dir.glob("*_pretrain_chapman") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _check_json(path: Path, required_keys: tuple[str, ...], problems: list[str]) -> None:
    if not path.exists():
        problems.append(f"{path.name} missing")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f"{path.name} is not valid JSON: {exc}")
        return
    missing = [k for k in required_keys if k not in data]
    if missing:
        problems.append(f"{path.name} missing keys: {missing}")


def validate_run_dir(run_dir: Path, *, strict: bool = False) -> list[str]:
    """Return a list of problems (empty = valid)."""
    run_dir = Path(run_dir)
    problems: list[str] = []
    if not run_dir.is_dir():
        return [f"run directory not found: {run_dir}"]

    model = run_dir / "backbone_pretrained.keras"
    if not model.exists():
        problems.append("backbone_pretrained.keras missing")
    elif model.stat().st_size < MIN_MODEL_BYTES:
        problems.append(f"backbone_pretrained.keras too small ({model.stat().st_size} B)")

    _check_json(run_dir / "config.json", REQUIRED_CONFIG_KEYS, problems)
    _check_json(run_dir / "metrics.json", REQUIRED_METRICS_KEYS, problems)

    if strict:
        for name in STRICT_EXTRA:
            path = run_dir / name
            if not path.exists():
                problems.append(f"{name} missing (strict)")
            elif name.endswith(".json"):
                _check_json(path, (), problems)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="require FASE-4 artifacts too")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_dir = args.run_dir or newest_run_dir()
    if run_dir is None:
        LOGGER.error("no pretrain run directory found under %s", EXPERIMENTS_DIR)
        return 2

    problems = validate_run_dir(run_dir, strict=args.strict)
    if problems:
        for problem in problems:
            LOGGER.error("validate %s: %s", run_dir.name, problem)
        return 1
    LOGGER.info("validate %s: OK%s", run_dir.name, " (strict)" if args.strict else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
