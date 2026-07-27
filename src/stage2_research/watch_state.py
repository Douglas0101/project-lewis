"""Read-only state collection for the E07R training-watch dashboard.

Walks one experiment matrix directory (E06.5-PD or E07-PD) and classifies
every candidate/fold/seed cell as ``done``, ``running`` or ``queued``,
extracting metrics, durations and progress hints from on-disk artifacts.
No writes are performed anywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MAX_LOG_LINE = 200


@dataclass(frozen=True)
class CellSnapshot:
    """Immutable view of one matrix cell."""

    candidate: str
    fold: int
    seed: int
    state: str  # "done" | "running" | "queued"
    f1_f: float | None = None
    macro_f1: float | None = None
    best_epoch: int | None = None
    duration_s: float | None = None
    current_epoch: int | None = None
    last_log_line: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _duration_s(manifest: dict[str, Any]) -> float | None:
    started = _parse_ts(manifest.get("started_at"))
    finished = _parse_ts(manifest.get("finished_at"))
    if started is None or finished is None:
        return None
    return max(0.0, (finished - started).total_seconds())


def _current_epoch(history_path: Path) -> int | None:
    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    epochs: list[int] = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) >= 2 and parts[1].strip().isdigit():
            epochs.append(int(parts[1].strip()))
    return max(epochs) if epochs else None


def _last_log_line(log_path: Path) -> str | None:
    try:
        lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines()]
    except OSError:
        return None
    non_empty = [line for line in lines if line]
    if not non_empty:
        return None
    return non_empty[-1][:MAX_LOG_LINE]


def _snapshot_cell(run_dir: Path, candidate: str, fold: int, seed: int) -> CellSnapshot:
    if not run_dir.is_dir():
        return CellSnapshot(candidate=candidate, fold=fold, seed=seed, state="queued")
    if (run_dir / "DONE").is_file():
        metrics = _read_json(run_dir / "metrics.json")
        manifest = _read_json(run_dir / "run_manifest.json")
        return CellSnapshot(
            candidate=candidate,
            fold=fold,
            seed=seed,
            state="done",
            f1_f=metrics.get("F1_F"),
            macro_f1=metrics.get("macro_F1"),
            best_epoch=metrics.get("best_epoch"),
            duration_s=_duration_s(manifest),
        )
    return CellSnapshot(
        candidate=candidate,
        fold=fold,
        seed=seed,
        state="running",
        current_epoch=_current_epoch(run_dir / "training_history.csv"),
        last_log_line=_last_log_line(run_dir / "stdout.log"),
    )


def collect_cells(
    matrix_root: Path,
    candidates: Iterable[str],
    folds: Iterable[int],
    seeds: Iterable[int],
) -> list[CellSnapshot]:
    """Classify every cell of one experiment matrix in deterministic order."""
    cells: list[CellSnapshot] = []
    for candidate in candidates:
        for fold in folds:
            for seed in seeds:
                run_dir = matrix_root / candidate / f"fold_{fold}" / f"seed_{seed}"
                cells.append(_snapshot_cell(run_dir, candidate, fold, seed))
    return cells


def matrix_summary(cells: list[CellSnapshot]) -> dict[str, Any]:
    """Aggregate counts, mean F1(F), mean duration and a simple ETA."""
    done = [cell for cell in cells if cell.state == "done"]
    running = [cell for cell in cells if cell.state == "running"]
    queued = [cell for cell in cells if cell.state == "queued"]
    durations = [cell.duration_s for cell in done if cell.duration_s is not None]
    mean_duration = (sum(durations) / len(durations)) if durations else None
    remaining = len(running) + len(queued)
    eta = mean_duration * remaining if mean_duration is not None else None
    f1_values = [cell.f1_f for cell in done if cell.f1_f is not None]
    return {
        "done": len(done),
        "running": len(running),
        "queued": len(queued),
        "total": len(cells),
        "mean_duration_s": mean_duration,
        "eta_s": eta,
        "mean_f1_f": (sum(f1_values) / len(f1_values)) if f1_values else None,
    }
