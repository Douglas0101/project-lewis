#!/usr/bin/env python3
"""Live TUI dashboard for E07R training matrices (E06.5-PD / E07-PD).

Read-only: renders the on-disk cell state produced by
``scripts/run_stage2_e07r_pd.py`` — progress, per-candidate grid, the
currently running cell and recent completions. Never writes to the tree.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_research.e07r_integrity import run_e07r_preflight  # noqa: E402
from src.stage2_research.pd_workflows import PD_CANDIDATES, PD_SAMPLERS  # noqa: E402
from src.stage2_research.watch_state import (  # noqa: E402
    CellSnapshot,
    collect_cells,
    matrix_summary,
)

RESEARCH_ROOT = PROJECT_ROOT / "experiments" / "stage2_v2.4_research"
FOLDS = (1, 2, 3, 4, 5)
SEEDS = (17, 29, 43, 71, 101)
STAGES = {
    "e065": ("E06.5-PD", RESEARCH_ROOT / "E06_5_PD" / "e06-5-pd-v4-0", PD_CANDIDATES),
    "e07": ("E07-PD", RESEARCH_ROOT / "E07_PD" / "e07-pd-v4-0", PD_SAMPLERS),
}
GLYPHS = {"done": "■", "running": "▶", "queued": "·"}
STYLES = {"done": "green", "running": "yellow", "queued": "dim"}


def _short_name(name: str) -> str:
    """Compact arm label: ``pd_s2_patient_uniform_capped`` → ``S2``."""
    if name.startswith("pd_s") and len(name) > 4 and name[4].isdigit():
        return f"S{name[4]}"
    return name


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    seconds = int(value)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _fmt_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _header(
    stage_label: str,
    matrix_root: Path,
    preflight: str,
    started: float,
    eta: float | None,
) -> Panel:
    elapsed = time.monotonic() - started
    text = (
        f"[bold]{stage_label}[/]  |  {matrix_root.relative_to(PROJECT_ROOT)}  |  "
        f"preflight: [{'green' if preflight == 'PASS' else 'red'}]{preflight}[/]  |  "
        f"elapsed {_fmt_seconds(elapsed)}  |  ETA {_fmt_seconds(eta)}"
    )
    return Panel(
        text,
        title="E07R training watch",
        subtitle=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


def _progress_bar(summary: dict) -> Progress:
    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("F1(F) médio {task.fields[f1]}"),
    )
    progress.add_task(
        "células",
        total=summary["total"],
        completed=summary["done"],
        f1=_fmt_metric(summary["mean_f1_f"]),
    )
    return progress


def _grid(cells: list[CellSnapshot], candidates: tuple[str, ...]) -> Table:
    legend = "  ".join(
        f"{_short_name(name)}={name}" for name in candidates if _short_name(name) != name
    )
    table = Table(
        title="matriz braço × fold (seeds 17/29/43/71/101)",
        caption=legend or None,
        expand=True,
    )
    table.add_column("braço", style="bold", no_wrap=True)
    for fold in FOLDS:
        table.add_column(f"fold {fold}", justify="center")
    for candidate in candidates:
        row = [_short_name(candidate)]
        for fold in FOLDS:
            fold_cells = [
                cell for cell in cells if cell.candidate == candidate and cell.fold == fold
            ]
            fold_cells.sort(key=lambda cell: SEEDS.index(cell.seed))
            marks = " ".join(
                f"[{STYLES[cell.state]}]{GLYPHS[cell.state]}[/]" for cell in fold_cells
            )
            f1_values = [cell.f1_f for cell in fold_cells if cell.f1_f is not None]
            if f1_values:
                marks += f"\n[dim]F1(F)={sum(f1_values) / len(f1_values):.2f}[/]"
            row.append(marks)
        table.add_row(*row)
    return table


def _current(cells: list[CellSnapshot]) -> Panel:
    running = [cell for cell in cells if cell.state == "running"]
    if not running:
        return Panel("[dim]nenhuma célula em execução[/]", title="célula atual")
    cell = running[0]
    epoch = "—" if cell.current_epoch is None else str(cell.current_epoch)
    log = cell.last_log_line or "—"
    return Panel(
        f"{cell.candidate}/fold_{cell.fold}/seed_{cell.seed}  |  época {epoch}\n{log}",
        title="célula atual",
    )


def _recent(cells: list[CellSnapshot], limit: int = 8) -> Table:
    table = Table(title="concluídas recentes", expand=True)
    for column in ("candidato", "fold", "seed", "F1(F)", "macro", "best_ep", "duração"):
        table.add_column(column, justify="center")
    done = [cell for cell in cells if cell.state == "done"]
    for cell in done[-limit:]:
        table.add_row(
            cell.candidate,
            str(cell.fold),
            str(cell.seed),
            _fmt_metric(cell.f1_f),
            _fmt_metric(cell.macro_f1),
            "—" if cell.best_epoch is None else str(cell.best_epoch),
            _fmt_seconds(cell.duration_s),
        )
    return table


def _render(stage_key: str, preflight: str, started: float) -> Group:
    stage_label, matrix_root, candidates = STAGES[stage_key]
    cells = collect_cells(matrix_root, candidates, FOLDS, SEEDS)
    summary = matrix_summary(cells)
    return Group(
        _header(stage_label, matrix_root, preflight, started, summary["eta_s"]),
        _progress_bar(summary),
        _grid(cells, candidates),
        _current(cells),
        _recent(cells),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), default="e065")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="renderiza um snapshot e sai")
    args = parser.parse_args(argv)

    console = Console()
    preflight = run_e07r_preflight(PROJECT_ROOT, workflow="FREEZE_VALIDATION", run_id="e07r-watch")
    preflight_status = preflight.status
    started = time.monotonic()

    if args.once:
        console.print(_render(args.stage, preflight_status, started))
        return 0 if preflight_status == "PASS" else 1

    with Live(
        _render(args.stage, preflight_status, started),
        console=console,
        refresh_per_second=1,
        screen=True,
    ) as live:
        try:
            while True:
                time.sleep(args.interval)
                live.update(_render(args.stage, preflight_status, started))
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
