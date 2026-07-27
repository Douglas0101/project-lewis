"""Unit tests for the E07R training-watch state collector."""

from __future__ import annotations

import json
from pathlib import Path

from src.stage2_research.watch_state import collect_cells, matrix_summary

CANDIDATES = ("baseline", "H6")
FOLDS = (1, 2)
SEEDS = (17, 29)


def _write_done_cell(run_dir: Path, *, f1_f: float, macro: float, duration_s: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({"F1_F": f1_f, "macro_F1": macro, "best_epoch": 3, "F_support": 10}),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "started_at": "2026-07-26T18:00:00+00:00",
                "finished_at": f"2026-07-26T18:00:{duration_s:02d}+00:00",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "DONE").write_text("done\n", encoding="utf-8")


def _write_running_cell(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".RUNNING.lock").write_text("pid\n", encoding="utf-8")
    (run_dir / "training_history.csv").write_text(
        "phase,epoch,accuracy,loss,val_accuracy,val_loss\n"
        "inner_selection,1,0.5,0.7,0.5,0.7\n"
        "inner_selection,2,0.6,0.6,0.6,0.6\n",
        encoding="utf-8",
    )
    (run_dir / "stdout.log").write_text("PASS candidate=H6\nbest_epoch=2\n", encoding="utf-8")


def _cell(cells, candidate, fold, seed):
    matches = [
        cell
        for cell in cells
        if cell.candidate == candidate and cell.fold == fold and cell.seed == seed
    ]
    assert len(matches) == 1
    return matches[0]


def test_collect_cells_classifies_done_running_and_queued(tmp_path: Path) -> None:
    root = tmp_path / "E06_5_PD" / "exp"
    _write_done_cell(root / "baseline" / "fold_1" / "seed_17", f1_f=0.42, macro=0.55, duration_s=30)
    _write_running_cell(root / "H6" / "fold_2" / "seed_29")

    cells = collect_cells(root, CANDIDATES, FOLDS, SEEDS)

    assert len(cells) == len(CANDIDATES) * len(FOLDS) * len(SEEDS)
    done = _cell(cells, "baseline", 1, 17)
    assert done.state == "done"
    assert done.f1_f == 0.42
    assert done.macro_f1 == 0.55
    assert done.best_epoch == 3
    assert done.duration_s == 30
    running = _cell(cells, "H6", 2, 29)
    assert running.state == "running"
    assert running.current_epoch == 2
    assert "best_epoch=2" in (running.last_log_line or "")
    assert _cell(cells, "H6", 1, 17).state == "queued"


def test_matrix_summary_counts_and_eta(tmp_path: Path) -> None:
    root = tmp_path / "E06_5_PD" / "exp"
    _write_done_cell(root / "baseline" / "fold_1" / "seed_17", f1_f=0.4, macro=0.5, duration_s=20)
    _write_done_cell(root / "baseline" / "fold_1" / "seed_29", f1_f=0.6, macro=0.7, duration_s=40)
    _write_running_cell(root / "H6" / "fold_2" / "seed_29")

    summary = matrix_summary(collect_cells(root, CANDIDATES, FOLDS, SEEDS))

    assert summary["done"] == 2
    assert summary["running"] == 1
    assert summary["queued"] == 5
    assert summary["total"] == 8
    assert summary["mean_duration_s"] == 30.0
    assert summary["eta_s"] == 30.0 * 6
    assert summary["mean_f1_f"] == 0.5


def test_collect_cells_tolerates_partial_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "E06_5_PD" / "exp"
    broken = root / "baseline" / "fold_1" / "seed_17"
    broken.mkdir(parents=True)
    (broken / ".RUNNING.lock").write_text("pid\n", encoding="utf-8")

    cells = collect_cells(root, CANDIDATES, FOLDS, SEEDS)

    assert _cell(cells, "baseline", 1, 17).state == "running"
    assert _cell(cells, "baseline", 1, 17).current_epoch is None
    summary = matrix_summary(cells)
    assert summary["mean_duration_s"] is None
    assert summary["eta_s"] is None
