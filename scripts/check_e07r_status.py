#!/usr/bin/env python3
"""Read-only E07R status panel: preflight, cell counts and H*-PD selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_research.e07r_integrity import run_e07r_preflight  # noqa: E402

RESEARCH_ROOT = PROJECT_ROOT / "experiments" / "stage2_v2.4_research"
E06_DIR = RESEARCH_ROOT / "E06_5_PD"
E07_DIR = RESEARCH_ROOT / "E07_PD"
SELECTION_PATH = E06_DIR / "e06-5-pd-v4-0" / "h_star_pd_selection.json"


def _done_count(root: Path) -> int:
    return sum(1 for _ in root.rglob("DONE")) if root.is_dir() else 0


def main() -> int:
    preflight = run_e07r_preflight(
        PROJECT_ROOT,
        workflow="FREEZE_VALIDATION",
        run_id="make-status",
    )
    selection = None
    if SELECTION_PATH.is_file():
        stored = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        selection = {
            "status": stored["status"],
            "decision_reasons": stored["decision_reasons"],
            "h6_minus_baseline_f1_f": stored["h6_minus_baseline_f1_f"],
        }
    payload = {
        "preflight": {
            "status": preflight.status,
            "checks": {check.code: check.status for check in preflight.checks},
        },
        "e065_pd": {
            "done": _done_count(E06_DIR),
            "total": 100,
            "selection": selection,
        },
        "e07_pd": {
            "done": _done_count(E07_DIR),
            "total": 150,
            "eligible": bool(selection and selection["status"] == "VALID_H_STAR_PD"),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if preflight.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
