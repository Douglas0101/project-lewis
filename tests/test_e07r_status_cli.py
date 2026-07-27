"""Smoke tests for the read-only E07R status CLI used by the make targets."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "check_e07r_status.py"


def _run_status() -> tuple[int, dict]:
    process = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    payload = json.loads(process.stdout)
    return process.returncode, payload


def test_status_cli_emits_expected_schema() -> None:
    returncode, payload = _run_status()

    assert returncode in (0, 1)
    assert payload["preflight"]["status"] in ("PASS", "BLOCKED")
    assert set(payload) == {"preflight", "e065_pd", "e07_pd"}
    assert set(payload["e065_pd"]) == {"done", "total", "selection"}
    assert set(payload["e07_pd"]) == {"done", "total", "eligible"}


def test_status_cli_counts_and_checks_are_consistent() -> None:
    returncode, payload = _run_status()

    assert returncode in (0, 1)
    assert len(payload["preflight"]["checks"]) == 9
    assert all(isinstance(code, str) for code in payload["preflight"]["checks"])
    for section, total in (("e065_pd", 100), ("e07_pd", 150)):
        done = payload[section]["done"]
        assert isinstance(done, int)
        assert 0 <= done <= payload[section]["total"] == total
    assert isinstance(payload["e07_pd"]["eligible"], bool)
    if payload["preflight"]["status"] == "PASS":
        assert returncode == 0
    else:
        assert returncode == 1
