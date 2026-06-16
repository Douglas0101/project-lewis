"""Pytest configuration and shared fixtures for Project-Lewis tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIRMWARE_ROOT = PROJECT_ROOT / "firmware"
REPORT_PATHS = [
    FIRMWARE_ROOT / "build" / "stm32f4" / "firmware_simulation_report.json",
    PROJECT_ROOT / "reports" / "firmware_simulation_report.json",
]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    return project_root / "data"


@pytest.fixture
def skip_if_no_data(request, data_dir: Path) -> None:
    """Auto-skip a test when ``data/raw_*`` does not exist.

    Most QG0 tests are integration tests that depend on real downloads.
    The fixture checks for *any* ``raw_*`` directory so that unit-only runs
    (e.g. ``pytest tests/test_download.py -k schema``) still execute.
    """
    if not any(
        (data_dir / f"raw_{ds}").exists() for ds in ("chapman", "mitbih", "svdb", "afdb", "incart")
    ):
        if os.environ.get("LEWIS_REQUIRE_DATA") == "1":
            pytest.fail("data/raw_* missing — set up the env or run `make download-all`")
        pytest.skip("data/raw_* missing — run `make download-all` to enable this test")


def _all_raw_present(data_dir: Path) -> bool:
    return all(
        (data_dir / f"raw_{ds}").exists() for ds in ("chapman", "mitbih", "svdb", "afdb", "incart")
    )


@pytest.fixture(scope="session", autouse=True)
def _autoskip_session(data_dir: Path) -> None:
    """Mark the session as data-required when all raw dirs are present."""
    if _all_raw_present(data_dir):
        os.environ.setdefault("LEWIS_DATA_PRESENT", "1")


@pytest.fixture(scope="module")
def firmware_report() -> dict:
    """Localiza e carrega o relatorio de simulacao do firmware.

    Os testes de QG7/QG8/QG9 dependem do artefato gerado por
    ``make -C firmware LEWIS_USE_TFLM=1 firmware-test``.
    """
    report_path = next((p for p in REPORT_PATHS if p.exists()), None)
    if report_path is None:
        pytest.skip(
            "Relatorio do firmware nao encontrado; execute "
            "`make -C firmware LEWIS_USE_TFLM=1 firmware-test` primeiro."
        )
    return json.loads(report_path.read_text())
