"""R04: forbid training=True in production inference paths while allowing tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_constant_bool(
    keyword: ast.keyword,
    *,
    desired: bool,
) -> bool:
    """Check if keyword value is a boolean Constant with the desired value."""
    value = keyword.value
    return (
        keyword.arg == "training"
        and isinstance(value, ast.Constant)
        and isinstance(value.value, bool)
        and bool(value.value) is desired
    )


def _find_training_true_calls(path: Path) -> list[tuple[str, int]]:
    """Return (file, line) for model(..., training=True) in Python files."""
    hits: list[tuple[str, int]] = []
    search_roots = [path / "src", path / "tests", path / "scripts"]
    for root in search_roots:
        if not root.exists():
            continue
        for source in root.rglob("*.py"):
            if ".venv" in source.parts or "node_modules" in source.parts:
                continue
            try:
                text = source.read_text(encoding="utf-8")
                tree = ast.parse(text)
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if _is_constant_bool(keyword, desired=True):
                        rel = source.relative_to(path)
                        hits.append((str(rel), node.lineno or 0))
    return hits


def _find_training_false_in_production(path: Path) -> list[tuple[str, int]]:
    """Return (file, line) for production model(..., training=False)."""
    hits: list[tuple[str, int]] = []
    root = path / "src"
    if not root.exists():
        return hits
    for source in root.rglob("*.py"):
        if ".venv" in source.parts or "node_modules" in source.parts:
            continue
        try:
            text = source.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if _is_constant_bool(keyword, desired=False):
                    rel = source.relative_to(path)
                    hits.append((str(rel), node.lineno or 0))
    return hits


@pytest.fixture(scope="module")
def all_training_true_hits() -> list[tuple[str, int]]:
    return _find_training_true_calls(PROJECT_ROOT)


def test_no_training_true_in_production(all_training_true_hits: list[tuple[str, int]]) -> None:
    """Only diagnostics, training scripts, or tests may use training=True."""
    production_hits = [
        (file, line)
        for file, line in all_training_true_hits
        if file.startswith("src/")
        and "callbacks" not in file
        and "slha" not in file
        and "models" not in file
    ]
    assert production_hits == [], f"training=True in production-like src path: {production_hits}"


def test_training_true_allowed_in_diagnostics(
    all_training_true_hits: list[tuple[str, int]],
) -> None:
    """R04 forensic scripts legitimately use training=True for diagnostics."""
    diagnostic_hits = [
        (file, line)
        for file, line in all_training_true_hits
        if "stage1_inference_mode_instance.py" in file or "audit_stage1_inference_mode.py" in file
    ]
    assert len(diagnostic_hits) >= 1


def test_training_false_in_production_is_safe() -> None:
    """training=False is an explicit, safe inference-mode annotation."""
    hits = _find_training_false_in_production(PROJECT_ROOT)
    # Only warmup.py and gradient_monitor.py use training=False; both are training-time code.
    assert all(
        "warmup.py" in file or "gradient_monitor.py" in file or "slha" in file for file, _ in hits
    ), f"Unexpected training=False in production: {hits}"
