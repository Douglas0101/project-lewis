"""Runtime identity guard for official ``uv`` project commands."""

from __future__ import annotations

import json
from pathlib import Path

from src.runtime_identity import DEFAULT_MANIFEST, inspect_runtime


def test_runtime_identity_matches_declared_interpreter() -> None:
    """Pytest under the official uv command must use the manifest runtime."""
    identity = inspect_runtime(DEFAULT_MANIFEST)

    assert identity.executable_matches
    assert identity.python_version_matches
    assert identity.keras_version == "3.14.1"
    assert identity.tensorflow_version == "2.21.0"


def test_runtime_identity_rejects_wrong_declared_version(tmp_path: Path) -> None:
    """A conflicting manifest must be detected rather than silently accepted."""
    manifest = tmp_path / "runtime_identity.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "canonical_interpreter": ".venv/bin/python3",
                "python_major": 3,
                "python_minor": 13,
            }
        ),
        encoding="utf-8",
    )

    identity = inspect_runtime(manifest)

    assert identity.executable_matches
    assert not identity.python_version_matches
