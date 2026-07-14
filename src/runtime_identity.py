"""Verify that official project commands use the declared Python runtime."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "runtime_identity.json"


class RuntimeManifest(BaseModel):
    """Canonical interpreter contract declared by the project."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(pattern=r"^1\.0$")
    canonical_interpreter: str
    python_major: int
    python_minor: int


class RuntimeIdentity(BaseModel):
    """Observed runtime identity for audit output."""

    model_config = ConfigDict(extra="forbid")

    sys_executable: str
    sys_executable_resolved: str
    python_version: str
    conda_prefix: str | None
    virtual_env: str | None
    keras_version: str
    tensorflow_version: str
    canonical_executable: str
    canonical_executable_resolved: str
    executable_matches: bool
    python_version_matches: bool


def _load_manifest(path: Path) -> RuntimeManifest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return RuntimeManifest.model_validate(data)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise ValueError(f"Invalid runtime identity manifest: {path}") from error


def inspect_runtime(manifest_path: Path = DEFAULT_MANIFEST) -> RuntimeIdentity:
    """Collect runtime versions and compare with the declared interpreter."""
    manifest = _load_manifest(manifest_path)
    canonical_declared = PROJECT_ROOT / manifest.canonical_interpreter
    canonical_resolved = canonical_declared.resolve(strict=True)
    observed = Path(sys.executable)
    observed_resolved = observed.resolve(strict=True)

    import keras
    import tensorflow as tf

    return RuntimeIdentity(
        sys_executable=sys.executable,
        sys_executable_resolved=str(observed_resolved),
        python_version=sys.version,
        conda_prefix=os.environ.get("CONDA_PREFIX"),
        virtual_env=os.environ.get("VIRTUAL_ENV"),
        keras_version=keras.__version__,
        tensorflow_version=tf.__version__,
        canonical_executable=str(canonical_declared),
        canonical_executable_resolved=str(canonical_resolved),
        executable_matches=observed_resolved == canonical_resolved,
        python_version_matches=(
            sys.version_info.major == manifest.python_major
            and sys.version_info.minor == manifest.python_minor
        ),
    )


def main() -> int:
    """Print the complete identity and fail outside the canonical runtime."""
    identity = inspect_runtime()
    print(identity.model_dump_json(indent=2))
    if not identity.executable_matches or not identity.python_version_matches:
        print("ERROR: official command is not using the canonical project interpreter")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
