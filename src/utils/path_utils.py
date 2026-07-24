from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-\.]+$")


def resolve_scaler_path(scaler_path: str) -> Path:
    """Resolve scaler path and ensure it stays inside features dir."""
    filename = os.path.basename(scaler_path)
    if (
        not filename
        or filename in (".", "..")
        or os.path.sep in filename
        or "/" in filename
        or "\\" in filename
        or not _SAFE_NAME_RE.match(filename)
    ):
        raise ValueError(f"Invalid scaler filename: {filename!r}")

    target = FEATURES_DIR / filename
    resolved = target.resolve()
    try:
        resolved.relative_to(FEATURES_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"Scaler path escapes features directory: {filename!r}") from exc
    return resolved


def resolve_output_dir(output_dir: str) -> Path:
    """Resolve output directory and ensure it stays inside PROJECT_ROOT."""
    target = Path(output_dir)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    resolved = target.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Output directory escapes project root: {output_dir!r}") from exc
    return resolved
