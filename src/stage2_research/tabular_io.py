"""Atomic tabular artifact writers shared by Stage 2 workflows."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.stage2_research.contracts import ExitCode, ResearchError
from src.stage2_research.integrity import atomic_write_text


def atomic_dataframe_csv(
    path: Path,
    frame: pd.DataFrame,
    *,
    error_label: str = "CSV",
) -> None:
    """Serialize a dataframe through the existing atomic text writer."""
    try:
        content = frame.to_csv(index=False)
    except (TypeError, ValueError) as error:
        raise ResearchError(
            f"cannot serialize {error_label}: {path}",
            ExitCode.EVALUATION_FAILURE,
        ) from error
    atomic_write_text(path, content)


def atomic_dataframe_parquet(
    path: Path,
    frame: pd.DataFrame,
    *,
    error_label: str = "parquet",
) -> None:
    """Atomically serialize a dataframe as parquet with cleanup on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError, ImportError) as error:
        raise ResearchError(
            f"cannot serialize {error_label}: {path}",
            ExitCode.EVALUATION_FAILURE,
        ) from error
    finally:
        if temporary.exists():
            temporary.unlink()
