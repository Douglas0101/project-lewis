"""Regression tests for shared atomic Stage 2 tabular writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.stage2_research.contracts import ExitCode, ResearchError
from src.stage2_research.tabular_io import atomic_dataframe_csv, atomic_dataframe_parquet


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate": ["baseline", "H6"],
            "F1_F": [0.0, 0.125],
            "fold": [1, 1],
        }
    )


def test_atomic_dataframe_csv_preserves_canonical_content(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "metrics.csv"
    frame = _frame()
    path.parent.mkdir(parents=True)
    path.write_text("stale", encoding="utf-8")

    atomic_dataframe_csv(path, frame)

    assert path.read_text(encoding="utf-8") == frame.to_csv(index=False)
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_atomic_dataframe_parquet_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "predictions.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"stale")
    frame = _frame()

    atomic_dataframe_parquet(path, frame)

    pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
    assert not list(path.parent.glob(f".{path.name}.*.tmp.parquet"))


def test_atomic_dataframe_csv_preserves_contextual_error_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "audit.csv"

    def fail_to_csv(self: pd.DataFrame, **kwargs: Any) -> str:
        del self, kwargs
        raise ValueError("serialization failed")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)

    with pytest.raises(ResearchError, match=r"cannot serialize audit CSV:") as captured:
        atomic_dataframe_csv(path, _frame(), error_label="audit CSV")

    assert captured.value.exit_code == ExitCode.EVALUATION_FAILURE
    assert not path.exists()


def test_atomic_dataframe_parquet_cleans_partial_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "audit.parquet"

    def fail_to_parquet(
        self: pd.DataFrame,
        temporary: Path,
        **kwargs: Any,
    ) -> None:
        del self, kwargs
        Path(temporary).write_bytes(b"partial")
        raise ValueError("serialization failed")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_to_parquet)

    with pytest.raises(ResearchError, match=r"cannot serialize audit parquet:") as captured:
        atomic_dataframe_parquet(path, _frame(), error_label="audit parquet")

    assert captured.value.exit_code == ExitCode.EVALUATION_FAILURE
    assert not path.exists()
    assert not list(tmp_path.glob(f".{path.name}.*.tmp.parquet"))
