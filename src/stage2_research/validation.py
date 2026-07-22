"""Shared scalar and leakage validation for Stage 2 research."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.stage2_research.contracts import ExitCode, ResearchError


def matches_bool(value: Any, expected: bool) -> bool:
    """Match a JSON boolean without accepting integers or truthy values."""
    return isinstance(value, bool) and value == expected


def safe_float(value: Any, name: str) -> float:
    """Convert one evaluation value and reject non-finite numbers."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResearchError(
            f"{name} is not numeric",
            ExitCode.EVALUATION_FAILURE,
        ) from error
    if not math.isfinite(result):
        raise ResearchError(
            f"{name} is not finite",
            ExitCode.EVALUATION_FAILURE,
        )
    return result


def safe_int(value: Any, name: str) -> int:
    """Convert one evaluation value to an integer with a stable failure code."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResearchError(
            f"{name} is not an integer",
            ExitCode.EVALUATION_FAILURE,
        ) from error


def validate_template_source_groups(
    groups: np.ndarray,
    *,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    inner_train: np.ndarray,
    inner_validation: np.ndarray,
    template_state: Mapping[str, Any],
    error_message: str,
) -> None:
    """Prove inner/outer template sources remain train-only."""
    normalized_groups = np.asarray(groups).astype(str)
    inner_sources = {str(item) for item in template_state["inner"]["source_groups"]}
    outer_sources = {str(item) for item in template_state["outer"]["source_groups"]}
    inner_allowed = set(normalized_groups[inner_train].tolist())
    inner_forbidden = set(normalized_groups[inner_validation].tolist()) | set(
        normalized_groups[outer_test].tolist()
    )
    outer_allowed = set(normalized_groups[outer_train].tolist())
    outer_forbidden = set(normalized_groups[outer_test].tolist())
    if (
        not inner_sources <= inner_allowed
        or inner_sources & inner_forbidden
        or not outer_sources <= outer_allowed
        or outer_sources & outer_forbidden
    ):
        raise ResearchError(error_message, ExitCode.LEAKAGE)
