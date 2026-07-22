"""Unit contracts for shared Stage 2 validation helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from src.stage2_research.contracts import ExitCode, ResearchError
from src.stage2_research.validation import (
    matches_bool,
    safe_float,
    safe_int,
    validate_template_source_groups,
)


def test_matches_bool_rejects_truthy_non_boolean_values() -> None:
    assert matches_bool(True, True)
    assert matches_bool(False, False)
    assert not matches_bool(1, True)
    assert not matches_bool("true", True)
    assert not matches_bool(None, False)


@pytest.mark.parametrize("value", [0.25, np.float32(0.5), "1.5"])
def test_safe_float_accepts_finite_numeric_values(value: Any) -> None:
    assert safe_float(value, "metric") == float(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_safe_float_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ResearchError, match="metric is not finite") as captured:
        safe_float(value, "metric")

    assert captured.value.exit_code == ExitCode.EVALUATION_FAILURE


def test_safe_int_uses_stable_evaluation_failure_contract() -> None:
    assert safe_int(np.int64(17), "seed") == 17

    with pytest.raises(ResearchError, match="seed is not an integer") as captured:
        safe_int("not-an-int", "seed")

    assert captured.value.exit_code == ExitCode.EVALUATION_FAILURE


def _valid_template_state() -> dict[str, Any]:
    return {
        "inner": {"source_groups": ["patient-a", "patient-b"]},
        "outer": {"source_groups": ["patient-a", "patient-b", "patient-c"]},
    }


def _validate_template_state(template_state: dict[str, Any]) -> None:
    validate_template_source_groups(
        np.asarray(["patient-a", "patient-b", "patient-c", "patient-d"]),
        outer_train=np.asarray([0, 1, 2]),
        outer_test=np.asarray([3]),
        inner_train=np.asarray([0, 1]),
        inner_validation=np.asarray([2]),
        template_state=template_state,
        error_message="template source leakage detected",
    )


def test_template_source_validation_accepts_train_only_groups() -> None:
    _validate_template_state(_valid_template_state())


@pytest.mark.parametrize(
    ("scope", "source_groups"),
    [
        ("inner", ["patient-a", "patient-c"]),
        ("inner", ["patient-a", "patient-d"]),
        ("outer", ["patient-a", "patient-d"]),
        ("outer", ["patient-a", "unknown-patient"]),
    ],
)
def test_template_source_validation_rejects_leakage(
    scope: str,
    source_groups: list[str],
) -> None:
    state = deepcopy(_valid_template_state())
    state[scope]["source_groups"] = source_groups

    with pytest.raises(ResearchError, match="template source leakage detected") as captured:
        _validate_template_state(state)

    assert captured.value.exit_code == ExitCode.LEAKAGE
