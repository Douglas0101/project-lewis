"""R04: prove RNG state does not advance during inference."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "stage1_float32_v2.0.keras"
FIXTURE_PATH = (
    PROJECT_ROOT / "artifacts" / "stage1_recall_investigation" / "R03" / "loader_fixture.npz"
)


def _run_instance(mode: str, output: Path, seed: int = 42) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "stage1_inference_mode_instance.py"),
        "--model-path",
        str(MODEL_PATH),
        "--fixture",
        str(FIXTURE_PATH),
        "--mode",
        mode,
        "--output",
        str(output),
        "--seed",
        str(seed),
    ]
    process = subprocess.run(cmd, text=True, capture_output=True)
    if process.returncode != 0:
        pytest.fail(process.stdout + process.stderr)
    try:
        value = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid R04 instance result: {output}") from error
    if not isinstance(value, dict):
        raise ValueError(f"R04 instance result is not an object: {output}")
    return value


@pytest.fixture(scope="module")
def predict_evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _run_instance("predict", tmp_path_factory.mktemp("rng-predict") / "predict.json")


@pytest.fixture(scope="module")
def training_false_evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _run_instance(
        "training_false", tmp_path_factory.mktemp("rng-false") / "training_false.json"
    )


@pytest.fixture(scope="module")
def training_true_evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _run_instance(
        "training_true", tmp_path_factory.mktemp("rng-true") / "training_true.json"
    )


def test_rng_state_does_not_advance_during_predict(predict_evidence: dict[str, Any]) -> None:
    assert predict_evidence["initial_rng_state"] == predict_evidence["final_rng_state"]


def test_rng_state_does_not_advance_during_training_false(
    training_false_evidence: dict[str, Any],
) -> None:
    assert (
        training_false_evidence["initial_rng_state"] == training_false_evidence["final_rng_state"]
    )


def test_rng_state_may_advance_during_training_true(
    training_true_evidence: dict[str, Any],
) -> None:
    """training=True is expected to consume RNG; weights must still be immutable."""
    assert bool(training_true_evidence["trainable_weights_immutable"])
    assert bool(training_true_evidence["run"]["any_rng_state_changed"])


def test_training_true_outputs_are_not_all_identical(
    training_true_evidence: dict[str, Any],
) -> None:
    """Stochastic Dropout must produce different outputs across training calls."""
    deltas = [pair["max_abs_delta"] for pair in training_true_evidence["run"]["pairwise"]]
    assert all(delta > 0.0 for delta in deltas)


def test_cross_process_deterministic_reload_with_same_seed(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """First training=True with same seed must be identical across fresh processes."""
    a = tmp_path_factory.mktemp("det-a") / "a.json"
    b = tmp_path_factory.mktemp("det-b") / "b.json"
    seed = 42
    result_a = _run_instance("deterministic_reload", a, seed=seed)
    result_b = _run_instance("deterministic_reload", b, seed=seed)
    assert (
        result_a["run"]["first_training_true"]["sha256"]
        == result_b["run"]["first_training_true"]["sha256"]
    )
