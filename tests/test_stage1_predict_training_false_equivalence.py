"""R04: prove predict() and training=False are equivalent and repeatable."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.requires_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "stage1_float32_v2.0.keras"
FIXTURE_PATH = (
    PROJECT_ROOT / "artifacts" / "stage1_recall_investigation" / "R03" / "loader_fixture.npz"
)
FIXTURE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "stage1_recall_investigation"
    / "R03"
    / "loader_fixture_manifest.json"
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
    output = tmp_path_factory.mktemp("stage1-predict") / "predict.json"
    return _run_instance("predict", output)


@pytest.fixture(scope="module")
def training_false_evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    output = tmp_path_factory.mktemp("stage1-training-false") / "training_false.json"
    return _run_instance("training_false", output)


def test_r03_fixture_is_immutable_and_untouched() -> None:
    """R03 fixture must match its manifest and remain unchanged."""
    assert FIXTURE_PATH.exists()
    assert FIXTURE_MANIFEST_PATH.exists()

    try:
        manifest = json.loads(FIXTURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Invalid R03 fixture manifest") from error
    file_sha256 = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()

    with np.load(FIXTURE_PATH, allow_pickle=False) as data:
        x_values = data["X_scaled"]
        values = np.ascontiguousarray(x_values)
        digest = hashlib.sha256()
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(json.dumps(list(values.shape)).encode("ascii"))
        digest.update(values.tobytes(order="C"))
        array_sha256 = digest.hexdigest()

    assert file_sha256 == manifest["sha256"]
    assert array_sha256 == manifest["array_sha256"]
    assert list(x_values.shape) == [11, 500, 1]
    assert str(x_values.dtype) == "float32"


def _all_pairwise_equal(evidence: dict[str, Any]) -> bool:
    return all(p["array_equal"] for p in evidence["run"]["pairwise"])


def test_predict_is_reproducible(predict_evidence: dict[str, Any]) -> None:
    """Three consecutive predict() calls on the same instance must be identical."""
    assert _all_pairwise_equal(predict_evidence)
    for pair in predict_evidence["run"]["pairwise"]:
        assert pair["argmax_disagreement_count"] == 0
        assert pair["threshold_disagreement_count"] == 0
        assert pair["max_abs_delta"] == 0.0


def test_training_false_is_reproducible(training_false_evidence: dict[str, Any]) -> None:
    """Three consecutive model(..., training=False) calls must be identical."""
    assert _all_pairwise_equal(training_false_evidence)
    for pair in training_false_evidence["run"]["pairwise"]:
        assert pair["argmax_disagreement_count"] == 0
        assert pair["threshold_disagreement_count"] == 0
        assert pair["max_abs_delta"] == 0.0


def test_predict_equals_training_false(
    predict_evidence: dict[str, Any], training_false_evidence: dict[str, Any]
) -> None:
    """The two canonical inference modes must produce identical decisions."""
    assert (
        predict_evidence["trainable_weight_hash_before"]
        == predict_evidence["trainable_weight_hash_after"]
    )
    assert (
        training_false_evidence["trainable_weight_hash_before"]
        == training_false_evidence["trainable_weight_hash_after"]
    )
    assert predict_evidence["fixture_shape"] == training_false_evidence["fixture_shape"]
    assert predict_evidence["fixture_dtype"] == training_false_evidence["fixture_dtype"]

    assert (
        predict_evidence["run"]["outputs_shape"] == training_false_evidence["run"]["outputs_shape"]
    )
    assert (
        predict_evidence["run"]["outputs_dtype"] == training_false_evidence["run"]["outputs_dtype"]
    )
    assert predict_evidence["run"]["nan_counts"] == [0, 0, 0]
    assert predict_evidence["run"]["inf_counts"] == [0, 0, 0]
    assert training_false_evidence["run"]["nan_counts"] == [0, 0, 0]
    assert training_false_evidence["run"]["inf_counts"] == [0, 0, 0]
