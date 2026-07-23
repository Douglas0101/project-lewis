"""Independent-lane equivalence tests for the Stage 1 Keras loader helper."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from src.models.keras_loader import inspect_loader_selection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "stage1_float32_v2.0.keras"
EXPECTED_MODEL_SHA256 = "cd5e2474f294d965d52662f80e12a21024d551749f8b9d787b9c80bd34dbc347"


def _run(command: list[str]) -> None:
    process = subprocess.run(command, text=True, capture_output=True)
    if process.returncode != 0:
        pytest.fail(process.stdout + process.stderr)


@pytest.fixture(scope="module")
def lane_evidence(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build one deterministic fixture and launch two independent subprocess lanes."""
    output_dir = tmp_path_factory.mktemp("stage1-loader-equivalence")
    fixture = output_dir / "loader_fixture.npz"
    manifest = output_dir / "loader_fixture_manifest.json"
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_stage1_loader_fixture.py"),
            "--output",
            str(fixture),
            "--manifest",
            str(manifest),
        ]
    )
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "audit_stage1_loader_equivalence.py"),
            "--fixture",
            str(fixture),
            "--output-dir",
            str(output_dir),
        ]
    )
    return output_dir


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid test evidence: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Test evidence is not an object: {path}")
    return value


@pytest.mark.requires_artifacts
def test_helper_selects_standalone_keras_safe_mode() -> None:
    """A keras.src artifact must never route to the tf_keras loader."""
    decision = inspect_loader_selection(MODEL_PATH, compile=False)

    assert decision.artifact_family_detected == "KERAS_3_STANDALONE"
    assert decision.selected_loader == "keras.saving.load_model"
    assert decision.safe_mode
    assert not decision.compile
    assert decision.custom_objects == []
    assert decision.model_sha256 == EXPECTED_MODEL_SHA256


def test_legacy_metadata_remains_recognizable(tmp_path: Path) -> None:
    """Synthetic legacy metadata must route separately without loading it."""
    archive = tmp_path / "legacy.keras"
    config = {
        "module": "tf_keras.src.engine.functional",
        "class_name": "Functional",
        "config": {},
    }
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("config.json", json.dumps(config))

    decision = inspect_loader_selection(archive, compile=False)

    assert decision.artifact_family_detected == "TF_KERAS_LEGACY"
    assert decision.selected_loader == "tf.keras.models.load_model"
    assert decision.safe_mode
    assert decision.model_sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()


@pytest.mark.requires_artifacts
def test_shared_fixture_is_hashed_and_policy_complete(lane_evidence: Path) -> None:
    """The shared batch must cover every deterministic selection role."""
    manifest = _read_json(lane_evidence / "loader_fixture_manifest.json")
    roles = {role for sample in manifest["samples"] for role in sample["roles"]}

    assert manifest["shape"][1:] == [500, 1]
    assert manifest["dtype"] == "float32"
    assert manifest["model_sha256"] == EXPECTED_MODEL_SHA256
    assert len(manifest["sha256"]) == 64
    assert len(manifest["array_sha256"]) == 64
    assert {
        "true_normal",
        "true_abnormal",
        "predicted_positive_highest_score",
        "predicted_negative_lowest_score",
        "immediately_below_threshold",
        "immediately_above_threshold",
        "score_quantile_p01",
        "score_quantile_p25",
        "score_quantile_p50",
        "score_quantile_p75",
        "score_quantile_p99",
    } <= roles


@pytest.mark.requires_artifacts
def test_lane_structures_weights_and_predictions_are_equivalent(lane_evidence: Path) -> None:
    """Independent loaders must produce identical numerical evidence."""
    reference = _read_json(lane_evidence / "reference_loader_result.json")
    helper = _read_json(lane_evidence / "helper_loader_result.json")
    comparison = _read_json(lane_evidence / "prediction_comparison.json")

    assert reference["fixture_sha256"] == helper["fixture_sha256"]
    assert reference["structure"] == helper["structure"]
    assert helper["selected_loader"] == "keras.saving.load_model"
    assert helper["safe_mode"]
    assert comparison["structural_equivalence"]
    assert comparison["all_weight_shapes_equal"]
    assert comparison["all_weight_dtypes_equal"]
    assert comparison["all_weights_array_equal"]
    assert comparison["max_abs_weight_delta"] == 0.0
    assert comparison["max_abs_prediction_delta"] == 0.0
    assert comparison["mean_abs_prediction_delta"] == 0.0
    assert comparison["p99_abs_prediction_delta"] == 0.0
    assert comparison["argmax_disagreement_count"] == 0
    assert comparison["threshold_0_58_disagreement_count"] == 0
    assert comparison["criteria_passed"]

    with (lane_evidence / "weight_comparison.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert all(row["array_equal"] == "True" for row in rows)


@pytest.mark.requires_artifacts
def test_compile_true_preserves_reference_predictions(lane_evidence: Path) -> None:
    """Restoring Adam and loss must not change inference outputs."""
    comparison = _read_json(lane_evidence / "prediction_comparison.json")

    assert comparison["compile_optimizer_restored"]
    assert comparison["compile_optimizer_class"] == "Adam"
    assert comparison["compile_loss"] == "sparse_categorical_crossentropy"
    assert comparison["compile_true_max_abs_prediction_delta"] == 0.0
    assert comparison["compile_true_argmax_disagreement_count"] == 0
    assert comparison["compile_true_threshold_disagreement_count"] == 0
