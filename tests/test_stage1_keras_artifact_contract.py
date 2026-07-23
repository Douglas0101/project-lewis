"""Structural contract tests for the immutable Stage 1 Keras archive."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.keras_artifact_inspector import inspect_keras_archive, sha256_file

pytestmark = pytest.mark.requires_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "stage1_float32_v2.0.keras"
EXPECTED_MODEL_SHA256 = "cd5e2474f294d965d52662f80e12a21024d551749f8b9d787b9c80bd34dbc347"


def test_stage1_keras_archive_contract() -> None:
    """The serialized artifact must retain its Keras 3 input/output contract."""
    inspection = inspect_keras_archive(MODEL_PATH)

    assert inspection.inspection_mode == "zip_only_no_model_deserialization"
    assert inspection.keras_family == "KERAS_3_STANDALONE"
    assert inspection.top_level_module == "keras.src.models.functional"
    assert inspection.top_level_class_name == "Functional"
    assert inspection.input_shape == [None, 500, 1]
    assert inspection.input_dtype == "float32"
    assert inspection.output_shape == [None, 2]
    assert inspection.output_units == 2
    assert inspection.output_activation == "softmax"
    assert inspection.output_domain == "probabilities_sum_to_one"


def test_stage1_keras_archive_layers_and_compile_contract() -> None:
    """Stateful and custom layers must be explicitly inventoried."""
    inspection = inspect_keras_archive(MODEL_PATH)

    assert inspection.layer_count == 11
    assert inspection.batch_normalization_layers == []
    assert inspection.dropout_layers == ["dropout"]
    assert inspection.lambda_layers == []
    assert inspection.custom_object_references == []
    assert inspection.compile_optimizer == "Adam"
    assert inspection.compile_loss == "sparse_categorical_crossentropy"
    assert inspection.compile_metrics == ["accuracy"]
    assert inspection.model_weight_count == 10
    assert inspection.model_parameter_count == 13218
    assert not inspection.label_mapping_serialized


def test_stage1_keras_archive_is_immutable_during_inspection() -> None:
    """ZIP inspection must preserve the exact model bytes."""
    before = sha256_file(MODEL_PATH)
    inspection = inspect_keras_archive(MODEL_PATH)
    after = sha256_file(MODEL_PATH)

    assert before == EXPECTED_MODEL_SHA256
    assert inspection.model_sha256_before == before
    assert inspection.model_sha256_after == before
    assert after == before
    assert [member.name for member in inspection.archive_members] == [
        "metadata.json",
        "config.json",
        "model.weights.h5",
    ]
