"""Testes para pruning estruturado de canais e QAT.

Test-Driven Development (TDD): testes unitários para as funções de
``src/models/pruning_qat.py`` e para o script ``scripts/apply_pruning_qat.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.models.backbone_1d import build_backbone_1d
from src.models.pruning_qat import (
    apply_qat,
    apply_structured_pruning,
    convert_to_tflite_int8,
    export_quantization_params,
    fine_tune_pruned_model,
    get_conv_filter_norms,
    prune_qat_pipeline,
    strip_qat_wrappers,
)


@pytest.fixture
def tiny_model() -> tf.keras.Model:
    """Modelo pequeno para testes rápidos de pruning."""
    model = build_backbone_1d(
        input_len=64,
        num_classes=2,
        conv_filters=(8, 16, 32),
        conv_kernels=(3, 3, 3),
        dense_units=16,
        name="tiny_backbone",
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


@pytest.fixture
def tiny_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dados sintéticos pequenos para testes."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(32, 64, 1)).astype(np.float32)
    y_train = rng.integers(0, 2, size=32).astype(np.int64)
    X_val = rng.normal(size=(8, 64, 1)).astype(np.float32)
    y_val = rng.integers(0, 2, size=8).astype(np.int64)
    return X_train, y_train, X_val, y_val


def test_get_conv_filter_norms_shape(tiny_model: tf.keras.Model) -> None:
    """As normas devem ter o mesmo número de filtros de cada Conv1D."""
    norms = get_conv_filter_norms(tiny_model)
    conv_layers = [
        layer for layer in tiny_model.layers if isinstance(layer, tf.keras.layers.Conv1D)
    ]
    assert len(norms) == len(conv_layers)
    for layer in conv_layers:
        assert layer.name in norms
        assert norms[layer.name].shape == (layer.filters,)


def test_apply_structured_pruning_reduces_conv_filters(tiny_model: tf.keras.Model) -> None:
    """O pruning estruturado deve reduzir o número de filtros das Conv1D."""
    original_filters = [
        layer.filters for layer in tiny_model.layers if isinstance(layer, tf.keras.layers.Conv1D)
    ]
    pruned = apply_structured_pruning(tiny_model, target_sparsity=0.3)
    pruned_filters = [
        layer.filters for layer in pruned.layers if isinstance(layer, tf.keras.layers.Conv1D)
    ]
    assert len(pruned_filters) == len(original_filters)
    assert all(new <= old for new, old in zip(pruned_filters, original_filters))
    assert any(new < old for new, old in zip(pruned_filters, original_filters))


def test_apply_structured_pruning_keeps_architecture_valid(
    tiny_model: tf.keras.Model, tiny_data: tuple
) -> None:
    """Modelo podado deve propagar shape corretamente e gerar predições."""
    X_train, _, X_val, _ = tiny_data
    pruned = apply_structured_pruning(tiny_model, target_sparsity=0.3)
    pruned.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    proba = pruned.predict(X_val, verbose=0)
    assert proba.shape == (len(X_val), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_fine_tune_pruned_model(tiny_model: tf.keras.Model, tiny_data: tuple) -> None:
    """Fine-tuning pós-pruning deve retornar histórico não vazio."""
    X_train, y_train, X_val, y_val = tiny_data
    pruned = apply_structured_pruning(tiny_model, target_sparsity=0.3)
    model, history = fine_tune_pruned_model(
        pruned,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
    )
    assert isinstance(model, tf.keras.Model)
    assert "loss" in history
    assert len(history["loss"]) == 1


def test_apply_qat_and_strip(tiny_model: tf.keras.Model) -> None:
    """QAT deve retornar modelo e flag de aplicação; strip deve ser seguro."""
    qat_model, qat_applied = apply_qat(tiny_model)
    assert isinstance(qat_model, tf.keras.Model)
    assert isinstance(qat_applied, bool)

    if qat_applied:
        # Após QAT, as camadas devem conter wrappers de quantização.
        assert any("quant" in layer.__class__.__name__.lower() for layer in qat_model.layers)
    else:
        # Fallback: modelo original é devolvido sem wrappers de quantização.
        assert not any("quant" in layer.__class__.__name__.lower() for layer in qat_model.layers)

    stripped = strip_qat_wrappers(qat_model)
    assert isinstance(stripped, tf.keras.Model)


def test_apply_qat_fallback_when_tfmot_unavailable(
    tiny_model: tf.keras.Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Se tfmot não estiver disponível, apply_qat deve retornar fallback PTQ-safe."""
    monkeypatch.setattr(
        "src.models.pruning_qat._tfmot",
        None,
    )

    def _fake_get_tfmot() -> None:
        return None

    monkeypatch.setattr("src.models.pruning_qat._get_tfmot", _fake_get_tfmot)

    qat_model, qat_applied = apply_qat(tiny_model)
    assert qat_applied is False
    assert qat_model is tiny_model


def test_convert_to_tflite_int8(tiny_model: tf.keras.Model, tiny_data: tuple) -> None:
    """Conversão full-integer INT8 deve produzir bytes de FlatBuffer válidos."""
    _, _, X_val, _ = tiny_data

    def representative_dataset():
        for sample in X_val:
            yield [np.expand_dims(sample, axis=0)]

    tflite_bytes = convert_to_tflite_int8(tiny_model, representative_dataset)
    assert isinstance(tflite_bytes, bytes)
    assert len(tflite_bytes) > 0


def test_export_quantization_params(tiny_model: tf.keras.Model, tiny_data: tuple) -> None:
    """Deve exportar JSON com escalas e zero-points de entrada e saída."""
    _, _, X_val, _ = tiny_data

    def representative_dataset():
        for sample in X_val:
            yield [np.expand_dims(sample, axis=0)]

    tflite_bytes = convert_to_tflite_int8(tiny_model, representative_dataset)
    params = export_quantization_params(tflite_bytes)
    assert "input_scale" in params
    assert "input_zero_point" in params
    assert "output_scale" in params
    assert "output_zero_point" in params
    assert params["input_scale"] > 0.0
    assert params["output_scale"] > 0.0


def test_prune_qat_pipeline_end_to_end(tiny_data: tuple) -> None:
    """Pipeline completo deve gerar modelo .tflite e parâmetros JSON."""
    X_train, y_train, X_val, y_val = tiny_data
    model = build_backbone_1d(
        input_len=64,
        num_classes=2,
        conv_filters=(8, 16, 32),
        conv_kernels=(3, 3, 3),
        dense_units=16,
        name="pipeline_backbone",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.keras"
        output_dir = Path(tmpdir) / "out"
        model.save(str(model_path))

        result = prune_qat_pipeline(
            model_path=model_path,
            output_dir=output_dir,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            target_sparsity=0.3,
            fine_tune_epochs=1,
            batch_size=8,
            learning_rate=1e-3,
        )

        assert "tflite_path" in result
        assert "params_path" in result
        assert Path(result["tflite_path"]).exists()
        assert Path(result["params_path"]).exists()

        with open(result["params_path"], "r", encoding="utf-8") as fh:
            params = json.load(fh)
        assert "input_scale" in params
        assert "output_scale" in params


def test_prune_qat_pipeline_ptq_fallback_when_qat_fails(
    tiny_data: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pipeline deve gerar .tflite válido via PTQ quando QAT é rejeitado."""
    X_train, y_train, X_val, y_val = tiny_data
    model = build_backbone_1d(
        input_len=64,
        num_classes=2,
        conv_filters=(8, 16, 32),
        conv_kernels=(3, 3, 3),
        dense_units=16,
        name="ptq_fallback_backbone",
    )

    def _fake_apply_qat(m: tf.keras.Model) -> tuple[tf.keras.Model, bool]:
        return m, False

    monkeypatch.setattr("src.models.pruning_qat.apply_qat", _fake_apply_qat)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.keras"
        output_dir = Path(tmpdir) / "out"
        model.save(str(model_path))

        result = prune_qat_pipeline(
            model_path=model_path,
            output_dir=output_dir,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            target_sparsity=0.3,
            fine_tune_epochs=1,
            batch_size=8,
            learning_rate=1e-3,
        )

        assert result["qat_applied"] is False
        assert Path(result["tflite_path"]).exists()
        assert Path(result["params_path"]).exists()
        tflite_bytes = Path(result["tflite_path"]).read_bytes()
        assert len(tflite_bytes) > 0


def test_apply_pruning_qat_script_entrypoint(
    tiny_data: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O script CLI deve aceitar argumentos e gerar artefatos."""
    X_train, y_train, X_val, y_val = tiny_data
    model = build_backbone_1d(
        input_len=64,
        num_classes=2,
        conv_filters=(8, 16, 32),
        conv_kernels=(3, 3, 3),
        dense_units=16,
        name="script_backbone",
    )
    model_path = tmp_path / "model.keras"
    output_dir = tmp_path / "out"
    model.save(str(model_path))

    train_npz = tmp_path / "data.npz"
    np.savez(train_npz, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_pruning_qat.py",
            "--model",
            str(model_path),
            "--output-dir",
            str(output_dir),
            "--data",
            str(train_npz),
            "--target-sparsity",
            "0.3",
            "--fine-tune-epochs",
            "1",
            "--batch-size",
            "8",
        ],
    )

    from scripts.apply_pruning_qat import main

    assert main() == 0
    assert any(output_dir.glob("*.tflite"))
    assert any(output_dir.glob("*.json"))
