"""Testes unitários para a arquitetura ECGClassifierV3."""

from __future__ import annotations

import pytest
import tensorflow as tf

from models.architecture_v3 import build_model


@pytest.fixture(scope="module")
def default_config() -> dict:
    # Configuração reduzida para manter o modelo abaixo de 150K parâmetros
    # enquanto preserva a topologia CNN + RNN + Dense do ECGClassifierV3.
    return {
        "cnn_filters": [24, 48, 72],
        "cnn_kernels": [7, 5, 3],
        "use_gru": False,
        "dense_units": [96, 48],
        "dropout": [0.5, 0.3],
        "learning_rate": 1e-3,
        "loss": "categorical_crossentropy",
    }


def test_v3_parameter_count_under_150k(default_config: dict) -> None:
    """Stage 1 e Stage 2 devem ter menos de 150K parâmetros treináveis."""
    for stage in (1, 2):
        model = build_model(stage=stage, config=default_config)
        total_params = model.count_params()
        assert (
            total_params < 150_000
        ), f"Stage {stage} excede limite de 150K params: {total_params:,}"
        tf.keras.backend.clear_session()


def test_v3_input_output_shape(default_config: dict) -> None:
    """Verifica shapes de entrada e saída para ambos os estágios."""
    model_stage1 = build_model(stage=1, config=default_config)
    assert model_stage1.input_shape == (None, 500, 1)
    assert model_stage1.output_shape == (None, 2)

    model_stage2 = build_model(stage=2, config=default_config)
    assert model_stage2.input_shape == (None, 500, 1)
    assert model_stage2.output_shape == (None, 3)


def _all_layer_names(model: tf.keras.Model) -> list[str]:
    """Retorna os nomes de todas as camadas, incluindo subcamadas."""
    names: list[str] = []
    for layer in model.layers:
        names.append(layer.name)
        if hasattr(layer, "layers"):
            names.extend(sub.name for sub in layer.layers)
        if hasattr(layer, "forward_layer"):
            names.append(layer.forward_layer.name)
        if hasattr(layer, "backward_layer"):
            names.append(layer.backward_layer.name)
    return names


def test_v3_gru_variant(default_config: dict) -> None:
    """A variante com GRU deve produzir um modelo válido e compilado."""
    config = {**default_config, "use_gru": True}
    model = build_model(stage=1, config=config)
    assert model is not None
    assert model.built
    assert any("gru" in name.lower() for name in _all_layer_names(model))
