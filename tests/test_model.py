"""Quality Gate QG4 — Arquitetura do modelo.

Validates:
* QG4.1 — Backbone 1D-CNN:
  Conv1D → MaxPool → Conv1D → MaxPool → Conv1D → MaxPool → GAP → Dense → Dropout → Dense
* QG4.2 — ~13K params, <20K limit
* QG4.3 — Sem BatchNorm, LSTM, SeparableConv, attention
* QG4.4 — Input shape (500, 1), output shape (5,)
* QG4.5 — TFLM constraints passam
* QG4.6 — Freeze/unfreeze conv layers funciona
* QG4.7 — Pré-treino (sigmoid) vs fine-tuning (softmax)
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.models.backbone_1d import (
    TFLMConstraints,
    build_backbone_1d,
    build_backbone_1d_multilabel,
    freeze_conv_layers,
    save_model_config,
    unfreeze_all,
)


@pytest.mark.qg4
class TestBackboneArchitecture:
    """Valida arquitetura do backbone."""

    def test_input_shape(self):
        model = build_backbone_1d(input_len=500, num_classes=5)
        assert model.input_shape == (None, 500, 1)

    def test_output_shape(self):
        model = build_backbone_1d(input_len=500, num_classes=5)
        assert model.output_shape == (None, 5)

    def test_num_params_under_limit(self):
        """QG4: ~13K params, must be < 20K."""
        model = build_backbone_1d(input_len=500, num_classes=5)
        info = TFLMConstraints.validate_model(model)
        assert info["total_params"] < 20_000, f"Params = {info['total_params']}"
        assert info["passes"] is True

    def test_no_batch_normalization(self):
        """QG4: Sem BatchNormalization (proibido em TFLM)."""
        model = build_backbone_1d(input_len=500, num_classes=5)
        layer_names = [layer.__class__.__name__ for layer in model.layers]
        assert "BatchNormalization" not in layer_names

    def test_no_lstm_gru_rnn(self):
        """QG4: Sem LSTM/GRU/RNN (alto SRAM, suporte limitado em TFLM)."""
        model = build_backbone_1d(input_len=500, num_classes=5)
        layer_names = [layer.__class__.__name__ for layer in model.layers]
        assert "LSTM" not in layer_names
        assert "GRU" not in layer_names
        assert "SimpleRNN" not in layer_names

    def test_no_separable_conv(self):
        """QG4: Sem SeparableConv1D (decomposição ruim em TFLM)."""
        model = build_backbone_1d(input_len=500, num_classes=5)
        layer_names = [layer.__class__.__name__ for layer in model.layers]
        assert "SeparableConv1D" not in layer_names

    def test_conv1d_layers_exist(self):
        model = build_backbone_1d(input_len=500, num_classes=5)
        conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv1D)]
        assert len(conv_layers) == 3
        assert conv_layers[0].filters == 16
        assert conv_layers[1].filters == 40
        assert conv_layers[2].filters == 80

    def test_maxpool_layers_exist(self):
        model = build_backbone_1d(input_len=500, num_classes=5)
        pool_layers = [
            layer for layer in model.layers if isinstance(layer, tf.keras.layers.MaxPooling1D)
        ]
        assert len(pool_layers) == 3

    def test_gap_exists(self):
        model = build_backbone_1d(input_len=500, num_classes=5)
        gap = [
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.layers.GlobalAveragePooling1D)
        ]
        assert len(gap) == 1

    def test_dense_layers(self):
        model = build_backbone_1d(input_len=500, num_classes=5)
        dense = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
        assert len(dense) == 2
        assert dense[0].units == 80  # embedding
        assert dense[1].units == 5  # classifier

    def test_dropout_rate(self):
        model = build_backbone_1d(input_len=500, num_classes=5, dropout_rate=0.3)
        dropout = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]
        assert len(dropout) == 1
        assert dropout[0].rate == 0.3

    def test_output_activation_softmax(self):
        model = build_backbone_1d(input_len=500, num_classes=5)
        assert model.layers[-1].activation == tf.keras.activations.softmax

    def test_scaled_backbone_passes_tflm_constraints(self):
        """QG4: variante maior deve caber em <64KB de FlatBuffer INT8."""
        model = build_backbone_1d(
            input_len=500,
            num_classes=5,
            conv_filters=[32, 64, 96],
            dense_units=96,
        )
        info = TFLMConstraints.validate_model(model)
        assert info["total_params"] > 20_000
        assert info["flatbuffer_kb_est"] <= TFLMConstraints.MAX_FLATBUFFER_KB, (
            f"Estimated FlatBuffer={info['flatbuffer_kb_est']}KB exceeds limit"
        )

    def test_backbone_with_features_passes_tflm_constraints(self):
        """QG4: backbone com features morfológicas adicionais cabe em <64KB."""
        from src.models.backbone_1d import build_backbone_1d_with_features

        model = build_backbone_1d_with_features(
            input_len=500,
            num_classes=2,
            num_features=2,
            conv_filters=[32, 64, 96],
            dense_units=96,
        )
        info = TFLMConstraints.validate_model(model)
        assert info["flatbuffer_kb_est"] <= TFLMConstraints.MAX_FLATBUFFER_KB, (
            f"Estimated FlatBuffer={info['flatbuffer_kb_est']}KB exceeds limit"
        )
        assert len(model.inputs) == 2

    def test_forward_pass(self):
        model = build_backbone_1d(input_len=500, num_classes=5)
        x = np.random.randn(4, 500, 1).astype(np.float32)
        y = model.predict(x, verbose=0)
        assert y.shape == (4, 5)
        assert np.allclose(y.sum(axis=1), 1.0)  # softmax sums to 1


@pytest.mark.qg4
class TestMultilabelBackbone:
    """Valida backbone para pré-treino multi-label."""

    def test_sigmoid_output(self):
        model = build_backbone_1d_multilabel(input_len=500, num_classes=5)
        assert model.layers[-1].activation == tf.keras.activations.sigmoid

    def test_multilabel_forward(self):
        model = build_backbone_1d_multilabel(input_len=500, num_classes=5)
        x = np.random.randn(2, 500, 1).astype(np.float32)
        y = model.predict(x, verbose=0)
        assert y.shape == (2, 5)
        assert np.all((y >= 0) & (y <= 1))  # sigmoid range


@pytest.mark.qg4
class TestFreezeUnfreeze:
    """Valida transfer learning (freeze convs)."""

    def test_freeze_conv_layers(self):
        model = build_backbone_1d(input_len=500, num_classes=5)
        model = freeze_conv_layers(model)

        for layer in model.layers:
            if layer.__class__.__name__ in ("Conv1D", "MaxPooling1D", "GlobalAveragePooling1D"):
                assert layer.trainable is False

    def test_classifier_remains_trainable(self):
        model = build_backbone_1d(input_len=500, num_classes=5)
        model = freeze_conv_layers(model)

        for layer in model.layers:
            if layer.__class__.__name__ == "Dense":
                assert layer.trainable is True

    def test_unfreeze_all(self):
        model = build_backbone_1d(input_len=500, num_classes=5)
        model = freeze_conv_layers(model)
        model = unfreeze_all(model)

        for layer in model.layers:
            assert layer.trainable is True

    def test_trainable_params_after_freeze(self):
        model = build_backbone_1d(input_len=500, num_classes=5)
        total_before = model.count_params()
        model = freeze_conv_layers(model)
        trainable_after = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        # Apenas Dense layers devem ser treináveis
        assert trainable_after < total_before
        assert trainable_after > 0


@pytest.mark.qg4
class TestModelConfig:
    """Valida persistência de config."""

    def test_save_config(self, tmp_path):
        model = build_backbone_1d(input_len=500, num_classes=5)
        config_path = tmp_path / "config.json"
        save_model_config(model, config_path, extra={"seed": 42})

        import json

        with config_path.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        assert cfg["name"] == "lewis_backbone"
        assert cfg["input_shape"] == [None, 500, 1]
        assert cfg["output_shape"] == [None, 5]
        assert cfg["total_params"] > 0
        assert cfg["seed"] == 42
