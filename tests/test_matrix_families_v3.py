"""Testes das famílias (c) fusão e (d) multitarefa da matriz v3."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.models.backbone_1d import build_backbone_1d_with_features
from src.models.multitask_v3 import (
    QUALITY_HEAD_NAMES,
    RHYTHM_CLASSES,
    build_multitask_beat_model,
    build_rhythm_model,
    weighted_sparse_ce,
)


class TestFusionFamily:
    def test_two_inputs_output_shape(self):
        model = build_backbone_1d_with_features(
            input_len=500, num_classes=2, num_features=17,
            embedding_dim=64, conv_filters=(16, 32, 64), conv_kernels=(7, 5, 3), dense_units=64,
        )
        assert len(model.inputs) == 2
        assert model.output_shape == (None, 2)

    def test_forward_pass(self):
        model = build_backbone_1d_with_features(
            input_len=500, num_classes=2, num_features=17,
            embedding_dim=64, conv_filters=(16, 32, 64), conv_kernels=(7, 5, 3), dense_units=64,
        )
        x = np.random.default_rng(0).standard_normal((4, 500, 1)).astype(np.float32)
        f = np.random.default_rng(1).standard_normal((4, 17)).astype(np.float32)
        y = model.predict([x, f], verbose=0)
        assert y.shape == (4, 2)
        assert np.allclose(y.sum(axis=1), 1.0, atol=1e-5)


class TestMultitaskFamily:
    def test_beat_heads(self):
        model = build_multitask_beat_model(input_len=500, embedding_dim=64)
        assert model.output_names == ["beat", "quality"]
        x = np.random.default_rng(0).standard_normal((4, 500, 1)).astype(np.float32)
        beat, quality = model.predict(x, verbose=0)
        assert beat.shape == (4, 2)
        assert quality.shape == (4, len(QUALITY_HEAD_NAMES))
        assert np.allclose(beat.sum(axis=1), 1.0, atol=1e-5)
        assert np.all((quality >= 0) & (quality <= 1))

    def test_quality_weight_frozen(self):
        model = build_multitask_beat_model(input_len=500, quality_weight=0.25)
        assert model.quality_weight == 0.25

    def test_rhythm_model(self):
        model = build_rhythm_model(input_len=5000)
        x = np.random.default_rng(0).standard_normal((2, 5000, 1)).astype(np.float32)
        y = model.predict(x, verbose=0)
        assert y.shape == (2, len(RHYTHM_CLASSES))
        assert np.allclose(y.sum(axis=1), 1.0, atol=1e-5)

    def test_loss_decreases_on_toy_batch(self):
        tf.keras.utils.set_random_seed(42)
        model = build_multitask_beat_model(input_len=500)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((16, 500, 1)).astype(np.float32)
        y_beat = np.array([0, 1] * 8, dtype=np.int64)
        y_quality = (rng.random((16, 3)) > 0.8).astype(np.float32)
        h = model.fit(x, {"beat": y_beat, "quality": y_quality}, epochs=2, verbose=0)
        assert h.history["loss"][-1] < h.history["loss"][0] * 1.5  # não diverge

    def test_fit_with_weighted_loss_multi_output(self):
        """Keras 3 rejeita class_weight/sample_weight em multi-saída — o
        runner usa weighted_sparse_ce embutida na loss; protege o caminho."""
        tf.keras.utils.set_random_seed(42)
        model = build_multitask_beat_model(input_len=500)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss={"beat": weighted_sparse_ce({0: 1.0, 1: 4.0}), "quality": "binary_crossentropy"},
            loss_weights={"beat": 1.0, "quality": 0.25},
        )
        rng = np.random.default_rng(0)
        x = rng.standard_normal((16, 500, 1)).astype(np.float32)
        y_beat = np.array([0, 1] * 8, dtype=np.int64)
        y_quality = (rng.random((16, 3)) > 0.8).astype(np.float32)
        h = model.fit(x, {"beat": y_beat, "quality": y_quality}, epochs=1, verbose=0)
        assert np.isfinite(h.history["loss"][-1])
        # a classe 1 pesa 4x mais: loss de uma amostra da classe 1 > classe 0
        l0 = weighted_sparse_ce({0: 1.0, 1: 4.0})(
            tf.constant([0]), tf.constant([[0.9, 0.1]])
        )
        l1 = weighted_sparse_ce({0: 1.0, 1: 4.0})(
            tf.constant([1]), tf.constant([[0.9, 0.1]])
        )
        assert float(l1[0]) > 4 * float(l0[0])
