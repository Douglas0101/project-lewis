"""Quality Gate QG5 — Fine-tuning MIT-BIH+.

Validates:
* QG5.1 — Transfer learning: convs congeladas, classifier treinável
* QG5.2 — sparse_categorical_crossentropy + softmax
* QG5.3 — Class weights balanceados
* QG5.4 — GroupKFold por paciente
* QG5.5 — Normalização z-score global (fit treino, transform teste)
* QG5.6 — Métricas AAMI EC57
"""

from __future__ import annotations

import numpy as np
import pytest

import tensorflow as tf

from src.models.backbone_1d import build_backbone_1d, freeze_conv_layers
from src.models.evaluate import evaluate_aami, evaluate_fold
from src.models.finetune_mitbih import SparseCategoricalFocalLoss
from src.models.train import _normalize_fold


@pytest.mark.qg5
class TestFineTuneSetup:
    """Valida configuração do fine-tuning."""

    def test_model_compiles_finetune(self):
        model = build_backbone_1d(input_len=500, num_classes=5)
        model = freeze_conv_layers(model)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        assert model.loss == "sparse_categorical_crossentropy"

    def test_frozen_convs(self):
        model = build_backbone_1d(input_len=500, num_classes=5)
        model = freeze_conv_layers(model)
        for layer in model.layers:
            if layer.__class__.__name__ == "Conv1D":
                assert layer.trainable is False


@pytest.mark.qg5
class TestNormalization:
    """Valida z-score global por fold."""

    def test_normalize_fold_shape(self):
        X_train = np.random.randn(50, 500, 1).astype(np.float32)
        X_test = np.random.randn(20, 500, 1).astype(np.float32)
        X_train_norm, X_test_norm, scaler = _normalize_fold(X_train, X_test)

        assert X_train_norm.shape == X_train.shape
        assert X_test_norm.shape == X_test.shape
        assert scaler is not None

    def test_normalize_fold_mean_std(self):
        X_train = np.random.randn(100, 500, 1).astype(np.float32) * 2.0 + 3.0
        X_test = np.random.randn(30, 500, 1).astype(np.float32) * 2.0 + 3.0
        X_train_norm, X_test_norm, scaler = _normalize_fold(X_train, X_test)

        # Treino deve ter média próxima de 0 e std próximo de 1
        assert abs(float(np.mean(X_train_norm))) < 0.1
        assert abs(float(np.std(X_train_norm)) - 1.0) < 0.1


@pytest.mark.qg5
class TestAAMIEvaluation:
    """Valida métricas AAMI EC57."""

    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        y_pred = y_true.copy()
        result = evaluate_aami(y_true, y_pred, class_names=["N", "S", "V", "F", "Q"])

        assert result["global"]["Acc"] == 1.0
        assert result["global"]["F1_macro"] == 1.0
        assert result["global"]["MCC"] == 1.0
        assert result["passes_qg5"] is True

    def test_all_wrong(self):
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1])
        result = evaluate_aami(y_true, y_pred, class_names=["N", "S", "V", "F", "Q"])

        assert result["global"]["Acc"] == 0.0
        assert result["global"]["F1_macro"] == 0.0
        assert result["passes_qg5"] is False

    def test_class_n_sensitivity(self):
        """QG5: Se(N) > 96%."""
        y_true = np.array([0] * 100 + [1] * 20)
        y_pred = np.array([0] * 97 + [1] * 3 + [1] * 20)  # 3 FN for N
        result = evaluate_aami(y_true, y_pred, class_names=["N", "S", "V", "F", "Q"])

        assert result["per_class"]["N"]["Se"] >= 0.96

    def test_fpr_global(self):
        """QG5: FPR global < 5%."""
        y_true = np.array([0] * 100 + [1] * 20)
        y_pred = np.array([0] * 95 + [1] * 5 + [1] * 20)  # 5 FP
        result = evaluate_aami(y_true, y_pred, class_names=["N", "S", "V", "F", "Q"])

        assert result["global"]["FPR_global"] < 0.05

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 2, 3, 4] * 10)
        y_pred = y_true.copy()
        result = evaluate_aami(y_true, y_pred, class_names=["N", "S", "V", "F", "Q"])

        cm = np.array(result["confusion_matrix"])
        assert cm.shape == (5, 5)

    def test_evaluate_fold(self):
        model = build_backbone_1d(input_len=500, num_classes=5)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        X_test = np.random.randn(20, 500, 1).astype(np.float32)
        y_test = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 2)
        result = evaluate_fold(model, X_test, y_test, class_names=["N", "S", "V", "F", "Q"])

        assert "per_class" in result
        assert "global" in result
        assert "y_pred" in result


@pytest.mark.qg5
class TestSparseCategoricalFocalLoss:
    """Valida focal loss para classes desbalanceadas."""

    def test_focal_loss_lower_for_correct_high_confidence(self):
        loss = SparseCategoricalFocalLoss(gamma=2.0)
        y_true = tf.constant([[0], [1], [2]], dtype=tf.int32)
        # Predições quase perfeitas
        y_pred = tf.constant(
            [
                [0.99, 0.005, 0.005],
                [0.005, 0.99, 0.005],
                [0.005, 0.005, 0.99],
            ],
            dtype=tf.float32,
        )
        loss_correct = float(tf.reduce_mean(loss(y_true, y_pred)))

        # Predições erradas (confidentes na classe errada)
        y_pred_wrong = tf.constant(
            [
                [0.005, 0.99, 0.005],
                [0.005, 0.005, 0.99],
                [0.99, 0.005, 0.005],
            ],
            dtype=tf.float32,
        )
        loss_wrong = float(tf.reduce_mean(loss(y_true, y_pred_wrong)))

        assert loss_correct < loss_wrong

    def test_focal_loss_with_alpha_scales_class(self):
        y_true = tf.constant([[0], [1]], dtype=tf.int32)
        y_pred = tf.constant(
            [
                [0.5, 0.5],
                [0.5, 0.5],
            ],
            dtype=tf.float32,
        )
        alpha = np.array([1.0, 2.0], dtype=np.float32)
        loss_weighted = SparseCategoricalFocalLoss(gamma=1.0, alpha=alpha)
        # Chamar call diretamente para obter per-sample loss (Loss.__call__ reduz)
        losses = loss_weighted.call(y_true, y_pred).numpy()
        # Classe 1 tem alpha dobrado => loss deve ser maior
        assert losses[1] > losses[0]

    def test_focal_loss_get_config_roundtrip(self):
        alpha = np.array([0.25, 0.75], dtype=np.float32)
        loss = SparseCategoricalFocalLoss(gamma=1.5, alpha=alpha, from_logits=True)
        config = loss.get_config()
        assert config["gamma"] == 1.5
        assert config["from_logits"] is True
        assert config["alpha"] == alpha.tolist()
