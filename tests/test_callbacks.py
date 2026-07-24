"""Testes unitários para callbacks de treinamento.

Cobrem GradientMonitor, CalibrationMonitor e F1MacroCheckpoint.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from sqlalchemy.orm import Session

from src.callbacks.calibration_monitor import CalibrationMonitor
from src.callbacks.f1_macro_checkpoint import F1MacroCheckpoint
from src.callbacks.gradient_monitor import GradientMonitor
from src.callbacks.metric_tracker import MetricTracker


def _build_toy_model(n_classes: int = 2, input_shape: tuple = (20, 1)) -> tf.keras.Model:
    """Constrói um modelo pequeno para uso nos testes."""
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(4, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(8, activation="relu", name="dense_1")(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="dense_out")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss="sparse_categorical_crossentropy",
    )
    return model


@pytest.fixture
def tmp_callback_dir():
    """Cria um diretório temporário para os logs dos callbacks."""
    tmpdir = tempfile.mkdtemp(prefix="callbacks_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def toy_model_binary():
    """Modelo binário para testes."""
    return _build_toy_model(n_classes=2)


@pytest.fixture
def toy_data_binary():
    """Dados sintéticos binários."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((32, 20, 1)).astype(np.float32)
    y = rng.integers(0, 2, size=(32,))
    return X, y


class TestGradientMonitor:
    """Testes para GradientMonitor."""

    def test_detects_dense_layers(self, toy_model_binary, tmp_callback_dir):
        """Deve detectar automaticamente as camadas Dense."""
        X, y = np.zeros((4, 20, 1), dtype=np.float32), np.zeros((4,), dtype=np.int32)
        log_path = os.path.join(tmp_callback_dir, "gradients.json")
        callback = GradientMonitor(val_data=X, val_labels=y, log_path=log_path)
        callback.set_model(toy_model_binary)
        callback.on_train_begin()

        assert callback.layer_names is not None
        assert "dense_1" in callback.layer_names
        assert "dense_out" in callback.layer_names

    def test_on_epoch_end_exports_json(self, toy_model_binary, toy_data_binary, tmp_callback_dir):
        """on_epoch_end deve computar estatísticas e exportar JSON."""
        X, y = toy_data_binary
        log_path = os.path.join(tmp_callback_dir, "gradients.json")
        callback = GradientMonitor(val_data=X, val_labels=y, log_path=log_path)
        callback.set_model(toy_model_binary)
        callback.on_train_begin()
        callback.on_epoch_end(epoch=0)

        assert os.path.exists(log_path)
        with open(log_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert len(data) == 1
        entry = data[0]
        assert entry["epoch"] == 0
        assert "layers" in entry
        assert len(entry["layers"]) >= 2

        layer_names = {layer["layer_name"] for layer in entry["layers"]}
        assert "dense_1" in layer_names
        assert "dense_out" in layer_names

    def test_gradient_stats_structure(self, toy_model_binary, toy_data_binary, tmp_callback_dir):
        """Estatísticas de gradiente devem conter as chaves esperadas."""
        X, y = toy_data_binary
        log_path = os.path.join(tmp_callback_dir, "gradients.json")
        callback = GradientMonitor(val_data=X, val_labels=y, log_path=log_path)
        callback.set_model(toy_model_binary)
        callback.on_train_begin()
        callback.on_epoch_end(epoch=1)

        with open(log_path, encoding="utf-8") as fh:
            data = json.load(fh)
        layer = data[0]["layers"][0]

        for key in [
            "layer_name",
            "l2_norm_mean",
            "weight_norm",
            "norm_ratio",
            "p95_gradient",
            "gradient_mean",
            "gradient_std",
            "gradient_mean_per_class",
        ]:
            assert key in layer

    def test_get_summary(self, toy_model_binary, toy_data_binary, tmp_callback_dir):
        """get_summary deve retornar resumo por camada monitorada."""
        X, y = toy_data_binary
        log_path = os.path.join(tmp_callback_dir, "gradients.json")
        callback = GradientMonitor(val_data=X, val_labels=y, log_path=log_path)
        callback.set_model(toy_model_binary)
        callback.on_train_begin()
        callback.on_epoch_end(epoch=0)
        callback.on_epoch_end(epoch=1)

        summary = callback.get_summary()
        assert "dense_1" in summary
        assert "dense_out" in summary
        for key in ["mean_norm", "mean_norm_ratio", "min_norm_ratio"]:
            assert key in summary["dense_1"]

    def test_max_samples_limits_dataset(self, toy_model_binary, tmp_callback_dir):
        """Deve amostrar apenas max_samples índices no início do treinamento."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((64, 20, 1)).astype(np.float32)
        y = rng.integers(0, 2, size=(64,))
        log_path = os.path.join(tmp_callback_dir, "gradients.json")

        callback = GradientMonitor(val_data=X, val_labels=y, log_path=log_path, max_samples=16)
        callback.set_model(toy_model_binary)
        callback.on_train_begin()

        assert callback.val_data.shape[0] == 16
        assert callback.val_labels.shape[0] == 16
        assert callback._sample_indices is not None
        assert callback._sample_indices.shape[0] == 16

    def test_max_samples_consistent_results(self, toy_model_binary, tmp_callback_dir):
        """Resultados com max_samples devem ser finitos e conter shapes esperados."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((64, 20, 1)).astype(np.float32)
        y = rng.integers(0, 2, size=(64,))
        log_path = os.path.join(tmp_callback_dir, "gradients.json")

        callback = GradientMonitor(val_data=X, val_labels=y, log_path=log_path, max_samples=16)
        callback.set_model(toy_model_binary)
        callback.on_train_begin()
        callback.on_epoch_end(epoch=0)

        with open(log_path, encoding="utf-8") as fh:
            data = json.load(fh)
        entry = data[0]
        assert entry["epoch"] == 0
        assert len(entry["layers"]) >= 1
        for layer in entry["layers"]:
            assert np.isfinite(layer["l2_norm_mean"])
            assert np.isfinite(layer["norm_ratio"])
            assert isinstance(layer["gradient_mean_per_class"], dict)

    def test_without_max_samples_uses_full_dataset(self, toy_model_binary, tmp_callback_dir):
        """Sem max_samples, o callback deve manter o dataset original."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((64, 20, 1)).astype(np.float32)
        y = rng.integers(0, 2, size=(64,))
        log_path = os.path.join(tmp_callback_dir, "gradients.json")

        callback = GradientMonitor(val_data=X, val_labels=y, log_path=log_path)
        callback.set_model(toy_model_binary)
        callback.on_train_begin()

        assert callback.val_data.shape[0] == 64
        assert callback.val_labels.shape[0] == 64
        assert callback._sample_indices is None

    def test_class_names_override_defaults(self, toy_model_binary, tmp_callback_dir):
        """class_names deve ser refletido no gradient_mean_per_class."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((16, 20, 1)).astype(np.float32)
        y = rng.integers(0, 2, size=(16,))
        log_path = os.path.join(tmp_callback_dir, "gradients.json")

        callback = GradientMonitor(
            val_data=X,
            val_labels=y,
            log_path=log_path,
            class_names=["N", "Anormal"],
        )
        callback.set_model(toy_model_binary)
        callback.on_train_begin()
        callback.on_epoch_end(epoch=0)

        with open(log_path, encoding="utf-8") as fh:
            data = json.load(fh)
        entry = data[0]["layers"][0]
        per_class = entry["gradient_mean_per_class"]
        assert "N" in per_class
        assert "Anormal" in per_class


class TestCalibrationMonitor:
    """Testes para CalibrationMonitor."""

    def test_on_epoch_end_exports_json(self, toy_model_binary, toy_data_binary, tmp_callback_dir):
        """on_epoch_end deve computar métricas de calibração e exportar JSON."""
        X, y = toy_data_binary
        log_path = os.path.join(tmp_callback_dir, "calibration.json")
        callback = CalibrationMonitor(
            val_data=X, val_labels=y, n_bins=5, log_path=log_path, class_names=["N", "Anormal"]
        )
        callback.set_model(toy_model_binary)
        callback.on_epoch_end(epoch=0)

        assert os.path.exists(log_path)
        with open(log_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert len(data) == 1
        entry = data[0]
        assert entry["epoch"] == 0
        for key in ["ece", "mce", "brier_score", "brier_per_class", "confidence_per_class"]:
            assert key in entry

    def test_reliability_bins_structure(self, toy_model_binary, toy_data_binary, tmp_callback_dir):
        """Reliability diagram deve conter bins com chaves esperadas."""
        X, y = toy_data_binary
        log_path = os.path.join(tmp_callback_dir, "calibration.json")
        callback = CalibrationMonitor(
            val_data=X, val_labels=y, n_bins=5, log_path=log_path, class_names=["N", "Anormal"]
        )
        callback.set_model(toy_model_binary)
        callback.on_epoch_end(epoch=0)

        with open(log_path, encoding="utf-8") as fh:
            data = json.load(fh)
        bins = data[0]["reliability_bins"]
        assert len(bins) == 5
        for b in bins:
            for key in ["bin", "lower_edge", "upper_edge", "accuracy", "confidence", "count"]:
                assert key in b

    def test_brier_per_class(self, toy_model_binary, toy_data_binary, tmp_callback_dir):
        """Brier score por classe deve conter nomes configurados."""
        X, y = toy_data_binary
        log_path = os.path.join(tmp_callback_dir, "calibration.json")
        callback = CalibrationMonitor(
            val_data=X, val_labels=y, n_bins=5, log_path=log_path, class_names=["N", "Anormal"]
        )
        callback.set_model(toy_model_binary)
        callback.on_epoch_end(epoch=0)

        with open(log_path, encoding="utf-8") as fh:
            data = json.load(fh)
        brier_per_class = data[0]["brier_per_class"]
        assert "N" in brier_per_class
        assert "Anormal" in brier_per_class

    def test_get_alert_summary(self, toy_model_binary, toy_data_binary, tmp_callback_dir):
        """Alertas devem ser emitidos quando calibração está ruim."""
        X, y = toy_data_binary
        log_path = os.path.join(tmp_callback_dir, "calibration.json")
        callback = CalibrationMonitor(
            val_data=X, val_labels=y, n_bins=5, log_path=log_path, class_names=["N", "Anormal"]
        )
        callback.set_model(toy_model_binary)
        # Injeta histórico artificial com ECE alto
        callback.history.append(
            {
                "epoch": 0,
                "ece": 0.25,
                "mce": 0.35,
                "brier_score": 0.6,
                "brier_per_class": {"N": 0.6, "Anormal": 0.4},
                "confidence_per_class": {},
                "reliability_bins": [],
            }
        )
        alerts = callback.get_alert_summary()
        assert any("ECE" in alert for alert in alerts)
        assert any("MCE" in alert for alert in alerts)

    def test_max_samples_limits_dataset(self, toy_model_binary, tmp_callback_dir):
        """Deve amostrar apenas max_samples índices no início do treinamento."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((64, 20, 1)).astype(np.float32)
        y = rng.integers(0, 2, size=(64,))
        log_path = os.path.join(tmp_callback_dir, "calibration.json")

        callback = CalibrationMonitor(
            val_data=X,
            val_labels=y,
            n_bins=5,
            log_path=log_path,
            class_names=["N", "Anormal"],
            max_samples=16,
        )
        callback.set_model(toy_model_binary)
        callback.on_train_begin()

        assert callback.val_data.shape[0] == 16
        assert callback.val_labels.shape[0] == 16
        assert callback._sample_indices is not None
        assert callback._sample_indices.shape[0] == 16

    def test_max_samples_consistent_results(self, toy_model_binary, tmp_callback_dir):
        """Resultados com max_samples devem ser finitos e conter métricas esperadas."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((64, 20, 1)).astype(np.float32)
        y = rng.integers(0, 2, size=(64,))
        log_path = os.path.join(tmp_callback_dir, "calibration.json")

        callback = CalibrationMonitor(
            val_data=X,
            val_labels=y,
            n_bins=5,
            log_path=log_path,
            class_names=["N", "Anormal"],
            max_samples=16,
        )
        callback.set_model(toy_model_binary)
        callback.on_train_begin()
        callback.on_epoch_end(epoch=0)

        with open(log_path, encoding="utf-8") as fh:
            data = json.load(fh)
        entry = data[0]
        assert entry["epoch"] == 0
        assert np.isfinite(entry["ece"])
        assert np.isfinite(entry["mce"])
        assert np.isfinite(entry["brier_score"])
        assert isinstance(entry["brier_per_class"], dict)

    def test_without_max_samples_uses_full_dataset(self, toy_model_binary, tmp_callback_dir):
        """Sem max_samples, o callback deve manter o dataset original."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((64, 20, 1)).astype(np.float32)
        y = rng.integers(0, 2, size=(64,))
        log_path = os.path.join(tmp_callback_dir, "calibration.json")

        callback = CalibrationMonitor(
            val_data=X, val_labels=y, n_bins=5, log_path=log_path, class_names=["N", "Anormal"]
        )
        callback.set_model(toy_model_binary)
        callback.on_train_begin()

        assert callback.val_data.shape[0] == 64
        assert callback.val_labels.shape[0] == 64
        assert callback._sample_indices is None


class TestF1MacroCheckpoint:
    """Testes para F1MacroCheckpoint."""

    def test_saves_best_weights(self, toy_model_binary, toy_data_binary, tmp_callback_dir):
        """Deve salvar pesos quando a métrica melhora."""
        X, y = toy_data_binary
        weights_path = Path(tmp_callback_dir) / "best.weights.h5"
        callback = F1MacroCheckpoint(
            X_val=X,
            y_val=y,
            filepath=weights_path,
            class_names=["N", "Anormal"],
            metric="F1_macro",
            patience=5,
        )
        callback.set_model(toy_model_binary)
        callback.on_train_begin()
        callback.on_epoch_end(epoch=0)

        assert weights_path.exists()

    def test_saves_threshold_json_for_binary(
        self, toy_model_binary, toy_data_binary, tmp_callback_dir
    ):
        """Em classificação binária deve salvar threshold JSON."""
        X, y = toy_data_binary
        weights_path = Path(tmp_callback_dir) / "best.weights.h5"
        callback = F1MacroCheckpoint(
            X_val=X,
            y_val=y,
            filepath=weights_path,
            class_names=["N", "Anormal"],
            metric="F1_macro",
            patience=5,
        )
        callback.set_model(toy_model_binary)
        callback.on_train_begin()
        callback.on_epoch_end(epoch=0)

        threshold_path = weights_path.with_suffix(".threshold.json")
        assert threshold_path.exists()
        with open(threshold_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert "threshold" in data
        assert isinstance(data["threshold"], float)

    def test_restores_best_weights_on_train_end(
        self, toy_model_binary, toy_data_binary, tmp_callback_dir
    ):
        """Deve restaurar os melhores pesos ao final do treinamento."""
        X, y = toy_data_binary
        weights_path = Path(tmp_callback_dir) / "best.weights.h5"
        callback = F1MacroCheckpoint(
            X_val=X,
            y_val=y,
            filepath=weights_path,
            class_names=["N", "Anormal"],
            metric="F1_macro",
            patience=5,
        )
        callback.set_model(toy_model_binary)
        callback.on_train_begin()
        callback.on_epoch_end(epoch=0)

        # Armazena pesos originais, altera o modelo e restaura
        original_weights = [w.numpy().copy() for w in toy_model_binary.weights]
        for w in toy_model_binary.weights:
            w.assign(w * 0.0)

        callback.on_train_end()

        restored_weights = [w.numpy() for w in toy_model_binary.weights]
        for orig, rest in zip(original_weights, restored_weights):
            np.testing.assert_allclose(orig, rest, atol=1e-6)

    def test_extract_score_per_class_metric(self, tmp_callback_dir):
        """Deve extrair corretamente métricas por classe."""
        result = {
            "global": {"F1_macro": 0.5},
            "per_class": {
                "Anormal": {"Se": 0.8, "F1": 0.7},
                "N": {"Se": 0.9, "F1": 0.85},
            },
        }
        callback = F1MacroCheckpoint(
            X_val=np.zeros((2, 20, 1)),
            y_val=np.zeros(2),
            filepath=Path(tmp_callback_dir) / "best.weights.h5",
            metric="Se_Anormal",
        )
        assert callback._extract_score(result) == pytest.approx(0.8)

        callback2 = F1MacroCheckpoint(
            X_val=np.zeros((2, 20, 1)),
            y_val=np.zeros(2),
            filepath=Path(tmp_callback_dir) / "best2.weights.h5",
            metric="F1_Anormal",
        )
        assert callback2._extract_score(result) == pytest.approx(0.7)

    def test_patience_stops_training(self, toy_model_binary, toy_data_binary, tmp_callback_dir):
        """Deve parar o treinamento após paciência sem melhora."""
        X, y = toy_data_binary
        weights_path = Path(tmp_callback_dir) / "best.weights.h5"
        callback = F1MacroCheckpoint(
            X_val=X,
            y_val=y,
            filepath=weights_path,
            class_names=["N", "Anormal"],
            metric="F1_macro",
            patience=2,
        )
        callback.set_model(toy_model_binary)
        callback.on_train_begin()
        callback.on_epoch_end(epoch=0)
        best_score = callback.best_score

        # Força métrica pior nas próximas épocas injetando pesos ruins
        for w in toy_model_binary.weights:
            w.assign(w * 0.0)

        callback.on_epoch_end(epoch=1)
        callback.on_epoch_end(epoch=2)

        assert callback.wait >= 2
        assert toy_model_binary.stop_training is True
        assert callback.best_score == best_score


class TestMetricTracker:
    """Testes para MetricTracker."""

    @pytest.fixture
    def tracking_session(self) -> Generator[Session, None, None]:
        """Sessão SQLAlchemy em banco in-memory para tracking."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from src.tracking.db import init_schema

        engine = create_engine("sqlite:///:memory:", future=True)
        init_schema(engine)
        with Session(engine) as session:
            yield session
            session.rollback()

    def test_logs_metrics_per_epoch(self, toy_model_binary, toy_data_binary, tracking_session):
        """Deve persistir métricas do logs por época no banco."""
        from src.tracking.repositories import ExperimentRepository, RunRepository
        from src.tracking.schemas import ExperimentCreate, RunCreate

        exp = ExperimentRepository(tracking_session).create(
            ExperimentCreate(
                name="metric_tracker_test", stage="stage1", config_path=None, git_commit=None
            )
        )
        run = RunRepository(tracking_session).create(
            RunCreate(experiment_id=exp.id, run_type="train")
        )
        run_id = run.id

        callback = MetricTracker(
            run_id=run_id,
            session_factory=lambda: tracking_session,
        )
        callback.set_model(toy_model_binary)
        callback.on_epoch_end(epoch=0, logs={"loss": 0.5, "accuracy": 0.8})
        callback.on_epoch_end(epoch=1, logs={"loss": 0.4, "accuracy": 0.85})
        callback.on_train_end()

        metrics = [
            m
            for m in tracking_session.query(
                __import__("src.tracking.models", fromlist=["Metric"]).Metric
            )
            .filter_by(run_id=run_id)
            .all()
        ]
        assert len(metrics) == 4
        steps = {m.step for m in metrics}
        assert steps == {1, 2}

    def test_ignores_non_numeric_values(self, toy_model_binary, toy_data_binary, tracking_session):
        """Deve ignorar valores não numéricos nos logs."""
        from src.tracking.repositories import ExperimentRepository, RunRepository
        from src.tracking.schemas import ExperimentCreate, RunCreate

        exp = ExperimentRepository(tracking_session).create(
            ExperimentCreate(
                name="metric_tracker_test2", stage="stage1", config_path=None, git_commit=None
            )
        )
        run = RunRepository(tracking_session).create(
            RunCreate(experiment_id=exp.id, run_type="train")
        )
        run_id = run.id

        callback = MetricTracker(
            run_id=run_id,
            session_factory=lambda: tracking_session,
        )
        callback.set_model(toy_model_binary)
        callback.on_epoch_end(epoch=0, logs={"loss": 0.5, "string_metric": "bad"})
        callback.on_train_end()

        Metric = __import__("src.tracking.models", fromlist=["Metric"]).Metric
        metrics = tracking_session.query(Metric).filter_by(run_id=run_id).all()
        assert len(metrics) == 1
        assert metrics[0].name == "loss"
