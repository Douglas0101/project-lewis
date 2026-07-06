"""Testes para scripts/analyze_training_dynamics.py.

Cobrem parsing de logs de treinamento, gradientes e calibração,
cálculo de correlações e geração de figuras e relatório.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.analyze_training_dynamics import (
    compute_correlations,
    generate_figures,
    generate_report,
    main,
    parse_calibration_log,
    parse_gradients_log,
    parse_training_log,
)


def _write_training_log(path, epochs=3):
    """Cria arquivo de log de treinamento fictício."""
    lines = []
    for i in range(1, epochs + 1):
        lines.append(f"Epoch {i}/{epochs}\n")
        lines.append(
            f" - loss: {0.5 - i * 0.05:.4f} - accuracy: {0.7 + i * 0.05:.4f} "
            f"- val_loss: {0.45 - i * 0.04:.4f} - val_accuracy: {0.72 + i * 0.04:.4f} "
            f"- val_f1_macro: {0.3 + i * 0.05:.4f}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _write_gradients_log(path, epochs=3):
    """Cria arquivo JSON de gradientes fictício."""
    data = []
    for i in range(epochs):
        data.append(
            {
                "epoch": i,
                "layers": [
                    {
                        "layer_name": "dense_1",
                        "norm_ratio": 0.01 + i * 0.001,
                        "p95_gradient": 0.1 + i * 0.01,
                    },
                    {
                        "layer_name": "dense_2",
                        "norm_ratio": 0.02 - i * 0.001,
                        "p95_gradient": 0.2 - i * 0.01,
                    },
                ],
            }
        )
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_calibration_log(path, epochs=3):
    """Cria arquivo JSON de calibração fictício."""
    data = []
    for i in range(epochs):
        brier_per_class = {
            "N": 0.1 + i * 0.01,
            "S": 0.2 - i * 0.01,
            "V": 0.15 + i * 0.005,
            "F": 0.3 - i * 0.02,
        }
        reliability_bins = []
        for b in range(5):
            reliability_bins.append(
                {
                    "bin": b,
                    "lower_edge": b * 0.2,
                    "upper_edge": (b + 1) * 0.2,
                    "accuracy": 0.5 + b * 0.1,
                    "confidence": 0.6 + b * 0.08,
                    "count": 10 + b,
                }
            )
        data.append(
            {
                "epoch": i,
                "ece": 0.2 - i * 0.03,
                "mce": 0.4 - i * 0.05,
                "brier_score": 0.25 - i * 0.02,
                "brier_per_class": brier_per_class,
                "confidence_per_class": {"N": 0.9, "S": 0.7, "V": 0.8, "F": 0.6},
                "reliability_bins": reliability_bins,
            }
        )
    path.write_text(json.dumps(data), encoding="utf-8")


class TestParseTrainingLog:
    """Testes para parse_training_log."""

    def test_parse_training_log_extracts_basic_metrics(self, tmp_path):
        log_path = tmp_path / "training.log"
        _write_training_log(log_path, epochs=3)

        parsed = parse_training_log(str(log_path))

        assert parsed["epochs"] == [1, 2, 3]
        assert len(parsed["train_loss"]) == 3
        assert len(parsed["val_acc"]) == 3
        assert len(parsed["val_f1_macro"]) == 3
        np.testing.assert_allclose(parsed["val_f1_macro"], [0.35, 0.40, 0.45], rtol=1e-3)

    def test_parse_training_log_returns_empty_when_no_epochs(self, tmp_path):
        log_path = tmp_path / "empty.log"
        log_path.write_text("Sem métricas aqui.\n", encoding="utf-8")

        parsed = parse_training_log(str(log_path))

        assert parsed["epochs"] == []
        assert parsed["train_loss"] == []
        assert parsed["val_f1_macro"] == []

    def test_parse_training_log_pads_f1_when_missing(self, tmp_path):
        log_path = tmp_path / "training_no_f1.log"
        lines = [
            "Epoch 1/1\n",
            " - loss: 0.5 - accuracy: 0.7 - val_loss: 0.4 - val_accuracy: 0.72\n",
        ]
        log_path.write_text("".join(lines), encoding="utf-8")

        parsed = parse_training_log(str(log_path))

        assert parsed["epochs"] == [1]
        assert parsed["val_f1_macro"] == [0.0]


class TestParseGradientsLog:
    """Testes para parse_gradients_log."""

    def test_parse_gradients_log_extracts_layers(self, tmp_path):
        log_path = tmp_path / "gradients.json"
        _write_gradients_log(log_path, epochs=3)

        parsed = parse_gradients_log(str(log_path))

        assert parsed["epochs"] == [0, 1, 2]
        assert set(parsed["norm_ratios"].keys()) == {"dense_1", "dense_2"}
        assert len(parsed["norm_ratios"]["dense_1"]) == 3
        assert len(parsed["p95_gradients"]["dense_2"]) == 3

    def test_parse_gradients_log_empty(self, tmp_path):
        log_path = tmp_path / "gradients_empty.json"
        log_path.write_text("[]", encoding="utf-8")

        parsed = parse_gradients_log(str(log_path))

        assert parsed["epochs"] == []
        assert parsed["norm_ratios"] == {}


class TestParseCalibrationLog:
    """Testes para parse_calibration_log."""

    def test_parse_calibration_log_extracts_metrics(self, tmp_path):
        log_path = tmp_path / "calibration.json"
        _write_calibration_log(log_path, epochs=3)

        parsed = parse_calibration_log(str(log_path))

        assert parsed["epochs"] == [0, 1, 2]
        assert len(parsed["ece"]) == 3
        assert len(parsed["brier_s"]) == 3
        np.testing.assert_allclose(parsed["brier_s"], [0.2, 0.19, 0.18], rtol=1e-3)

    def test_parse_calibration_log_keeps_reliability_bins(self, tmp_path):
        log_path = tmp_path / "calibration.json"
        _write_calibration_log(log_path, epochs=3)

        parsed = parse_calibration_log(str(log_path))

        assert "reliability_bins" in parsed
        assert len(parsed["reliability_bins"]) == 3
        assert len(parsed["reliability_bins"][0]) == 5

    def test_parse_calibration_log_missing_classes_are_zero(self, tmp_path):
        log_path = tmp_path / "calibration_partial.json"
        data = [
            {
                "epoch": 0,
                "ece": 0.1,
                "mce": 0.2,
                "brier_score": 0.3,
                "brier_per_class": {"N": 0.1},
            }
        ]
        log_path.write_text(json.dumps(data), encoding="utf-8")

        parsed = parse_calibration_log(str(log_path))

        assert parsed["brier_s"] == [0.0]
        assert parsed["brier_v"] == [0.0]


class TestComputeCorrelations:
    """Testes para compute_correlations."""

    def test_compute_correlations_basic(self):
        training = {
            "epochs": [1, 2, 3],
            "val_f1_macro": [0.3, 0.4, 0.5],
        }
        gradients = {
            "epochs": [0, 1, 2],
            "norm_ratios": {
                "dense_1": [0.01, 0.015, 0.02],
            },
        }
        calibration = {
            "epochs": [0, 1, 2],
            "ece": [0.2, 0.15, 0.1],
            "brier_s": [0.2, 0.19, 0.18],
            "brier_v": [0.3, 0.25, 0.2],
            "brier_f": [0.4, 0.35, 0.3],
        }

        correlations = compute_correlations(training, gradients, calibration)

        assert "norm_ratio_dense_1_vs_f1_macro" in correlations
        assert "ece_vs_f1_macro" in correlations
        assert "brier_S_vs_f1_macro" in correlations
        assert all(abs(v) <= 1.0 for v in correlations.values())

    def test_compute_corlations_with_constant_series(self):
        training = {
            "epochs": [1, 2, 3],
            "val_f1_macro": [0.4, 0.4, 0.4],
        }
        gradients = {
            "epochs": [0, 1, 2],
            "norm_ratios": {"dense_1": [0.01, 0.01, 0.01]},
        }
        calibration = {
            "epochs": [0, 1, 2],
            "ece": [0.1, 0.1, 0.1],
            "brier_s": [0.2, 0.2, 0.2],
            "brier_v": [0.2, 0.2, 0.2],
            "brier_f": [0.2, 0.2, 0.2],
        }

        correlations = compute_correlations(training, gradients, calibration)

        assert correlations == {}


class TestGenerateFigures:
    """Testes para generate_figures."""

    def test_generate_figures_creates_expected_files(self, tmp_path):
        training = {
            "epochs": [1, 2, 3],
            "val_f1_macro": [0.3, 0.4, 0.5],
        }
        gradients = {
            "epochs": [0, 1, 2],
            "norm_ratios": {"dense_1": [0.01, 0.015, 0.02]},
        }
        calibration = {
            "epochs": [0, 1, 2],
            "ece": [0.2, 0.15, 0.1],
            "brier_s": [0.2, 0.19, 0.18],
            "brier_v": [0.3, 0.25, 0.2],
            "brier_f": [0.4, 0.35, 0.3],
            "reliability_bins": [
                [
                    {"lower_edge": 0.0, "upper_edge": 0.2, "accuracy": 0.5, "confidence": 0.6},
                    {"lower_edge": 0.2, "upper_edge": 0.4, "accuracy": 0.6, "confidence": 0.7},
                ]
            ],
        }
        correlations = {"ece_vs_f1_macro": -0.9}

        generate_figures(training, gradients, calibration, correlations, str(tmp_path))

        assert (tmp_path / "correlation_heatmap.png").exists()
        assert (tmp_path / "ece_vs_f1_dual_axis.png").exists()
        assert (tmp_path / "reliability_diagram.png").exists()

    def test_generate_figures_skips_reliability_when_missing(self, tmp_path):
        training = {"epochs": [1], "val_f1_macro": [0.3]}
        gradients = {"epochs": [0], "norm_ratios": {}}
        calibration = {
            "epochs": [0],
            "ece": [0.1],
            "brier_s": [0.0],
            "brier_v": [0.0],
            "brier_f": [0.0],
        }
        correlations = {}

        generate_figures(training, gradients, calibration, correlations, str(tmp_path))

        assert (tmp_path / "correlation_heatmap.png").exists()
        assert (tmp_path / "ece_vs_f1_dual_axis.png").exists()
        assert not (tmp_path / "reliability_diagram.png").exists()


class TestGenerateReport:
    """Testes para generate_report."""

    def test_generate_report_creates_markdown(self, tmp_path):
        report_path = tmp_path / "report.md"
        training = {"epochs": [1, 2, 3], "val_f1_macro": [0.3, 0.4, 0.5]}
        gradients = {
            "epochs": [0, 1, 2],
            "norm_ratios": {"dense_1": [0.01, 0.015, 0.02]},
            "p95_gradients": {"dense_1": [0.1, 0.2, 0.3]},
        }
        calibration = {
            "epochs": [0, 1, 2],
            "ece": [0.2, 0.15, 0.1],
            "mce": [0.4, 0.35, 0.3],
        }
        correlations = {"ece_vs_f1_macro": -0.85}

        generate_report(training, gradients, calibration, correlations, str(report_path))

        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "# Análise de Dinâmica de Treinamento" in content
        assert "ece_vs_f1_macro" in content

    def test_generate_report_detects_vanishing_gradient(self, tmp_path):
        report_path = tmp_path / "report.md"
        training = {"epochs": [1], "val_f1_macro": [0.3]}
        gradients = {
            "epochs": [0],
            "norm_ratios": {"dense_1": [1e-7]},
            "p95_gradients": {"dense_1": [0.1]},
        }
        calibration = {"epochs": [0], "ece": [0.1], "mce": [0.2]}
        correlations = {}

        generate_report(training, gradients, calibration, correlations, str(report_path))

        content = report_path.read_text(encoding="utf-8")
        assert "GRADIENTE VANISHING" in content


class TestMain:
    """Testes de integração para a função main."""

    def test_main_end_to_end(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        training_log = tmp_path / "training.log"
        gradients_log = tmp_path / "gradients.json"
        calibration_log = tmp_path / "calibration.json"
        output_dir = tmp_path / "figures"
        report_path = tmp_path / "report.md"

        _write_training_log(training_log, epochs=3)
        _write_gradients_log(gradients_log, epochs=3)
        _write_calibration_log(calibration_log, epochs=3)

        args = SimpleNamespace(
            training_log=str(training_log),
            gradients_log=str(gradients_log),
            calibration_log=str(calibration_log),
            output_dir=str(output_dir),
            report_path=str(report_path),
        )

        monkeypatch.setattr(
            "scripts.analyze_training_dynamics.argparse.ArgumentParser.parse_args",
            lambda self: args,
        )

        main()

        assert output_dir.exists()
        assert (output_dir / "correlation_heatmap.png").exists()
        assert (output_dir / "ece_vs_f1_dual_axis.png").exists()
        assert (output_dir / "reliability_diagram.png").exists()
        assert report_path.exists()

    def test_main_missing_file_raises(self, tmp_path, monkeypatch):
        class Args:
            training_log = str(tmp_path / "nao_existe.log")
            gradients_log = str(tmp_path / "nao_existe.json")
            calibration_log = str(tmp_path / "nao_existe.json")
            output_dir = str(tmp_path / "figures")
            report_path = str(tmp_path / "report.md")

        monkeypatch.setattr(
            "scripts.analyze_training_dynamics.argparse.ArgumentParser.parse_args",
            lambda self: Args(),
        )

        with pytest.raises(FileNotFoundError):
            main()
