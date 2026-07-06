"""Fixtures e helpers compartilhados para testes dos scripts de treinamento."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def make_summary():
    """Retorna um summary mínimo para mocks de ``train_group_kfold``."""
    return {
        "best_fold": 0,
        "mean_metrics": {"F1_macro": 0.5, "Acc": 0.5, "MCC": 0.0},
        "std_metrics": {"F1_macro": 0.0, "Acc": 0.0, "MCC": 0.0},
        "passes_qg5": True,
    }


def make_toy_data(n_classes: int = 2, n_samples: int = 40):
    """Gera dados sintéticos pequenos para testes de integração."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, 500, 1)).astype(np.float32)
    y = rng.integers(0, n_classes, size=(n_samples,)).astype(np.int64)
    df = pd.DataFrame(
        {
            "record_id": [f"rec_{i % 4}" for i in range(n_samples)],
            "patient_id": [f"pat_{i % 4}" for i in range(n_samples)],
            "label": y,
        }
    )
    return X, y, df


def make_stage_config(with_instrumentation: bool = True):
    """Configuração mínima usada para mockar os scripts de treinamento."""
    cfg = {
        "dataset": {
            "feature_npz": "dummy.npz",
            "feature_parquet": "dummy.parquet",
            "max_class_weight": 10.0,
        },
        "training": {
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "monitor": "val_loss",
            "loss": "sparse_categorical_crossentropy",
        },
        "group_kfold": {"n_splits": 2, "seed": 42},
        "model": {
            "embedding_dim": 16,
            "conv_filters": [8, 16],
            "conv_kernels": [7, 5],
            "dense_units": 16,
        },
        "output": {
            "model_filename": "model.keras",
            "scaler_filename": "scaler.pkl",
        },
        "quality_gate": {
            "qg5_stage1": {
                "min_acc": 0.0,
                "min_f1_macro": 0.0,
                "min_mcc": 0.0,
                "max_fpr_global": 1.0,
                "recall_anormal": 0.0,
                "precision_anormal": 0.0,
            },
            "qg5_stage2": {
                "min_acc": 0.0,
                "min_f1_macro": 0.0,
                "min_mcc": 0.0,
                "max_fpr_global": 1.0,
                "f1": {"S": 0.0, "V": 0.0, "F": 0.0},
            },
        },
        "augmentation": {},
        "threshold_tuning": {"enabled": False},
    }
    if with_instrumentation:
        cfg["instrumentation"] = {
            "gradient_monitor": {
                "enabled": True,
                "log_path": "logs/gradients_stage1.json",
                "layer_names": None,
            },
            "calibration_monitor": {
                "enabled": True,
                "log_path": "logs/calibration_stage1.json",
                "n_bins": 10,
            },
        }
    return cfg


def run_stage_script(
    stage: str,
    extra_argv: list[str] | None = None,
    *,
    with_instrumentation: bool = True,
    patch_tracking: bool = True,
):
    """Roda ``main`` de um script de estágio com mocks para evitar treinamento real.

    Retorna o objeto mock de ``train_group_kfold`` para que o chamador possa
    inspecionar os kwargs repassados.
    """
    from scripts.run_stage1_training import main as stage1_main
    from scripts.run_stage2_training import main as stage2_main

    extra_argv = extra_argv or []
    cfg = make_stage_config(with_instrumentation=with_instrumentation)
    X, y, df = make_toy_data(n_classes=2 if stage == "stage1" else 3)
    summary = make_summary()

    load_features_module = (
        "scripts.run_stage1_training._load_features"
        if stage == "stage1"
        else "scripts.run_stage2_training._load_features"
    )
    train_module = (
        "scripts.run_stage1_training.train_group_kfold"
        if stage == "stage1"
        else "scripts.run_stage2_training.train_group_kfold"
    )

    argv = [
        "prog",
        "--config",
        "config/dummy.yaml",
        "--n-splits",
        "2",
        "--epochs",
        "1",
    ] + extra_argv

    with ExitStack() as stack:
        stack.enter_context(patch("yaml.safe_load", return_value=cfg))
        stack.enter_context(patch(load_features_module, return_value=(X, y, df)))
        mock_train = stack.enter_context(patch(train_module, return_value=summary))
        stack.enter_context(patch("pathlib.Path.mkdir", MagicMock()))
        stack.enter_context(patch("pathlib.Path.open", MagicMock()))
        stack.enter_context(patch("json.dump", MagicMock()))
        stack.enter_context(patch("shutil.copy", MagicMock()))
        if patch_tracking:
            stack.enter_context(
                patch("src.tracking.integrations.start_tracking_experiment", return_value=1)
            )
            stack.enter_context(patch("src.tracking.integrations.record_summary_metrics"))
            stack.enter_context(patch("src.tracking.integrations.finish_tracking_experiment"))
        stack.enter_context(patch("sys.argv", argv))

        if stage == "stage1":
            stage1_main()
        else:
            stage2_main()

    return mock_train.call_args.kwargs
