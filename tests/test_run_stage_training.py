"""Testes de regressão para scripts de treinamento two-stage.

Garantem que argumentos CLI críticos (ex.: ``--freeze-backbone``) sejam
propagados até ``train_group_kfold``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_summary():
    return {
        "best_fold": 0,
        "mean_metrics": {"F1_macro": 0.5, "Acc": 0.5, "MCC": 0.0},
        "std_metrics": {"F1_macro": 0.0, "Acc": 0.0, "MCC": 0.0},
        "passes_qg5": True,
    }


def _make_features():
    n = 40
    X = np.random.randn(n, 500, 1).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20, dtype=np.int64)
    df = pd.DataFrame(
        {
            "record_id": [f"rec_{i % 4}" for i in range(n)],
            "patient_id": [f"pat_{i % 4}" for i in range(n)],
            "label": y,
        }
    )
    return X, y, df


def _run_stage_script(script_module: str, extra_argv: list[str]):
    """Roda ``main`` do script com mocks para evitar treinamento real."""
    from scripts.run_stage1_training import main as stage1_main
    from scripts.run_stage2_training import main as stage2_main

    patched_cfg = {
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

    X, y, df = _make_features()
    summary = _make_summary()

    load_features_module = (
        "scripts.run_stage1_training._load_features"
        if script_module == "stage1"
        else "scripts.run_stage2_training._load_features"
    )
    train_module = (
        "scripts.run_stage1_training.train_group_kfold"
        if script_module == "stage1"
        else "scripts.run_stage2_training.train_group_kfold"
    )

    with patch("yaml.safe_load", return_value=patched_cfg), patch(
        load_features_module, return_value=(X, y, df)
    ), patch(train_module, return_value=summary) as mock_train, patch(
        "pathlib.Path.mkdir", MagicMock()
    ), patch(
        "pathlib.Path.open", MagicMock()
    ), patch(
        "json.dump", MagicMock()
    ), patch(
        "shutil.copy", MagicMock()
    ):
        argv = [
            "prog",
            "--config",
            "config/dummy.yaml",
            "--n-splits",
            "2",
            "--epochs",
            "1",
        ] + extra_argv
        with patch("sys.argv", argv):
            if script_module == "stage1":
                stage1_main()
            else:
                stage2_main()
            return mock_train


@pytest.mark.qg5
class TestRunStageTrainingFreezeBackbone:
    """Garante que ``--freeze-backbone`` chegue a ``train_group_kfold``."""

    def test_stage1_propagates_freeze_backbone_true(self):
        mock = _run_stage_script("stage1", ["--freeze-backbone"])
        kwargs = mock.call_args.kwargs
        assert kwargs["freeze_backbone"] is True

    def test_stage1_default_freeze_backbone_false(self):
        mock = _run_stage_script("stage1", [])
        kwargs = mock.call_args.kwargs
        assert kwargs["freeze_backbone"] is False

    def test_stage2_propagates_freeze_backbone_true(self):
        mock = _run_stage_script("stage2", ["--freeze-backbone"])
        kwargs = mock.call_args.kwargs
        assert kwargs["freeze_backbone"] is True

    def test_stage2_default_freeze_backbone_false(self):
        mock = _run_stage_script("stage2", [])
        kwargs = mock.call_args.kwargs
        assert kwargs["freeze_backbone"] is False
