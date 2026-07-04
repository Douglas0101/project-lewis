"""Testes de integração para callbacks de instrumentação nos pipelines de treinamento.

Garantem que GradientMonitor e CalibrationMonitor sejam construídos dentro de
cada fold de ``train_group_kfold`` a partir de ``instrumentation_config``,
usando dados normalizados do fold atual, e que os scripts de estágio passem a
configuração correta. Verifica também que nenhum callback extra é criado quando
a configuração está ausente ou vazia.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.callbacks.calibration_monitor import CalibrationMonitor
from src.callbacks.gradient_monitor import GradientMonitor
from tests.helpers.stage_training import make_toy_data, run_stage_script


def _make_summary():
    return {
        "best_fold": 0,
        "mean_metrics": {"F1_macro": 0.5, "Acc": 0.5, "MCC": 0.0},
        "std_metrics": {"F1_macro": 0.0, "Acc": 0.0, "MCC": 0.0},
        "passes_qg5": True,
    }


class TestFinetuneMitbihExtraCallbacks:
    """Integração de callbacks extras em ``src.models.finetune_mitbih``."""

    def test_extra_callbacks_are_included_in_model_fit(self, tmp_path):
        """Callbacks extras devem aparecer na lista passada para ``model.fit``."""
        from src.models.finetune_mitbih import finetune_mitbih

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((16, 500, 1)).astype(np.float32)
        y_train = np.array([0] * 8 + [1] * 8, dtype=np.int64)
        x_val = rng.standard_normal((8, 500, 1)).astype(np.float32)
        y_val = np.array([0] * 4 + [1] * 4, dtype=np.int64)

        mock_model = MagicMock()
        mock_model.layers = []
        mock_model.fit.return_value = MagicMock(
            history={
                "loss": [0.5],
                "val_loss": [0.5],
                "accuracy": [0.5],
                "val_accuracy": [0.5],
            }
        )

        extra_callback = GradientMonitor(
            val_data=x_val, val_labels=y_val, log_path=str(tmp_path / "grad.json")
        )

        with patch("src.models.finetune_mitbih.save_model_config"):
            finetune_mitbih(
                model=mock_model,
                X_train=X_train,
                y_train=y_train,
                X_val=x_val,
                y_val=y_val,
                epochs=1,
                batch_size=8,
                extra_callbacks=[extra_callback],
            )

        fit_kwargs = mock_model.fit.call_args.kwargs
        passed_callbacks = fit_kwargs["callbacks"]
        assert extra_callback in passed_callbacks

    def test_no_extra_callbacks_keeps_default_callbacks(self, tmp_path):
        """Sem callbacks extras, apenas os callbacks padrão devem ser usados."""
        from src.models.finetune_mitbih import finetune_mitbih

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((16, 500, 1)).astype(np.float32)
        y_train = np.array([0] * 8 + [1] * 8, dtype=np.int64)
        x_val = rng.standard_normal((8, 500, 1)).astype(np.float32)
        y_val = np.array([0] * 4 + [1] * 4, dtype=np.int64)

        mock_model = MagicMock()
        mock_model.layers = []
        mock_model.fit.return_value = MagicMock(
            history={
                "loss": [0.5],
                "val_loss": [0.5],
                "accuracy": [0.5],
                "val_accuracy": [0.5],
            }
        )

        with patch("src.models.finetune_mitbih.save_model_config"):
            finetune_mitbih(
                model=mock_model,
                X_train=X_train,
                y_train=y_train,
                X_val=x_val,
                y_val=y_val,
                epochs=1,
                batch_size=8,
            )

        fit_kwargs = mock_model.fit.call_args.kwargs
        passed_callbacks = fit_kwargs["callbacks"]
        assert all(not isinstance(cb, GradientMonitor) for cb in passed_callbacks)
        assert all(not isinstance(cb, CalibrationMonitor) for cb in passed_callbacks)


class TestTrainGroupKfoldInstrumentation:
    """Criação de callbacks de instrumentação dentro de ``train_group_kfold``."""

    def test_instrumentation_config_creates_callbacks_per_fold(self, tmp_path):
        """``train_group_kfold`` deve instanciar GradientMonitor e
        CalibrationMonitor a partir do config."""
        from src.models.train import train_group_kfold

        X, y, _ = make_toy_data(n_classes=2, n_samples=20)
        groups = np.array([0, 0, 1, 1] * 5, dtype=np.int64)

        instrumentation_config = {
            "gradient_monitor": {
                "enabled": True,
                "log_path": str(tmp_path / "gradients_stage1.json"),
                "layer_names": None,
            },
            "calibration_monitor": {
                "enabled": True,
                "log_path": str(tmp_path / "calibration_stage1.json"),
                "n_bins": 10,
            },
        }

        with (
            patch("src.models.train.finetune_mitbih") as mock_finetune,
            patch("src.models.train.evaluate_fold") as mock_eval,
        ):
            mock_model = MagicMock()
            mock_finetune.return_value = (mock_model, {"loss": [0.5]})
            mock_eval.return_value = {
                "global": {"F1_macro": 0.5, "Acc": 0.5, "MCC": 0.0},
                "passes_qg5": True,
            }

            train_group_kfold(
                X=X,
                y=y,
                groups=groups,
                n_splits=2,
                epochs=1,
                batch_size=4,
                experiment_dir=tmp_path,
                instrumentation_config=instrumentation_config,
            )

            finetune_kwargs = mock_finetune.call_args.kwargs
            extras = finetune_kwargs.get("extra_callbacks", [])
            assert any(isinstance(cb, GradientMonitor) for cb in extras)
            assert any(isinstance(cb, CalibrationMonitor) for cb in extras)

            # Os dados passados para os callbacks devem estar normalizados (média ~0, std ~1)
            grad_monitor = next(cb for cb in extras if isinstance(cb, GradientMonitor))
            cal_monitor = next(cb for cb in extras if isinstance(cb, CalibrationMonitor))
            assert abs(float(np.mean(grad_monitor.val_data))) < 0.5
            assert abs(float(np.std(grad_monitor.val_data)) - 1.0) < 0.5
            assert np.allclose(grad_monitor.val_data, cal_monitor.val_data)

    def test_instrumentation_paths_include_fold(self, tmp_path):
        """Caminhos de log devem conter ``fold_{idx}`` após o path base do config."""
        from src.models.train import train_group_kfold

        X, y, _ = make_toy_data(n_classes=2, n_samples=20)
        groups = np.array([0, 0, 1, 1] * 5, dtype=np.int64)

        instrumentation_config = {
            "gradient_monitor": {
                "enabled": True,
                "log_path": str(tmp_path / "gradients_stage1.json"),
            },
            "calibration_monitor": {
                "enabled": True,
                "log_path": str(tmp_path / "calibration_stage1.json"),
            },
        }

        with (
            patch("src.models.train.finetune_mitbih") as mock_finetune,
            patch("src.models.train.evaluate_fold") as mock_eval,
        ):
            mock_model = MagicMock()
            mock_finetune.return_value = (mock_model, {"loss": [0.5]})
            mock_eval.return_value = {
                "global": {"F1_macro": 0.5, "Acc": 0.5, "MCC": 0.0},
                "passes_qg5": True,
            }

            train_group_kfold(
                X=X,
                y=y,
                groups=groups,
                n_splits=2,
                epochs=1,
                batch_size=4,
                experiment_dir=tmp_path,
                instrumentation_config=instrumentation_config,
            )

            calls = mock_finetune.call_args_list
            assert len(calls) == 2
            for fold_idx, call in enumerate(calls):
                extras = call.kwargs.get("extra_callbacks", [])
                grad_monitor = next(cb for cb in extras if isinstance(cb, GradientMonitor))
                cal_monitor = next(cb for cb in extras if isinstance(cb, CalibrationMonitor))
                assert f"fold_{fold_idx}" in str(grad_monitor.log_path)
                assert f"fold_{fold_idx}" in str(cal_monitor.log_path)

    def test_empty_instrumentation_config_passes_no_callbacks(self, tmp_path):
        """Configuração vazia/ausente não deve criar callbacks extras."""
        from src.models.train import train_group_kfold

        X, y, _ = make_toy_data(n_classes=2, n_samples=20)
        groups = np.array([0, 0, 1, 1] * 5, dtype=np.int64)

        with (
            patch("src.models.train.finetune_mitbih") as mock_finetune,
            patch("src.models.train.evaluate_fold") as mock_eval,
        ):
            mock_model = MagicMock()
            mock_finetune.return_value = (mock_model, {"loss": [0.5]})
            mock_eval.return_value = {
                "global": {"F1_macro": 0.5, "Acc": 0.5, "MCC": 0.0},
                "passes_qg5": True,
            }

            train_group_kfold(
                X=X,
                y=y,
                groups=groups,
                n_splits=2,
                epochs=1,
                batch_size=4,
                experiment_dir=tmp_path,
                instrumentation_config={},
            )

            extras = mock_finetune.call_args.kwargs.get("extra_callbacks", [])
            assert extras == []


class TestStageScriptsInstrumentation:
    """Scripts de treinamento repassam configuração de instrumentação."""

    def test_stage1_passes_instrumentation_config(self):
        """Estágio 1 deve passar ``instrumentation_config`` para ``train_group_kfold``."""
        kwargs = run_stage_script("stage1")
        assert "instrumentation_config" in kwargs
        assert kwargs["instrumentation_config"]["gradient_monitor"]["enabled"] is True
        assert kwargs["instrumentation_config"]["calibration_monitor"]["enabled"] is True

    def test_stage2_passes_instrumentation_config(self):
        """Estágio 2 deve passar ``instrumentation_config`` para ``train_group_kfold``."""
        kwargs = run_stage_script("stage2")
        assert "instrumentation_config" in kwargs
        assert kwargs["instrumentation_config"]["gradient_monitor"]["enabled"] is True
        assert kwargs["instrumentation_config"]["calibration_monitor"]["enabled"] is True

    def test_stage1_disabled_instrumentation_passes_none(self):
        """Sem configuração, estágio 1 deve passar ``instrumentation_config`` vazio ou None."""
        kwargs = run_stage_script("stage1", with_instrumentation=False)
        instr_cfg = kwargs.get("instrumentation_config")
        assert instr_cfg is None or instr_cfg == {}

    def test_stage2_disabled_instrumentation_passes_none(self):
        """Sem configuração, estágio 2 deve passar ``instrumentation_config`` vazio ou None."""
        kwargs = run_stage_script("stage2", with_instrumentation=False)
        instr_cfg = kwargs.get("instrumentation_config")
        assert instr_cfg is None or instr_cfg == {}
