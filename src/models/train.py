"""Treinamento com GroupKFold por paciente (inter-patient).

NUNCA misturar batimentos do mesmo paciente entre treino e teste.
Fit scaler no treino apenas; carregar backbone pré-treinado; congelar convs.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.callbacks.calibration_monitor import CalibrationMonitor
from src.callbacks.gradient_monitor import GradientMonitor
from src.callbacks.metric_tracker import MetricTracker
from src.tracking.integrations import (
    finish_tracking_run,
    start_tracking_run,
)

from .backbone_1d import build_backbone_1d
from .evaluate import evaluate_fold
from .finetune_mitbih import finetune_mitbih

LOGGER = logging.getLogger("lewis.camada04.train")


def _set_seeds(seed: int = 42) -> None:
    """Fixa seeds para reprodutibilidade."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


@dataclass
class TrainingConfig:
    """Hiper-parâmetros de treinamento."""

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    seed: int = 42
    monitor: str = "val_loss"
    selection_metric: str = "F1_macro"
    loss: Union[str, tf.keras.losses.Loss] = "sparse_categorical_crossentropy"
    optimize_thresholds: bool = False


@dataclass
class AugmentationConfig:
    """Configuração de augmentação/over-sampling."""

    augment_class: Optional[int] = None
    augment_factor: int = 1
    augment_config: Optional[Dict[str, Any]] = None


@dataclass
class TrackingConfig:
    """Configuração de tracking de experimentos."""

    tracking_experiment_id: Optional[int] = None
    tracking_stage_label: str = ""


@dataclass
class ModelConfig:
    """Configuração do modelo e backbone."""

    backbone_weights: Optional[Path] = None
    freeze_backbone: bool = True
    model_builder: Optional[Any] = None
    normalize: bool = True


def _normalize_fold(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Normalização z-score global: fit no treino, transform em teste.

    Parameters
    ----------
    x_train : np.ndarray
        Shape (n_train, 500, 1).
    x_test : np.ndarray
        Shape (n_test, 500, 1).

    Returns
    -------
    tuple
        (x_train_norm, x_test_norm, scaler)
    """
    scaler = StandardScaler()
    n_train, seq_len, channels = x_train.shape
    n_test = x_test.shape[0]

    # Fit no treino (reshape para 2D)
    x_train_2d = x_train.reshape(-1, channels)
    scaler.fit(x_train_2d)

    # Transform treino e teste
    x_train_norm = scaler.transform(x_train_2d).reshape(n_train, seq_len, channels)
    x_test_norm = scaler.transform(x_test.reshape(-1, channels)).reshape(n_test, seq_len, channels)

    return x_train_norm, x_test_norm, scaler


def _build_instrumentation_callbacks(
    instrumentation_config: Optional[Dict[str, Any]],
    x_val_norm: np.ndarray,
    y_val: np.ndarray,
    class_names: Optional[List[str]] = None,
    fold_idx: int = 0,
) -> List[tf.keras.callbacks.Callback]:
    """Constrói callbacks de instrumentação para um fold específico.

    Os callbacks são instanciados APÓS a normalização do fold, garantindo que
    as estatísticas de gradiente e calibração sejam computadas nos mesmos dados
    em escala de treino. Os caminhos de log recebem o sufixo ``fold_{idx}``.

    Parameters
    ----------
    instrumentation_config : dict, optional
        Configuração com as chaves ``gradient_monitor`` e ``calibration_monitor``.
    x_val_norm : np.ndarray
        Dados de validação normalizados do fold atual.
    y_val : np.ndarray
        Labels de validação do fold atual.
    class_names : list[str], optional
        Nomes das classes para o CalibrationMonitor.
    fold_idx : int
        Índice do fold atual.

    Returns
    -------
    list
        Lista de callbacks extras (vazia se config ausente, vazia ou desabilitada).
    """
    if not instrumentation_config:
        return []

    callbacks: List[tf.keras.callbacks.Callback] = []
    grad_cfg = instrumentation_config.get("gradient_monitor", {})
    if grad_cfg.get("enabled", False):
        base_path = Path(grad_cfg.get("log_path", "logs/gradients_stage1.json"))
        fold_path = base_path.parent / f"fold_{fold_idx}" / base_path.name
        callbacks.append(
            GradientMonitor(
                val_data=x_val_norm,
                val_labels=y_val,
                log_path=str(fold_path),
                layer_names=grad_cfg.get("layer_names"),
                max_samples=grad_cfg.get("max_samples"),
                class_names=class_names,
            )
        )
        LOGGER.info(
            "GradientMonitor habilitado | fold=%d | log_path=%s",
            fold_idx,
            fold_path,
        )

    cal_cfg = instrumentation_config.get("calibration_monitor", {})
    if cal_cfg.get("enabled", False):
        base_path = Path(cal_cfg.get("log_path", "logs/calibration_stage1.json"))
        fold_path = base_path.parent / f"fold_{fold_idx}" / base_path.name
        callbacks.append(
            CalibrationMonitor(
                val_data=x_val_norm,
                val_labels=y_val,
                n_bins=cal_cfg.get("n_bins", 15),
                log_path=str(fold_path),
                class_names=class_names,
                max_samples=cal_cfg.get("max_samples"),
            )
        )
        LOGGER.info(
            "CalibrationMonitor habilitado | fold=%d | log_path=%s | n_bins=%d",
            fold_idx,
            fold_path,
            cal_cfg.get("n_bins", 15),
        )

    return callbacks


def _normalize_or_identity(
    x_train: np.ndarray,
    x_test: np.ndarray,
    normalize: bool,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Aplica z-score quando normalize=True; caso contrário, retorna identidade."""
    if normalize:
        return _normalize_fold(x_train, x_test)
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(x_train.shape[-1], dtype=np.float32)
    scaler.scale_ = np.ones(x_train.shape[-1], dtype=np.float32)
    return x_train, x_test, scaler


def _build_model(
    x: np.ndarray,
    y: np.ndarray,
    model_config: ModelConfig,
) -> tf.keras.Model:
    """Constrói o modelo e carrega pesos do backbone quando necessário."""
    num_classes = len(np.unique(y))
    if model_config.model_builder is None:
        model = build_backbone_1d(input_len=x.shape[1], num_classes=num_classes)
    else:
        model = model_config.model_builder(input_len=x.shape[1], num_classes=num_classes)
    if model_config.backbone_weights is not None:
        model.load_weights(str(model_config.backbone_weights))
        LOGGER.info("  Backbone weights loaded from %s", model_config.backbone_weights)
    return model


def _log_freeze_status(freeze_backbone: bool) -> None:
    """Loga a estratégia de congelamento do backbone."""
    if freeze_backbone:
        LOGGER.info("  Freezing conv layers for transfer learning")
    else:
        LOGGER.info("  Training all layers (no backbone freezing)")


def _start_tracking(
    tracking_config: TrackingConfig,
    fold_idx: int,
    fold_dir: Path,
) -> Optional[int]:
    """Inicia run de tracking quando configurado."""
    if tracking_config.tracking_experiment_id is None:
        return None
    try:
        return start_tracking_run(
            experiment_id=tracking_config.tracking_experiment_id,
            run_type="train",
            fold_idx=fold_idx,
            artifact_dir=fold_dir,
        )
    except Exception:
        LOGGER.exception("Falha ao criar run de tracking para fold %d", fold_idx)
        return None


def _finish_tracking(
    tracking_config: TrackingConfig,
    fold_run_id: Optional[int],
    eval_result: Dict[str, Any],
    fold_idx: int,
) -> None:
    """Finaliza run de tracking quando configurado."""
    if fold_run_id is None or tracking_config.tracking_experiment_id is None:
        return
    try:
        finish_tracking_run(
            run_id=fold_run_id,
            status="completed",
            eval_result=eval_result,
            experiment_id=tracking_config.tracking_experiment_id,
            stage_label=tracking_config.tracking_stage_label,
        )
    except Exception:
        LOGGER.exception("Falha ao finalizar run do fold %d no tracking", fold_idx)


def _train_single_fold(
    fold_idx: int,
    n_splits: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    experiment_dir: Path,
    class_names: Optional[List[str]],
    thresholds: Optional[Dict[str, Any]],
    class_weight: Optional[Dict[int, float]],
    instrumentation_config: Optional[Dict[str, Any]],
    training_config: TrainingConfig,
    augmentation_config: AugmentationConfig,
    tracking_config: TrackingConfig,
    model_config: ModelConfig,
) -> Tuple[Dict[str, Any], float]:
    """Treina e avalia um único fold."""
    LOGGER.info("=== Fold %d/%d ===", fold_idx + 1, n_splits)
    fold_dir = experiment_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    LOGGER.info(
        "  Train: n=%d | Test: n=%d | Patients train=%d | Patients test=%d",
        len(x_train),
        len(x_test),
        len(np.unique(groups[train_idx])),
        len(np.unique(groups[test_idx])),
    )

    x_train_norm, x_test_norm, scaler = _normalize_or_identity(
        x_train, x_test, model_config.normalize
    )

    fold_run_id = _start_tracking(tracking_config, fold_idx, fold_dir)

    fold_callbacks = _build_instrumentation_callbacks(
        instrumentation_config=instrumentation_config,
        x_val_norm=x_test_norm,
        y_val=y_test,
        class_names=class_names,
        fold_idx=fold_idx,
    )
    if fold_run_id is not None:
        fold_callbacks.append(MetricTracker(run_id=fold_run_id))

    import joblib

    joblib.dump(scaler, fold_dir / "input_scaler.pkl")

    model = _build_model(x, y, model_config)
    _log_freeze_status(model_config.freeze_backbone)

    model, history = finetune_mitbih(
        model=model,
        X_train=x_train_norm,
        y_train=y_train,
        X_val=x_test_norm,
        y_val=y_test,
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
        seed=training_config.seed,
        experiment_dir=fold_dir,
        monitor=training_config.monitor,
        freeze_backbone=model_config.freeze_backbone,
        class_names=class_names,
        thresholds=thresholds,
        class_weight=class_weight,
        selection_metric=training_config.selection_metric,
        augment_class=augmentation_config.augment_class,
        augment_factor=augmentation_config.augment_factor,
        augment_config=augmentation_config.augment_config,
        loss=training_config.loss,
        optimize_thresholds=training_config.optimize_thresholds,
        extra_callbacks=fold_callbacks,
    )

    eval_result = evaluate_fold(
        model,
        x_test_norm,
        y_test,
        class_names=class_names,
        thresholds=thresholds,
        optimize_thresholds=training_config.optimize_thresholds,
    )
    eval_result["fold"] = fold_idx
    eval_result["history"] = history

    _finish_tracking(tracking_config, fold_run_id, eval_result, fold_idx)

    f1_macro = eval_result["global"]["F1_macro"]
    LOGGER.info(
        "  Fold %d | F1-macro=%.4f | Acc=%.4f | MCC=%.4f",
        fold_idx,
        f1_macro,
        eval_result["global"]["Acc"],
        eval_result["global"]["MCC"],
    )
    return eval_result, f1_macro


def _write_summary(
    experiment_dir: Path,
    fold_results: List[Dict[str, Any]],
    best_fold: int,
) -> Dict[str, Any]:
    """Calcula métricas agregadas e persiste summary.json."""
    f1_macros = [r["global"]["F1_macro"] for r in fold_results]
    accs = [r["global"]["Acc"] for r in fold_results]
    mccs = [r["global"]["MCC"] for r in fold_results]

    mean_metrics: Dict[str, float] = {
        "F1_macro": round(float(np.mean(f1_macros)), 4),
        "Acc": round(float(np.mean(accs)), 4),
        "MCC": round(float(np.mean(mccs)), 4),
    }
    std_metrics: Dict[str, float] = {
        "F1_macro": round(float(np.std(f1_macros)), 4),
        "Acc": round(float(np.std(accs)), 4),
        "MCC": round(float(np.std(mccs)), 4),
    }
    summary: Dict[str, Any] = {
        "folds": fold_results,
        "best_fold": best_fold,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "passes_qg5": all(r["passes_qg5"] for r in fold_results),
    }

    with (experiment_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    LOGGER.info(
        "GroupKFold completo | Best fold=%d | Mean F1-macro=%.4f ± %.4f | Mean Acc=%.4f ± %.4f",
        best_fold,
        summary["mean_metrics"]["F1_macro"],
        summary["std_metrics"]["F1_macro"],
        summary["mean_metrics"]["Acc"],
        summary["std_metrics"]["Acc"],
    )
    return summary


def train_group_kfold(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    experiment_dir: Optional[Path] = None,
    class_names: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    class_weight: Optional[Dict[int, float]] = None,
    instrumentation_config: Optional[Dict[str, Any]] = None,
    training_config: Optional[TrainingConfig] = None,
    augmentation_config: Optional[AugmentationConfig] = None,
    tracking_config: Optional[TrackingConfig] = None,
    model_config: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    """Treinamento GroupKFold por paciente.

    Parameters
    ----------
    x : np.ndarray
        Dados (shape: (n, 500, 1)).
    y : np.ndarray
        Labels inteiros (shape: (n,)).
    groups : np.ndarray
        IDs de paciente (shape: (n,)).
    n_splits : int
        Número de folds.
    experiment_dir : Path, optional
        Diretório raiz dos experimentos.
    class_names : list[str], optional
        Nomes das classes para avaliação AAMI.
    thresholds : dict, optional
        Thresholds configuráveis para ``evaluate_aami``.
    class_weight : dict, optional
        Pesos por classe para ``model.fit``.
    instrumentation_config : dict, optional
        Configuração para callbacks de instrumentação.
    training_config : TrainingConfig, optional
        Hiper-parâmetros de treinamento.
    augmentation_config : AugmentationConfig, optional
        Configuração de augmentação/over-sampling.
    tracking_config : TrackingConfig, optional
        Configuração de tracking de experimentos.
    model_config : ModelConfig, optional
        Configuração do modelo e backbone.

    Returns
    -------
    dict
        {
            "folds": [resultados por fold],
            "best_fold": índice do melhor fold,
            "mean_metrics": médias,
            "std_metrics": desvios,
        }
    """
    training_config = training_config or TrainingConfig()
    augmentation_config = augmentation_config or AugmentationConfig()
    tracking_config = tracking_config or TrackingConfig()
    model_config = model_config or ModelConfig()

    _set_seeds(training_config.seed)

    if experiment_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path("experiments") / f"exp_{ts}_groupkfold"
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "GroupKFold | n_splits=%d | n_samples=%d | n_patients=%d",
        n_splits,
        len(x),
        len(np.unique(groups)),
    )

    gkf = GroupKFold(n_splits=n_splits)
    fold_results: List[Dict[str, Any]] = []
    best_f1_macro = -1.0
    best_fold = -1

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(x, y, groups)):
        eval_result, f1_macro = _train_single_fold(
            fold_idx=fold_idx,
            n_splits=n_splits,
            train_idx=train_idx,
            test_idx=test_idx,
            x=x,
            y=y,
            groups=groups,
            experiment_dir=experiment_dir,
            class_names=class_names,
            thresholds=thresholds,
            class_weight=class_weight,
            instrumentation_config=instrumentation_config,
            training_config=training_config,
            augmentation_config=augmentation_config,
            tracking_config=tracking_config,
            model_config=model_config,
        )
        fold_results.append(eval_result)

        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_fold = fold_idx

    return _write_summary(experiment_dir, fold_results, best_fold)
