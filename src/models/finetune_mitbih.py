"""Fine-tuning em MIT-BIH+ com transfer learning.

Congela camadas convolucionais, retreina classifier com class weights.
Loss: sparse_categorical_crossentropy | Activation: softmax
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .backbone_1d import freeze_conv_layers, save_model_config

LOGGER = logging.getLogger("lewis.camada04.finetune")


def _set_seeds(seed: int = 42) -> None:
    """Fixa seeds para reprodutibilidade."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """Calcula class weights balanceados.

    Parameters
    ----------
    y_train : np.ndarray
        Labels inteiros.

    Returns
    -------
    dict
        {class_index: weight}
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


def _make_callbacks(
    experiment_dir: Path,
    patience_es: int = 10,
    patience_lr: int = 5,
    monitor: str = "val_loss",
) -> list:
    """Cria callbacks para fine-tuning."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(experiment_dir / "finetuned_float32.keras"),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(experiment_dir / "training.log"),
            separator=",",
            append=False,
        ),
    ]


def finetune_mitbih(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    seed: int = 42,
    experiment_dir: Optional[Path] = None,
    monitor: str = "val_loss",
) -> Tuple[tf.keras.Model, dict]:
    """Fine-tuning com backbone congelado.

    Parameters
    ----------
    model : tf.keras.Model
        Backbone pré-treinado (camadas conv já congeladas).
    X_train : np.ndarray
        Dados de treino (shape: (n, 500, 1)).
    y_train : np.ndarray
        Labels inteiros (shape: (n,)).
    X_val : np.ndarray
        Dados de validação.
    y_val : np.ndarray
        Labels de validação.
    epochs : int
        Épocas máximas.
    batch_size : int
        Tamanho do batch.
    learning_rate : float
        Taxa de aprendizado (menor que pré-treino).
    seed : int
        Seed.
    experiment_dir : Path, optional
        Diretório do experimento.
    monitor : str
        Métrica para early stopping.

    Returns
    -------
    tuple
        (model, history_dict)
    """
    _set_seeds(seed)

    if experiment_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path("experiments") / f"exp_{ts}_finetune_mitbih"
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Fine-tuning MIT-BIH+ | experiment_dir=%s | epochs=%d | lr=%.1e | batch=%d",
        experiment_dir,
        epochs,
        learning_rate,
        batch_size,
    )

    # Garantir que camadas conv estão congeladas
    model = freeze_conv_layers(model)

    # Compilar
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", curve="ROC"),
        ],
    )

    # Class weights
    class_weight = _compute_class_weights(y_train)
    LOGGER.info("Class weights: %s", class_weight)

    # Callbacks
    callbacks = _make_callbacks(experiment_dir, monitor=monitor)

    # Treinar
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # Salvar config
    save_model_config(
        model,
        experiment_dir / "config.json",
        extra={
            "stage": "finetune_mitbih",
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "class_weight": class_weight,
            "monitor": monitor,
        },
    )

    # Salvar metrics
    metrics = {
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history.get("val_loss", [np.nan])[-1]),
        "final_acc": float(history.history.get("accuracy", [np.nan])[-1]),
        "final_val_acc": float(history.history.get("val_accuracy", [np.nan])[-1]),
        "final_val_auc": float(history.history.get("val_auc", [np.nan])[-1]),
        "stopped_epoch": len(history.history["loss"]),
    }
    with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    LOGGER.info(
        "Fine-tuning concluído | loss=%.4f | val_loss=%.4f | val_acc=%.4f",
        metrics["final_loss"],
        metrics["final_val_loss"],
        metrics["final_val_acc"],
    )
    return model, history.history
