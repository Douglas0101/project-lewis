"""Helpers compartilhados entre os smoke tests do Estágio 1."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.models.backbone_1d import build_backbone_1d
from src.models.evaluate import evaluate_fold

LOGGER = logging.getLogger("lewis.smoke_stage1")


def load_fold(fold_idx: int = 2, n_splits: int = 5):
    """Carrega o Stage1 e retorna índices do fold solicitado."""
    npz = np.load("data/features/stage1_binary.npz")
    X, y = npz["X"], npz["y"]
    meta = pd.read_parquet("data/features/stage1_binary.parquet")
    groups = meta["record_id"].values

    gkf = GroupKFold(n_splits=n_splits)
    for i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        if i == fold_idx:
            return X, y, train_idx, test_idx
    raise ValueError(f"fold_idx {fold_idx} fora do range {n_splits}")


def normalize_split(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aplica StandardScaler global em formato (n, 500, 1)."""
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, 1))
    X_train_norm = scaler.transform(X_train.reshape(-1, 1)).reshape(-1, 500, 1)
    X_test_norm = scaler.transform(X_test.reshape(-1, 1)).reshape(-1, 500, 1)
    return X_train_norm, X_test_norm


def build_base_model():
    """Constrói o backbone padrão usado nos smoke tests."""
    return build_backbone_1d(
        input_len=500,
        num_classes=2,
        embedding_dim=64,
        dense_units=64,
        conv_filters=(16, 32, 64),
        conv_kernels=(7, 5, 3),
    )


def run_smoke_test(
    name: str,
    out_dir: Path,
    model_setup: Callable[[tf.keras.Model], tf.keras.Model],
    learning_rate: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 128,
) -> dict:
    """Executa um smoke test do Estágio 1 e persiste métricas/modelo."""
    X, y, train_idx, test_idx = load_fold()

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train_norm, X_test_norm = normalize_split(X_train, X_test)

    model = build_base_model()
    model = model_setup(model)

    class_weight = {
        0: 0.5678565541403311,
        1: 4.184242490165153,
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train_norm,
        y_train,
        validation_data=(X_test_norm, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        verbose=2,
    )

    eval_result = evaluate_fold(model, X_test_norm, y_test, class_names=["N", "Anormal"])
    y_proba = model.predict(X_test_norm, batch_size=512, verbose=0)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, y_proba[:, 1])

    LOGGER.info(
        "%s | AUC=%.4f | F1_macro=%.4f | Acc=%.4f",
        name,
        auc,
        eval_result["global"]["F1_macro"],
        eval_result["global"]["Acc"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "auc": float(auc),
                "f1_macro": float(eval_result["global"]["F1_macro"]),
                "acc": float(eval_result["global"]["Acc"]),
                "per_class": eval_result["per_class"],
                "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    model.save(str(out_dir / "model.keras"), save_format="keras")

    return {
        "auc": float(auc),
        "f1_macro": float(eval_result["global"]["F1_macro"]),
        "acc": float(eval_result["global"]["Acc"]),
    }
