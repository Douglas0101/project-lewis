"""Smoke test: treina fold 2 do Stage1 do zero (sem pre-treino).

Objetivo: testar se o pre-treino Chapman esta atrapalhando o Estagio 1.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.backbone_1d import build_backbone_1d
from src.models.evaluate import evaluate_fold

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("smoke_test_stage1_scratch")


def main():
    npz = np.load("data/features/stage1_binary.npz")
    X, y = npz["X"], npz["y"]
    meta = pd.read_parquet("data/features/stage1_binary.parquet")
    groups = meta["record_id"].values

    fold_idx = 2
    gkf = GroupKFold(n_splits=5)
    for i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        if i == fold_idx:
            break

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, 1))
    X_train_norm = scaler.transform(X_train.reshape(-1, 1)).reshape(-1, 500, 1)
    X_test_norm = scaler.transform(X_test.reshape(-1, 1)).reshape(-1, 500, 1)

    model = build_backbone_1d(
        input_len=500,
        num_classes=2,
        embedding_dim=64,
        dense_units=64,
        conv_filters=(16, 32, 64),
        conv_kernels=(7, 5, 3),
    )
    LOGGER.info("Modelo criado do zero (sem pre-treino)")

    class_weight = {
        0: 0.5678565541403311,
        1: 4.184242490165153,
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train_norm,
        y_train,
        validation_data=(X_test_norm, y_test),
        epochs=10,
        batch_size=128,
        class_weight=class_weight,
        verbose=2,
    )

    eval_result = evaluate_fold(model, X_test_norm, y_test, class_names=["N", "Anormal"])
    y_proba = model.predict(X_test_norm, batch_size=512, verbose=0)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, y_proba[:, 1])

    LOGGER.info(
        "Fold %d scratch | AUC=%.4f | F1_macro=%.4f | Acc=%.4f",
        fold_idx,
        auc,
        eval_result["global"]["F1_macro"],
        eval_result["global"]["Acc"],
    )

    out = Path("experiments/smoke_stage1_scratch")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
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
    model.save(str(out / "model.keras"), save_format="keras")


if __name__ == "__main__":
    main()
