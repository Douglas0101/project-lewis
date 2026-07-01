"""Treina MLP leve sobre features morfológicas/time-domain para Estágio 1.

Este é o fallback previsto no UNIFIED_DOCUMENT (Decisão 7) quando CNN pura
sobre sinal raw falha em separar N vs Anormal.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.evaluate import evaluate_fold

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("train_stage1_mlp")


def build_mlp(input_dim: int, num_classes: int = 2) -> tf.keras.Model:
    """MLP leve para classificação binária com features."""
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = tf.keras.layers.Dense(32, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="output"
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="stage1_mlp")
    return model


def compute_class_weights(y: np.ndarray) -> dict:
    """Pesos balanceados limitados a max_weight=20."""
    classes = np.unique(y)
    counts = np.array([np.sum(y == c) for c in classes])
    weights = 1.0 / counts
    weights = weights / weights.min()
    weights = np.minimum(weights, 20.0)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weight: dict,
    fold_idx: int,
    output_dir: Path,
    scaler=None,
) -> dict:
    """Treina um único fold."""
    model = build_mlp(input_dim=X_train.shape[1], num_classes=2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    eval_result = evaluate_fold(model, X_val, y_val, class_names=["N", "Anormal"])
    y_proba = model.predict(X_val, batch_size=1024, verbose=0)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_val, y_proba[:, 1])

    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(fold_dir / "model.keras"), save_format="keras")
    if scaler is not None:
        import joblib
        joblib.dump(scaler, fold_dir / "input_scaler.pkl")

    LOGGER.info(
        "Fold %d | AUC=%.4f | F1_macro=%.4f | Acc=%.4f | epochs=%d",
        fold_idx,
        auc,
        eval_result["global"]["F1_macro"],
        eval_result["global"]["Acc"],
        len(history.history["loss"]),
    )

    return {
        "fold": fold_idx,
        "auc": float(auc),
        "eval_result": eval_result,
        "epochs_trained": len(history.history["loss"]),
    }


def main() -> int:
    npz = np.load("data/features/stage1_binary_features.npz")
    X, y, groups = npz["X"], npz["y"], npz["groups"]
    feature_names = json.loads(
        Path("data/features/stage1_binary_features.json").read_text(encoding="utf-8")
    )["feature_names"]

    LOGGER.info("Dataset: X=%s, y=%s", X.shape, y.shape)
    LOGGER.info("Features: %s", feature_names)

    output_dir = Path("experiments/stage1_mlp_features_v2.1")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        LOGGER.info("=== Fold %d/%d ===", fold_idx + 1, n_splits)
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        class_weight = compute_class_weights(y_train)
        result = train_fold(
            X_train, y_train, X_val, y_val, class_weight, fold_idx, output_dir,
            scaler=scaler,
        )
        fold_results.append(result)

    # Agregação
    f1_macros = [r["eval_result"]["global"]["F1_macro"] for r in fold_results]
    accs = [r["eval_result"]["global"]["Acc"] for r in fold_results]
    aucs = [r["auc"] for r in fold_results]

    summary = {
        "experiment": "stage1_mlp_features_v2.1",
        "feature_names": feature_names,
        "folds": fold_results,
        "mean": {
            "Acc": float(np.mean(accs)),
            "F1_macro": float(np.mean(f1_macros)),
            "AUC": float(np.mean(aucs)),
        },
        "std": {
            "Acc": float(np.std(accs)),
            "F1_macro": float(np.std(f1_macros)),
            "AUC": float(np.std(aucs)),
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    LOGGER.info(
        "=== Resultado agregado === Acc=%.4f±%.4f | F1_macro=%.4f±%.4f | AUC=%.4f±%.4f",
        summary["mean"]["Acc"],
        summary["std"]["Acc"],
        summary["mean"]["F1_macro"],
        summary["std"]["F1_macro"],
        summary["mean"]["AUC"],
        summary["std"]["AUC"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
