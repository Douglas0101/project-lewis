"""Treina MLP leve sobre features morfológicas/time-domain para Estágio 2.

Classificador S vs V vs F usando as mesmas 13 features do Estágio 1.
Pesos de classe são balanceados com teto configurável para evitar viés excessivo.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.evaluate import evaluate_fold

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("train_stage2_mlp")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_output_dir(output_dir: str) -> Path:
    """Resolve output directory and ensure it stays inside PROJECT_ROOT."""
    target = Path(output_dir)
    if not target.is_absolute():
        target = PROJECT_ROOT / output_dir
    resolved = target.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Output directory escapes project root: {output_dir!r}") from exc
    return resolved


def build_mlp(input_dim: int, num_classes: int = 3, hidden_units: int = 32) -> tf.keras.Model:
    """MLP leve para classificação multiclasse com features."""
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = tf.keras.layers.Dense(hidden_units, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="output"
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="stage2_mlp")
    return model


def compute_class_weights(
    y: np.ndarray,
    max_weight: float = 10.0,
    f_weight_override: float | None = None,
    s_weight_override: float | None = None,
    v_weight_override: float | None = None,
) -> dict:
    """Pesos balanceados com teto para evitar domínio da classe minoritária."""
    classes = np.unique(y)
    counts = np.array([np.sum(y == c) for c in classes])
    weights = 1.0 / counts
    weights = weights / weights.min()
    weights = np.minimum(weights, max_weight)
    overrides = {
        0: s_weight_override,  # S
        1: v_weight_override,  # V
        2: f_weight_override,  # F
    }
    for cls, override in overrides.items():
        if override is not None:
            idx = np.where(classes == cls)[0]
            if len(idx) > 0:
                weights[idx[0]] = override
    return {int(c): float(w) for c, w in zip(classes, weights)}


def oversample_class(
    X: np.ndarray,
    y: np.ndarray,
    target_class: int,
    target_ratio: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Real oversampling: duplicate minority class samples up to target ratio.

    target_ratio is relative to the largest class count.
    """
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(counts.max())
    target_count = min(int(max_count * target_ratio), max_count)

    idx = np.where(y == target_class)[0]
    current_count = len(idx)
    if current_count == 0 or target_count <= current_count:
        return X, y

    rng = np.random.default_rng(seed)
    n_needed = target_count - current_count
    extra_idx = rng.choice(idx, size=n_needed, replace=True)

    X_aug = np.concatenate([X, X[extra_idx]], axis=0)
    y_aug = np.concatenate([y, y[extra_idx]], axis=0)

    # Shuffle
    perm = rng.permutation(len(y_aug))
    return X_aug[perm], y_aug[perm]


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
    model = build_mlp(input_dim=X_train.shape[1], num_classes=3)
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

    eval_result = evaluate_fold(
        model, X_val, y_val, class_names=["S", "V", "F"], optimize_thresholds=False
    )
    y_proba = model.predict(X_val, batch_size=1024, verbose=0)

    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(fold_dir / "model.keras"), save_format="keras")
    if scaler is not None:
        import joblib
        joblib.dump(scaler, fold_dir / "input_scaler.pkl")

    LOGGER.info(
        "Fold %d | F1_macro=%.4f | Acc=%.4f | per_class F1=%s | epochs=%d",
        fold_idx,
        eval_result["global"]["F1_macro"],
        eval_result["global"]["Acc"],
        {k: v["F1"] for k, v in eval_result["per_class"].items()},
        len(history.history["loss"]),
    )

    return {
        "fold": fold_idx,
        "eval_result": eval_result,
        "epochs_trained": len(history.history["loss"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Treina Estágio 2 MLP sobre features.")
    parser.add_argument(
        "--f-oversample-ratio",
        type=float,
        default=0.5,
        help="Target ratio of F samples relative to the largest class after real oversampling.",
    )
    parser.add_argument(
        "--s-weight",
        type=float,
        default=None,
        help="Override class weight for class S (encoded as 0).",
    )
    parser.add_argument(
        "--v-weight",
        type=float,
        default=None,
        help="Override class weight for class V (encoded as 1).",
    )
    parser.add_argument(
        "--f-weight",
        type=float,
        default=None,
        help="Override class weight for class F (encoded as 2).",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=10.0,
        help="Max class weight cap for balanced weighting.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/stage2_mlp_features_v2.1",
        help="Directory to save fold models and summary.",
    )
    args = parser.parse_args()

    npz = np.load("data/features/stage2_multiclass_features.npz")
    X, y, groups = npz["X"], npz["y"], npz["groups"]
    feature_names = json.loads(
        Path("data/features/stage2_multiclass_features.json").read_text(encoding="utf-8")
    )["feature_names"]

    LOGGER.info("Dataset: X=%s, y=%s", X.shape, y.shape)
    LOGGER.info("Features: %s", feature_names)
    LOGGER.info(
        "Mitigation config: f_oversample_ratio=%s, s_weight=%s, v_weight=%s, f_weight=%s, max_weight=%s",
        args.f_oversample_ratio,
        args.s_weight,
        args.v_weight,
        args.f_weight,
        args.max_weight,
    )

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        LOGGER.info("=== Fold %d/%d ===", fold_idx + 1, n_splits)
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Real oversampling of class F (no synthetic SMOTE) before scaling.
        X_train, y_train = oversample_class(
            X_train,
            y_train,
            target_class=2,
            target_ratio=args.f_oversample_ratio,
            seed=42 + fold_idx,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        class_weight = compute_class_weights(
            y_train,
            max_weight=args.max_weight,
            s_weight_override=args.s_weight,
            v_weight_override=args.v_weight,
            f_weight_override=args.f_weight,
        )
        result = train_fold(
            X_train, y_train, X_val, y_val, class_weight, fold_idx, output_dir,
            scaler=scaler,
        )
        fold_results.append(result)

    # Agregação
    f1_macros = [r["eval_result"]["global"]["F1_macro"] for r in fold_results]
    accs = [r["eval_result"]["global"]["Acc"] for r in fold_results]
    per_class_f1 = {
        cls: [r["eval_result"]["per_class"][cls]["F1"] for r in fold_results]
        for cls in ("S", "V", "F")
    }

    summary = {
        "experiment": "stage2_mlp_features_v2.1",
        "feature_names": feature_names,
        "folds": fold_results,
        "mean": {
            "Acc": float(np.mean(accs)),
            "F1_macro": float(np.mean(f1_macros)),
            "F1_S": float(np.mean(per_class_f1["S"])),
            "F1_V": float(np.mean(per_class_f1["V"])),
            "F1_F": float(np.mean(per_class_f1["F"])),
        },
        "std": {
            "Acc": float(np.std(accs)),
            "F1_macro": float(np.std(f1_macros)),
            "F1_S": float(np.std(per_class_f1["S"])),
            "F1_V": float(np.std(per_class_f1["V"])),
            "F1_F": float(np.std(per_class_f1["F"])),
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    LOGGER.info(
        "=== Resultado agregado === Acc=%.4f±%.4f | F1_macro=%.4f±%.4f | "
        "F1(S)=%.4f±%.4f | F1(V)=%.4f±%.4f | F1(F)=%.4f±%.4f",
        summary["mean"]["Acc"],
        summary["std"]["Acc"],
        summary["mean"]["F1_macro"],
        summary["std"]["F1_macro"],
        summary["mean"]["F1_S"],
        summary["std"]["F1_S"],
        summary["mean"]["F1_V"],
        summary["std"]["F1_V"],
        summary["mean"]["F1_F"],
        summary["std"]["F1_F"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
