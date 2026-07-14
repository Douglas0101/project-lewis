"""Treinamento de baseline minimal MLP sobre features enhanced (E06)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from src.models.split_protocol import _StratifiedGroupKFold as StratifiedGroupKFold

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("train_stage2_baseline_enhanced")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_minimal_mlp(input_dim: int, n_classes: int) -> keras.Model:
    """MLP minimal: uma camada oculta de 128 unidades."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _train_minimal_baseline(
    input_npz: Path,
    output_dir: Path,
    n_classes: int = 3,
    max_epochs: int = 30,
    batch_size: int = 256,
) -> dict:
    """Treina baseline MLP com StratifiedGroupKFold e retorna metricas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz = np.load(input_npz)
    X = np.asarray(npz["X"], dtype=np.float32)
    y = np.asarray(npz["y"], dtype=np.int64)
    groups = np.asarray(npz["groups"])

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups)):
        LOGGER.info("Fold %d", fold + 1)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        y_train = y[train_idx]
        y_val = y[val_idx]

        model = _build_minimal_mlp(X.shape[1], n_classes)
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=0
        )
        try:
            model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=max_epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0,
                class_weight={0: 1.0, 1: 1.0, 2: 8.0},
            )
        except Exception as exc:
            raise ValueError(f"Falha no treinamento do fold {fold}: {exc}") from exc

        y_pred = model.predict(X_val, verbose=0).argmax(axis=1)
        f1_macro = f1_score(
            y_val, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
        )  # type: ignore
        f1_per_class = np.asarray(
            f1_score(y_val, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
        )  # type: ignore
        metrics = {
            "fold": int(fold + 1),
            "f1_macro": float(f1_macro),
            "f1_S": float(f1_per_class[0]),
            "f1_V": float(f1_per_class[1]),
            "f1_F": float(f1_per_class[2]),
        }
        fold_metrics.append(metrics)
        LOGGER.info("Fold %d metrics: %s", fold + 1, metrics)

    try:
        df = pd.DataFrame(fold_metrics)
    except Exception as exc:
        raise ValueError(f"Falha ao construir DataFrame de metricas: {exc}") from exc

    try:
        summary = {
            "fold_metrics": fold_metrics,
            "mean": {
                "f1_macro": float(df["f1_macro"].mean()),
                "f1_S": float(df["f1_S"].mean()),
                "f1_V": float(df["f1_V"].mean()),
                "f1_F": float(df["f1_F"].mean()),
            },
            "std": {
                "f1_macro": float(df["f1_macro"].std()),
                "f1_S": float(df["f1_S"].std()),
                "f1_V": float(df["f1_V"].std()),
                "f1_F": float(df["f1_F"].std()),
            },
        }

        with open(output_dir / "baseline_enhanced_metrics.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise ValueError(f"Falha ao salvar metricas: {exc}") from exc

    LOGGER.info("Baseline enhanced summary: %s", summary["mean"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Baseline minimal MLP sobre features enhanced para classe F."
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "features"
        / "stage2_multiclass_features_enhanced_e06_v1.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "stage2_v2.4_research" / "E06_feature_engineering",
    )
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    try:
        _train_minimal_baseline(
            args.input_npz,
            args.output_dir,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        LOGGER.error("Falha no treinamento baseline enhanced: %s", exc)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
