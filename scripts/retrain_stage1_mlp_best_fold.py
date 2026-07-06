"""Retreina o melhor fold do MLP (fold 3) e salva modelo + scaler em models/."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_stage1_mlp import (
    build_mlp,
    compute_class_weights,
    train_fold,
    _resolve_output_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    npz = np.load("data/features/stage1_binary_features.npz")
    X, y, groups = npz["X"], npz["y"], npz["groups"]
    feature_names = json.loads(
        Path("data/features/stage1_binary_features.json").read_text(encoding="utf-8")
    )["feature_names"]

    best_fold = 3
    gkf = GroupKFold(n_splits=5)
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        if fold_idx != best_fold:
            continue
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        class_weight = compute_class_weights(y_train)
        output_dir = _resolve_output_dir("experiments/stage1_mlp_features_v2.1")
        result = train_fold(
            X_train, y_train, X_val, y_val, class_weight, fold_idx, output_dir,
            scaler=scaler,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Copia melhor modelo e scaler para models/
        models_dir = _resolve_output_dir("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            output_dir / f"fold_{fold_idx}" / "model.keras",
            models_dir / "stage1_mlp_features_v2.1.keras",
        )
        shutil.copy(
            output_dir / f"fold_{fold_idx}" / "input_scaler.pkl",
            models_dir / "input_scaler_stage1_mlp_features_v2.1.pkl",
        )
        (models_dir / "stage1_mlp_features_v2.1.config.json").write_text(
            json.dumps({"feature_names": feature_names}, indent=2, ensure_ascii=False)
        )
        print("Modelo e scaler salvos em models/")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
