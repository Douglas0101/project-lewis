"""Retreina o melhor fold do MLP (fold 3) e salva modelo + scaler em models/."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_stage1_mlp import compute_class_weights, train_fold  # noqa: E402

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"


def _resolve_scaler_path(scaler_path: str) -> Path:
    """Resolve scaler path and ensure it stays inside features dir."""
    target = Path(scaler_path)
    if not target.is_absolute():
        target = FEATURES_DIR / target
    resolved = target.resolve()
    try:
        resolved.relative_to(FEATURES_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"Scaler path escapes features directory: {scaler_path}") from exc
    return resolved


def main() -> int:
    npz = np.load("data/features/stage1_binary_features.npz")
    X, y, groups = npz["X"], npz["y"], npz["groups"]
    metadata_path = Path("data/features/stage1_binary_features.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = metadata["feature_names"]
    scaler_path = _resolve_scaler_path(metadata["scaler_path"])

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler não encontrado em {scaler_path}. "
            "Execute scripts/prepare_stage1_features.py primeiro."
        )
    _ = joblib.load(scaler_path)
    LOGGER.info("Scaler loaded from %s", scaler_path.name)

    best_fold = 3
    gkf = GroupKFold(n_splits=5)
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        if fold_idx != best_fold:
            continue
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        class_weight = compute_class_weights(y_train)
        output_dir = Path("experiments/stage1_mlp_features_v2.1")
        result = train_fold(
            X_train, y_train, X_val, y_val, class_weight, fold_idx, output_dir
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Copia melhor modelo e scaler para models/
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            output_dir / f"fold_{fold_idx}" / "model.keras",
            models_dir / "stage1_mlp_features_v2.1.keras",
        )
        shutil.copy(
            scaler_path,
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
