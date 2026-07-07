"""Otimiza threshold do Estágio 1 para F1-macro com restrições QG5' v2.2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    npz_path = Path("data/features/stage1_binary_features.npz")
    model_path = Path("models/stage1_mlp_features_v2.1.keras")
    scaler_path = Path("models/input_scaler_stage1_mlp_features_v2.1.pkl")
    out_path = Path("models/stage1_threshold_v2.1.json")

    npz = np.load(npz_path)
    X, y = npz["X"].astype(np.float32), npz["y"].astype(np.int64)

    scaler = joblib.load(scaler_path)
    model = tf.keras.models.load_model(str(model_path), compile=False)

    X_scaled = scaler.transform(X).astype(np.float32)
    proba = model.predict(X_scaled, batch_size=1024, verbose=0)[:, 1]

    thresholds = np.arange(0.05, 0.95, 0.01)
    best = {"threshold": 0.5, "f1_macro": 0.0}

    for thr in thresholds:
        y_pred = (proba >= thr).astype(np.int64)
        f1 = f1_score(y, y_pred, average="macro", zero_division=0.0)
        rec = recall_score(y, y_pred, pos_label=1, zero_division=0.0)
        prec = precision_score(y, y_pred, pos_label=1, zero_division=0.0)
        f1_n = f1_score(y, y_pred, labels=[0, 1], average=None, zero_division=0.0)[0]

        # Restrições QG5' v2.2: recall/precision Anormal mínimos E F1(N) alto.
        if rec >= 0.30 and prec >= 0.25 and f1_n >= 0.90 and f1 > best["f1_macro"]:
            best = {
                "threshold": float(thr),
                "f1_macro": float(f1),
                "recall_anormal": float(rec),
                "precision_anormal": float(prec),
                "f1_n": float(f1_n),
            }

    out_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
