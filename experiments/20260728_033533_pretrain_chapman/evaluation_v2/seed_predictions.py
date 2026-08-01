"""Pré-semente de predições para o avaliador canônico (fallback sem provenance.json).

A run A0-histórica ``20260728_033533_pretrain_chapman`` não possui
``provenance.json``, portanto a regeneração automática de
``src.evaluation.canonical_evaluator`` falharia. Este script reconstrói o split
de validação determinístico (val_ratio=0.1, seed=42, batch=64, segment_len=500),
prediz com o checkpoint salvo e grava ``predictions/predictions.npz`` +
``predictions_meta.json`` para consumo do CLI canônico.

Uso (a partir da raiz do repo):
    uv run python experiments/20260728_033533_pretrain_chapman/evaluation_v2/seed_predictions.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = Path(__file__).resolve().parent
RUN_DIR = OUT_DIR.parent
SEED = 42


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    from src.models.keras_loader import load_keras_model
    from src.models.pretrain_chapman import build_datasets
    from src.models.pretrain_evaluation import predict_validation

    model_path = RUN_DIR / "backbone_pretrained.keras"
    sha = _sha256(model_path)

    _, val_ds, _, val_steps = build_datasets(
        val_ratio=0.1,
        batch_size=64,
        segment_len=500,
        seed=SEED,
        steps_per_epoch=1,
        validation_steps=None,  # estimativa automática de cardinalidade
    )
    model = load_keras_model(str(model_path), compile=False)
    y_true, y_prob = predict_validation(model, val_ds, val_steps)

    pred_dir = OUT_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pred_dir / "predictions.npz", y_true=y_true, y_score=y_prob)
    meta = {
        "sha256_model": sha,
        "seed": SEED,
        "validation_steps": int(val_steps),
        "n_samples": int(len(y_prob)),
        "source": "regenerated_fallback_no_provenance",
    }
    (pred_dir / "predictions_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(
        f"OK: n_samples={len(y_prob)} | val_steps={val_steps} | "
        f"y_true{y_true.shape} y_score{y_prob.shape} | sha256={sha}"
    )


if __name__ == "__main__":
    main()
