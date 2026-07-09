"""Teste de Quality Gate 5' (config v2.3) para os modelos MLP sobre features.

Avalia os modelos publicados em ``models/`` versão v2.3 sobre as features
pré-preparadas em ``data/features/*_features.npz``.  O pipeline de extração
de features é coberto por QG3; aqui verificamos que os classificadores MLP
cumprem os thresholds do QG5' v2.3.

Thresholds QG5' v2.3:
    - Estágio 1: Recall(Anormal) >= 0.30, Precision(Anormal) >= 0.25,
                 F1-macro >= 0.55
    - Estágio 2: F1(S) >= 0.55, F1(V) >= 0.70, F1(F) >= 0.15,
                 F1-macro >= 0.45
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.inference.two_stage_mlp_pipeline import TwoStageMLPPipeline

# Thresholds QG5' v2.3
STAGE1_MIN_RECALL_ANORMAL = 0.30
STAGE1_MIN_PRECISION_ANORMAL = 0.25
STAGE1_MIN_F1_MACRO = 0.55
STAGE2_MIN_F1_S = 0.55
STAGE2_MIN_F1_V = 0.70
STAGE2_MIN_F1_F = 0.15
STAGE2_MIN_F1_MACRO = 0.45

STAGE2_MAX_SAMPLES_PER_CLASS = 683


@dataclass(frozen=True)
class StageMetrics:
    recall: float
    precision: float
    f1_macro: float
    accuracy: float
    per_class_f1: Dict[str, float]


def _stage1_metrics(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> StageMetrics:
    """Avalia Estágio 1 (N vs Anormal)."""
    proba = model.predict(X, batch_size=4096, verbose=0)[:, 1]
    y_pred = (proba >= threshold).astype(np.int64)
    y_true_bin = (y != 0).astype(np.int64)

    return StageMetrics(
        recall=float(
            recall_score(y_true_bin, y_pred, pos_label=1, zero_division=0.0)
        ),
        precision=float(
            precision_score(y_true_bin, y_pred, pos_label=1, zero_division=0.0)
        ),
        f1_macro=float(
            f1_score(y_true_bin, y_pred, labels=[0, 1], average="macro", zero_division=0.0)
        ),
        accuracy=float(accuracy_score(y_true_bin, y_pred)),
        per_class_f1={},
    )


def _stage2_metrics(
    model: tf.keras.Model,
    scaler: Any,
    X: np.ndarray,
    y: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> StageMetrics:
    """Avalia Estágio 2 (S vs V vs F).

    Se ``thresholds`` for fornecido, aplica decisão one-vs-rest com fallback para
    a classe majoritária V e desempate pela maior probabilidade, espelhando o
    comportamento do pipeline de inferência.
    """
    X_scaled = scaler.transform(X)
    proba = model.predict(X_scaled, batch_size=4096, verbose=0)

    if thresholds is None:
        y_pred = np.argmax(proba, axis=1).astype(np.int64)
    else:
        class_names = ["S", "V", "F"]
        thresh_array = np.array(
            [thresholds[name] for name in class_names], dtype=proba.dtype
        )
        above = proba >= thresh_array
        n_above = above.sum(axis=1)
        n_samples = proba.shape[0]

        y_pred = np.full(n_samples, 1, dtype=np.int64)

        single_mask = n_above == 1
        y_pred[single_mask] = np.argmax(above[single_mask], axis=1)

        multi_mask = n_above > 1
        if np.any(multi_mask):
            masked_scores = np.where(above[multi_mask], proba[multi_mask], -np.inf)
            y_pred[multi_mask] = np.argmax(masked_scores, axis=1)

        none_mask = n_above == 0
        y_pred[none_mask] = np.argmax(proba[none_mask], axis=1)

    return StageMetrics(
        recall=float(
            recall_score(y, y_pred, labels=[0, 1, 2], average="macro", zero_division=0.0)
        ),
        precision=float(
            precision_score(y, y_pred, labels=[0, 1, 2], average="macro", zero_division=0.0)
        ),
        f1_macro=float(
            f1_score(y, y_pred, labels=[0, 1, 2], average="macro", zero_division=0.0)
        ),
        accuracy=float(accuracy_score(y, y_pred)),
        per_class_f1={
            cls: float(score)
            for cls, score in zip(
                ["S", "V", "F"],
                f1_score(y, y_pred, labels=[0, 1, 2], average=None, zero_division=0.0),
            )
        },
    )


@pytest.fixture(scope="module")
def published_artifacts() -> Dict[str, Path]:
    """Verifica e retorna os artefatos publicados v2.3."""
    project_root = Path(__file__).resolve().parents[1]
    artifacts = {
        "stage1_model": project_root / "models" / "stage1_float32_v2.3.keras",
        "stage1_scaler": project_root / "models" / "input_scaler_stage1_v2.3.pkl",
        "stage2_model": project_root / "models" / "stage2_float32_v2.3.keras",
        "stage2_scaler": project_root / "models" / "input_scaler_stage2_v2.3.pkl",
        "stage1_threshold": project_root / "models" / "stage1_threshold_v2.3.json",
        "stage2_threshold": project_root / "models" / "stage2_threshold_v2.3.json",
        "stage1_features": project_root / "data" / "features" / "stage1_binary_features.npz",
        "stage2_features": project_root / "data" / "features" / "stage2_multiclass_features.npz",
    }
    missing = [p for p in artifacts.values() if not p.exists()]
    if missing:
        pytest.skip(f"Artefatos v2.3 não disponíveis: {missing}")
    return artifacts


@pytest.mark.qg5
@pytest.mark.mlp
@pytest.mark.slow
def test_two_stage_mlp_qg5_stage1(published_artifacts: Dict[str, Path]) -> None:
    """QG5' Estágio 1: recall/precision/F1-macro sobre features preparadas."""
    model = tf.keras.models.load_model(
        str(published_artifacts["stage1_model"]), compile=False
    )
    data = np.load(published_artifacts["stage1_features"])
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)

    threshold = json.loads(
        published_artifacts["stage1_threshold"].read_text(encoding="utf-8")
    )["threshold"]

    metrics = _stage1_metrics(model, X, y, threshold=threshold)

    print("\n[QG5' v2.3] Estágio 1 MLP")
    print(
        f"  Recall(Anormal)={metrics.recall:.4f} "
        f"Precision(Anormal)={metrics.precision:.4f} "
        f"F1-macro={metrics.f1_macro:.4f} Acc={metrics.accuracy:.4f}"
    )

    tol = 1e-6
    assert metrics.recall + tol >= STAGE1_MIN_RECALL_ANORMAL, (
        f"Recall(Anormal)={metrics.recall:.4f} abaixo de {STAGE1_MIN_RECALL_ANORMAL}"
    )
    assert metrics.precision + tol >= STAGE1_MIN_PRECISION_ANORMAL, (
        f"Precision(Anormal)={metrics.precision:.4f} abaixo de {STAGE1_MIN_PRECISION_ANORMAL}"
    )
    assert metrics.f1_macro + tol >= STAGE1_MIN_F1_MACRO, (
        f"F1-macro={metrics.f1_macro:.4f} abaixo de {STAGE1_MIN_F1_MACRO}"
    )


@pytest.mark.qg5
@pytest.mark.mlp
@pytest.mark.slow
def test_two_stage_mlp_qg5_stage2(published_artifacts: Dict[str, Path]) -> None:
    """QG5' Estágio 2: F1 por classe sobre subset estratificado balanceado."""
    model = tf.keras.models.load_model(
        str(published_artifacts["stage2_model"]), compile=False
    )
    scaler = joblib.load(published_artifacts["stage2_scaler"])
    data = np.load(published_artifacts["stage2_features"])
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)

    rng = np.random.default_rng(42)
    selected: List[int] = []
    for cls in range(3):
        idx = np.where(y == cls)[0]
        n = min(len(idx), STAGE2_MAX_SAMPLES_PER_CLASS)
        selected.extend(rng.choice(idx, size=n, replace=False).tolist())
    selected = np.array(selected)
    rng.shuffle(selected)

    stage2_thresholds: dict[str, float] | None = None
    if published_artifacts["stage2_threshold"].exists():
        stage2_thresholds = json.loads(
            published_artifacts["stage2_threshold"].read_text(encoding="utf-8")
        ).get("thresholds")

    metrics = _stage2_metrics(
        model, scaler, X[selected], y[selected], thresholds=stage2_thresholds
    )

    print("\n[QG5' v2.3] Estágio 2 MLP")
    print(
        f"  F1-macro={metrics.f1_macro:.4f} "
        f"F1(S)={metrics.per_class_f1['S']:.4f} "
        f"F1(V)={metrics.per_class_f1['V']:.4f} "
        f"F1(F)={metrics.per_class_f1['F']:.4f}"
    )

    tol = 1e-6
    assert metrics.f1_macro + tol >= STAGE2_MIN_F1_MACRO, (
        f"F1-macro={metrics.f1_macro:.4f} abaixo de {STAGE2_MIN_F1_MACRO}"
    )
    assert metrics.per_class_f1["S"] + tol >= STAGE2_MIN_F1_S, (
        f"F1(S)={metrics.per_class_f1['S']:.4f} abaixo de {STAGE2_MIN_F1_S}"
    )
    assert metrics.per_class_f1["V"] + tol >= STAGE2_MIN_F1_V, (
        f"F1(V)={metrics.per_class_f1['V']:.4f} abaixo de {STAGE2_MIN_F1_V}"
    )
    assert metrics.per_class_f1["F"] + tol >= STAGE2_MIN_F1_F, (
        f"F1(F)={metrics.per_class_f1['F']:.4f} abaixo de {STAGE2_MIN_F1_F}"
    )


@pytest.mark.qg5
@pytest.mark.mlp
@pytest.mark.slow
def test_two_stage_mlp_pipeline_sanity(published_artifacts: Dict[str, Path]) -> None:
    """Sanity check: o pipeline publicado carrega e executa sem erros."""
    project_root = Path(__file__).resolve().parents[1]
    pipeline = TwoStageMLPPipeline.from_directory(project_root / "models", use_quantized=False)
    pipeline.load()

    # Dummy 500-sample segment (cannot be all zeros due to morphological extraction)
    t = np.linspace(0, 1, 500, dtype=np.float32)
    X = np.sin(2 * np.pi * 5 * t).reshape(1, 500)
    temporal = [{
        "rr_prev": 800.0,
        "rr_next": 800.0,
        "rr_ratio": 1.0,
        "rr_local_mean": 800.0,
        "rr_local_std": 20.0,
        "rmssd": 30.0,
        "heart_rate": 75.0,
    }]
    result = pipeline.predict(X, temporal_features=temporal)
    assert len(result["class"]) == 1
    assert result["class"][0] in {"N", "S", "V", "F"}
