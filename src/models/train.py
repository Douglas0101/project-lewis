"""Treinamento com GroupKFold por paciente (inter-patient).

NUNCA misturar batimentos do mesmo paciente entre treino e teste.
Fit scaler no treino apenas; carregar backbone pré-treinado; congelar convs.
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .backbone_1d import build_backbone_1d, freeze_conv_layers
from .evaluate import evaluate_fold
from .finetune_mitbih import finetune_mitbih

LOGGER = logging.getLogger("lewis.camada04.train")


def _set_seeds(seed: int = 42) -> None:
    """Fixa seeds para reprodutibilidade."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _normalize_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Normalização z-score global: fit no treino, transform em teste.

    Parameters
    ----------
    X_train : np.ndarray
        Shape (n_train, 500, 1).
    X_test : np.ndarray
        Shape (n_test, 500, 1).

    Returns
    -------
    tuple
        (X_train_norm, X_test_norm, scaler)
    """
    scaler = StandardScaler()
    n_train, seq_len, channels = X_train.shape
    n_test = X_test.shape[0]

    # Fit no treino (reshape para 2D)
    X_train_2d = X_train.reshape(-1, channels)
    scaler.fit(X_train_2d)

    # Transform treino e teste
    X_train_norm = scaler.transform(X_train_2d).reshape(n_train, seq_len, channels)
    X_test_norm = scaler.transform(X_test.reshape(-1, channels)).reshape(n_test, seq_len, channels)

    return X_train_norm, X_test_norm, scaler


def train_group_kfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    backbone_weights: Path,
    n_splits: int = 5,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    seed: int = 42,
    experiment_dir: Optional[Path] = None,
    monitor: str = "val_loss",
) -> Dict[str, Any]:
    """Treinamento GroupKFold por paciente.

    Parameters
    ----------
    X : np.ndarray
        Dados (shape: (n, 500, 1)).
    y : np.ndarray
        Labels inteiros (shape: (n,)).
    groups : np.ndarray
        IDs de paciente (shape: (n,)).
    backbone_weights : Path
        Caminho para pesos do backbone pré-treinado.
    n_splits : int
        Número de folds.
    epochs : int
        Épocas por fold.
    batch_size : int
        Batch size.
    learning_rate : float
        LR para fine-tuning.
    seed : int
        Seed.
    experiment_dir : Path, optional
        Diretório raiz dos experimentos.
    monitor : str
        Métrica para early stopping.

    Returns
    -------
    dict
        {
            "folds": [resultados por fold],
            "best_fold": índice do melhor fold,
            "mean_metrics": médias,
            "std_metrics": desvios,
        }
    """
    _set_seeds(seed)

    if experiment_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path("experiments") / f"exp_{ts}_groupkfold"
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "GroupKFold | n_splits=%d | n_samples=%d | n_patients=%d",
        n_splits,
        len(X),
        len(np.unique(groups)),
    )

    gkf = GroupKFold(n_splits=n_splits)
    fold_results: List[dict] = []
    best_f1_macro = -1.0
    best_fold = -1

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        LOGGER.info("=== Fold %d/%d ===", fold_idx + 1, n_splits)
        fold_dir = experiment_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        LOGGER.info(
            "  Train: n=%d | Test: n=%d | Patients train=%d | Patients test=%d",
            len(X_train),
            len(X_test),
            len(np.unique(groups[train_idx])),
            len(np.unique(groups[test_idx])),
        )

        # 1. Normalização global (fit no treino)
        X_train_norm, X_test_norm, scaler = _normalize_fold(X_train, X_test)

        # Salvar scaler
        import joblib

        joblib.dump(scaler, fold_dir / "input_scaler.pkl")

        # 2. Carregar backbone pré-treinado
        model = build_backbone_1d(input_len=X.shape[1], num_classes=len(np.unique(y)))
        model.load_weights(str(backbone_weights))
        model = freeze_conv_layers(model)

        # 3. Fine-tuning
        model, history = finetune_mitbih(
            model=model,
            X_train=X_train_norm,
            y_train=y_train,
            X_val=X_test_norm,
            y_val=y_test,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            experiment_dir=fold_dir,
            monitor=monitor,
        )

        # 4. Avaliação
        eval_result = evaluate_fold(model, X_test_norm, y_test)
        eval_result["fold"] = fold_idx
        eval_result["history"] = history
        fold_results.append(eval_result)

        f1_macro = eval_result["global"]["F1_macro"]
        LOGGER.info(
            "  Fold %d | F1-macro=%.4f | Acc=%.4f | MCC=%.4f",
            fold_idx,
            f1_macro,
            eval_result["global"]["Acc"],
            eval_result["global"]["MCC"],
        )

        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_fold = fold_idx

    # Resumo
    f1_macros = [r["global"]["F1_macro"] for r in fold_results]
    accs = [r["global"]["Acc"] for r in fold_results]
    mccs = [r["global"]["MCC"] for r in fold_results]

    mean_metrics: Dict[str, float] = {
        "F1_macro": round(float(np.mean(f1_macros)), 4),
        "Acc": round(float(np.mean(accs)), 4),
        "MCC": round(float(np.mean(mccs)), 4),
    }
    std_metrics: Dict[str, float] = {
        "F1_macro": round(float(np.std(f1_macros)), 4),
        "Acc": round(float(np.std(accs)), 4),
        "MCC": round(float(np.std(mccs)), 4),
    }
    summary: Dict[str, Any] = {
        "folds": fold_results,
        "best_fold": best_fold,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "passes_qg5": all(r["passes_qg5"] for r in fold_results),
    }

    with (experiment_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    LOGGER.info(
        "GroupKFold completo | Best fold=%d | Mean F1-macro=%.4f ± %.4f | Mean Acc=%.4f ± %.4f",
        best_fold,
        summary["mean_metrics"]["F1_macro"],
        summary["std_metrics"]["F1_macro"],
        summary["mean_metrics"]["Acc"],
        summary["std_metrics"]["Acc"],
    )
    return summary
