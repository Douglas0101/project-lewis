"""F1MacroCheckpoint — Callback Keras para salvar/restaurar melhores pesos.

Seleciona o melhor modelo segundo métrica AAMI configurável (F1_macro, Se_Anormal,
F1_V etc.), salva os pesos e, quando aplicável, exporta o threshold de decisão.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

LOGGER = logging.getLogger("lewis.callbacks.f1_macro_checkpoint")


class F1MacroCheckpoint(tf.keras.callbacks.Callback):
    """Salva o melhor modelo segundo métrica AAMI na validação.

    Substitui o ModelCheckpoint baseado em ``val_loss``, que é distorcido pelos
    class weights. Monitora diretamente a métrica alvo do QG5.

    Parameters
    ----------
    X_val : np.ndarray
        Dados de validação.
    y_val : np.ndarray
        Labels de validação (inteiros).
    filepath : Path or str
        Caminho para salvar os melhores pesos.
    class_names : list[str], optional
        Nomes das classes para avaliação AAMI.
    thresholds : dict, optional
        Thresholds configuráveis para ``evaluate_aami``.
    metric : str
        Métrica de seleção (ex.: "F1_macro", "Se_Anormal", "F1_V").
    patience : int
        Épocas de paciência para early stopping.
    optimize_thresholds : bool
        Se True, realiza threshold tuning one-vs-rest para multiclasse.
    """

    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        filepath: Path | str,
        class_names: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, Any]] = None,
        metric: str = "F1_macro",
        patience: int = 10,
        optimize_thresholds: bool = False,
    ):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.filepath = Path(filepath)
        self.class_names = class_names
        self.thresholds = thresholds
        self.metric = metric
        self.patience = patience
        self.optimize_thresholds = optimize_thresholds
        self.best_score = -1.0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_threshold: Optional[float] = None
        self.best_thresholds: Optional[Dict[str, float]] = None

    def _extract_score(self, result: Dict[str, Any]) -> float:
        """Extrai a métrica de seleção do resultado AAMI."""
        if self.metric == "F1_macro":
            return float(result["global"]["F1_macro"])
        if self.metric.startswith("Se_") or self.metric.startswith("F1_"):
            metric_name, cls = self.metric.split("_", 1)
            return float(result["per_class"][cls][metric_name])
        raise ValueError(f"Unsupported selection metric: {self.metric}")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Avalia validação e salva pesos se a métrica melhorou."""
        from src.models.evaluate import (
            evaluate_aami,
            find_best_threshold,
            find_best_thresholds_multiclass,
        )

        y_proba = self.model.predict(self.X_val, verbose=0)

        if self.class_names is not None and len(self.class_names) == 2:
            result = find_best_threshold(
                self.y_val,
                y_proba[:, 1],
                class_names=self.class_names,
                thresholds=self.thresholds,
                target_class_idx=1,
            )
            threshold = result["threshold"]
            thresholds_dict = None
        elif self.optimize_thresholds and self.class_names is not None:
            result = find_best_thresholds_multiclass(
                self.y_val,
                y_proba,
                class_names=self.class_names,
                thresholds_cfg=self.thresholds,
            )
            threshold = None
            thresholds_dict = result.get("thresholds")
        else:
            y_pred = np.argmax(y_proba, axis=1)
            result = evaluate_aami(
                self.y_val,
                y_pred,
                class_names=self.class_names,
                thresholds=self.thresholds,
            )
            threshold = None
            thresholds_dict = None

        score = self._extract_score(result)
        logs = logs or {}
        logs[f"val_{self.metric}"] = score
        threshold_str = (
            f"{threshold:.2f}"
            if threshold is not None
            else (str(thresholds_dict) if thresholds_dict is not None else "argmax")
        )
        LOGGER.info(
            "Epoch %d | val_%s=%.4f | QG=%s | threshold=%s",
            epoch + 1,
            self.metric,
            score,
            result["passes_qg5"],
            threshold_str,
        )

        if score > self.best_score:
            self.best_score = score
            self.wait = 0
            self.model.save_weights(str(self.filepath))
            if threshold is not None:
                self.best_threshold = threshold
                threshold_path = self.filepath.with_suffix(".threshold.json")
                with threshold_path.open("w", encoding="utf-8") as fh:
                    json.dump({"threshold": float(threshold)}, fh, indent=2)
            elif thresholds_dict is not None:
                self.best_thresholds = thresholds_dict
                threshold_path = self.filepath.with_suffix(".threshold.json")
                with threshold_path.open("w", encoding="utf-8") as fh:
                    json.dump({"thresholds": thresholds_dict}, fh, indent=2)
            LOGGER.info("%s improved -> saved weights to %s", self.metric, self.filepath)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                LOGGER.info(
                    "Early stop at epoch %d (best %s=%.4f)",
                    epoch + 1,
                    self.metric,
                    self.best_score,
                )

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        """Restaura os melhores pesos salvos, se existirem."""
        if self.filepath.exists():
            LOGGER.info("Restoring best weights (best %s=%.4f)", self.metric, self.best_score)
            self.model.load_weights(str(self.filepath))
