"""CalibrationMonitor — Callback Keras para análise de calibração do modelo.

Computa ECE, MCE, Brier Score e reliability diagram por época,
com foco em classes minoritárias (S, V, F).

Uso:
    callbacks = [
        CalibrationMonitor(val_data, val_labels, n_bins=15, log_path="logs/calibration.json"),
        ...
    ]
    model.fit(..., callbacks=callbacks)

Restrições:
    - Apenas numpy/scipy/tensorflow (sem dependências novas).
    - Usa softmax outputs (não argmax).
    - Callback desacoplável.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from tensorflow import keras


class CalibrationMonitor(keras.callbacks.Callback):
    """Callback que monitora calibração do modelo por época.

    Parameters
    ----------
    val_data : np.ndarray
        Dados de validação.
    val_labels : np.ndarray
        Labels de validação (one-hot ou índices inteiros).
    n_bins : int
        Número de bins para ECE/MCE.
    log_path : str
        Caminho para salvar logs JSON.
    class_names : list[str], optional
        Nomes das classes para logging.
    max_samples : int, optional
        Número máximo de amostras de validação a processar por época.
        Se None ou maior/igual ao dataset, usa todo o conjunto.

    Attributes
    ----------
    history : list[dict]
        Histórico de métricas de calibração por época.
    """

    def __init__(
        self,
        val_data: np.ndarray,
        val_labels: np.ndarray,
        n_bins: int = 15,
        log_path: str = "logs/calibration.json",
        class_names: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.val_data = val_data
        self.val_labels = val_labels
        self.n_bins = n_bins
        self.log_path = log_path
        self.class_names = class_names or ["N", "S", "V", "F"]
        self.max_samples = max_samples
        self.history: List[Dict[str, Any]] = []
        self._sample_indices: Optional[np.ndarray] = None

        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        """Amostra dados de validação se max_samples for menor que o dataset."""
        n_samples = self.val_data.shape[0]
        if self.max_samples is not None and 0 < self.max_samples < n_samples:
            rng = np.random.default_rng(seed=42)
            self._sample_indices = rng.choice(n_samples, size=self.max_samples, replace=False)
            self.val_data = self.val_data[self._sample_indices]
            self.val_labels = self.val_labels[self._sample_indices]
            print(
                f"[CalibrationMonitor] Amostrando {self.max_samples} de {n_samples} "
                f"amostras de validação."
            )

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Computa métricas de calibração ao final de cada época."""
        stats = self._compute_calibration_stats(epoch)
        self.history.append(stats)
        self._export_json()

    def _compute_calibration_stats(self, epoch: int) -> Dict[str, Any]:
        """Calcula ECE, MCE, Brier e reliability diagram.

        Returns
        -------
        dict
            Dicionário com métricas de calibração.
        """
        predictions = self.model.predict(self.val_data, verbose=0)

        if len(self.val_labels.shape) > 1 and self.val_labels.shape[-1] > 1:
            labels = np.argmax(self.val_labels, axis=-1)
        else:
            labels = self.val_labels.flatten().astype(int)

        n_classes = predictions.shape[-1]

        ece, mce = self._compute_ece_mce(predictions, labels, n_classes)
        brier = self._compute_brier_score(predictions, labels, n_classes)
        brier_per_class = self._compute_brier_per_class(predictions, labels, n_classes)
        confidence_per_class = self._compute_confidence_per_class(predictions, labels)
        reliability_bins = self._compute_reliability_bins(predictions, labels)

        return {
            "epoch": epoch,
            "ece": float(ece),
            "mce": float(mce),
            "brier_score": float(brier),
            "brier_per_class": brier_per_class,
            "confidence_per_class": confidence_per_class,
            "reliability_bins": reliability_bins,
        }

    def _compute_ece_mce(
        self, predictions: np.ndarray, labels: np.ndarray, n_classes: int
    ) -> tuple[float, float]:
        """Calcula Expected Calibration Error e Maximum Calibration Error."""
        confidences = np.max(predictions, axis=-1)
        predicted_classes = np.argmax(predictions, axis=-1)
        accuracies = (predicted_classes == labels).astype(float)

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece = 0.0
        mce = 0.0

        for i in range(self.n_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]

            if i == self.n_bins - 1:
                in_bin = (confidences >= lower) & (confidences <= upper)
            else:
                in_bin = (confidences >= lower) & (confidences < upper)

            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += calibration_error * prop_in_bin
                mce = max(mce, calibration_error)

        return ece, mce

    def _compute_brier_score(
        self, predictions: np.ndarray, labels: np.ndarray, n_classes: int
    ) -> float:
        """Calcula Brier Score multi-classe."""
        labels_onehot = np.zeros_like(predictions)
        labels_onehot[np.arange(len(labels)), labels] = 1.0
        brier = np.mean(np.sum((predictions - labels_onehot) ** 2, axis=-1))
        return float(brier)

    def _compute_brier_per_class(
        self, predictions: np.ndarray, labels: np.ndarray, n_classes: int
    ) -> Dict[str, float]:
        """Calcula Brier Score desagregado por classe."""
        brier_per_class: Dict[str, float] = {}
        for cls in range(n_classes):
            mask = labels == cls
            name = self.class_names[cls] if cls < len(self.class_names) else f"cls_{cls}"
            if np.sum(mask) == 0:
                brier_per_class[name] = 0.0
                continue

            p_cls = predictions[mask, cls]
            brier = np.mean((1.0 - p_cls) ** 2)
            brier_per_class[name] = float(brier)
        return brier_per_class

    def _compute_confidence_per_class(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Calcula confiança média predita para cada classe verdadeira."""
        confidences = np.max(predictions, axis=-1)
        confidence_per_class: Dict[str, float] = {}

        for cls in range(predictions.shape[-1]):
            mask = labels == cls
            name = self.class_names[cls] if cls < len(self.class_names) else f"cls_{cls}"
            if np.sum(mask) == 0:
                confidence_per_class[name] = 0.0
                continue
            confidence_per_class[name] = float(np.mean(confidences[mask]))
        return confidence_per_class

    def _compute_reliability_bins(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Calcula dados para reliability diagram."""
        confidences = np.max(predictions, axis=-1)
        predicted_classes = np.argmax(predictions, axis=-1)
        accuracies = (predicted_classes == labels).astype(float)

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        reliability_bins: List[Dict[str, Any]] = []

        for i in range(self.n_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]

            if i == self.n_bins - 1:
                in_bin = (confidences >= lower) & (confidences <= upper)
            else:
                in_bin = (confidences >= lower) & (confidences < upper)

            count = int(np.sum(in_bin))
            if count > 0:
                acc = float(np.mean(accuracies[in_bin]))
                conf = float(np.mean(confidences[in_bin]))
            else:
                acc = 0.0
                conf = 0.0

            reliability_bins.append(
                {
                    "bin": i,
                    "lower_edge": float(lower),
                    "upper_edge": float(upper),
                    "accuracy": acc,
                    "confidence": conf,
                    "count": count,
                }
            )

        return reliability_bins

    def _export_json(self) -> None:
        """Exporta histórico para arquivo JSON."""
        with open(self.log_path, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2, ensure_ascii=False)

    def get_alert_summary(self) -> List[str]:
        """Retorna lista de alertas baseados nos thresholds definidos.

        Thresholds:
            - ECE > 0.15: calibração ruim.
            - MCE > 0.30: máxima calibração ruim.
            - Brier > 0.50 para S/V/F: classe não calibrada.
        """
        if not self.history:
            return []

        latest = self.history[-1]
        alerts: List[str] = []

        if latest["ece"] > 0.15:
            alerts.append(f"CALIBRAÇÃO RUIM: ECE = {latest['ece']:.3f} > 0.15")
        if latest["mce"] > 0.30:
            alerts.append(f"MÁXIMA CALIBRAÇÃO RUIM: MCE = {latest['mce']:.3f} > 0.30")

        for cls_name in ["S", "V", "F"]:
            if cls_name in latest["brier_per_class"]:
                brier = latest["brier_per_class"][cls_name]
                if brier > 0.50:
                    alerts.append(f"CLASSE NÃO CALIBRADA: Brier-{cls_name} = {brier:.3f}")

        return alerts
