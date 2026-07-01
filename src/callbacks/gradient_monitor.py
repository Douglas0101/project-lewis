"""GradientMonitor — Callback Keras para análise de gradientes em camadas lineares.

Monitora normas, razões e distribuições de gradientes por época,
com foco em detectar vanishing, exploding e bias de classe.

Uso:
    callbacks = [
        GradientMonitor(val_data, val_labels, log_path="logs/gradients.json"),
        ...
    ]
    model.fit(..., callbacks=callbacks)

Restrições:
    - Não altera arquitetura do modelo.
    - Apenas tensorflow e numpy (sem dependências novas).
    - Callback desacoplável (remover sem quebrar o pipeline).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras


class GradientMonitor(keras.callbacks.Callback):
    """Callback que monitora gradientes em camadas Dense/FC por época.

    Parameters
    ----------
    val_data : np.ndarray
        Dados de validação.
    val_labels : np.ndarray
        Labels de validação (one-hot ou índices inteiros).
    log_path : str
        Caminho para salvar logs JSON.
    layer_names : list[str], optional
        Lista de camadas a monitorar. Se None, monitora todas as Dense.
    max_samples : int, optional
        Número máximo de amostras de validação a processar por época.
        Se None ou maior/igual ao dataset, usa todo o conjunto.

    Attributes
    ----------
    history : list[dict]
        Histórico de estatísticas por época.
    """

    def __init__(
        self,
        val_data: np.ndarray,
        val_labels: np.ndarray,
        log_path: str = "logs/gradients.json",
        layer_names: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.val_data = val_data
        self.val_labels = val_labels
        self.log_path = log_path
        self.layer_names = layer_names
        self.max_samples = max_samples
        self.class_names = class_names
        self.history: List[Dict[str, Any]] = []
        self._sample_indices: Optional[np.ndarray] = None

        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        """Identifica camadas Dense e amostra dados de validação se necessário."""
        if self.layer_names is None:
            self.layer_names = [
                layer.name for layer in self.model.layers if isinstance(layer, keras.layers.Dense)
            ]

        n_samples = self.val_data.shape[0]
        if self.max_samples is not None and 0 < self.max_samples < n_samples:
            rng = np.random.default_rng(seed=42)
            self._sample_indices = rng.choice(n_samples, size=self.max_samples, replace=False)
            self.val_data = self.val_data[self._sample_indices]
            self.val_labels = self.val_labels[self._sample_indices]
            print(
                f"[GradientMonitor] Amostrando {self.max_samples} de {n_samples} "
                f"amostras de validação."
            )

        print(f"[GradientMonitor] Monitorando camadas: {self.layer_names}")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Computa estatísticas de gradiente ao final de cada época."""
        stats = self._compute_gradient_stats(epoch)
        self.history.append(stats)
        self._export_json()

    def _compute_gradient_stats(self, epoch: int) -> Dict[str, Any]:
        """Calcula métricas de gradiente para cada camada monitorada.

        Returns
        -------
        dict
            Dicionário com métricas por camada e médias globais.
        """
        epoch_stats: Dict[str, Any] = {"epoch": epoch, "layers": []}

        with tf.GradientTape(persistent=True) as tape:
            predictions = self.model(self.val_data, training=False)
            loss = self._loss_for_labels(self.val_labels, predictions)
            mean_loss = tf.reduce_mean(loss)

        for layer_name in self.layer_names or []:
            layer = self.model.get_layer(layer_name)
            trainable_vars = layer.trainable_variables
            if not trainable_vars:
                continue

            weights = trainable_vars[0]  # kernel
            biases = trainable_vars[1] if len(trainable_vars) > 1 else None

            grads = tape.gradient(mean_loss, weights)
            if grads is None:
                continue

            grad_norm = float(tf.norm(grads).numpy())
            weight_norm = float(tf.norm(weights).numpy())
            norm_ratio = grad_norm / (weight_norm + 1e-10)
            grads_np = grads.numpy()
            p95_gradient = float(np.percentile(np.abs(grads_np), 95))
            grad_mean = float(tf.reduce_mean(grads).numpy())
            grad_std = float(tf.math.reduce_std(grads).numpy())

            grad_per_class = self._gradient_mean_per_class(tape, weights, predictions, biases)

            layer_stats = {
                "layer_name": layer_name,
                "l2_norm_mean": grad_norm,
                "weight_norm": weight_norm,
                "norm_ratio": norm_ratio,
                "p95_gradient": p95_gradient,
                "gradient_mean": grad_mean,
                "gradient_std": grad_std,
                "gradient_mean_per_class": grad_per_class,
            }
            epoch_stats["layers"].append(layer_stats)

        del tape
        return epoch_stats

    def _loss_for_labels(self, labels: np.ndarray, predictions: tf.Tensor) -> tf.Tensor:
        """Seleciona a função de perda conforme o formato dos labels."""
        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            return keras.losses.categorical_crossentropy(labels, predictions)
        return keras.losses.sparse_categorical_crossentropy(
            tf.cast(tf.reshape(labels, [-1]), tf.int32), predictions
        )

    def _gradient_mean_per_class(
        self,
        tape: tf.GradientTape,
        weights: tf.Variable,
        predictions: tf.Tensor,
        biases: Optional[tf.Variable] = None,
    ) -> Dict[str, float]:
        """Calcula a média do gradiente absoluto ponderada por classe.

        Usa sample weights para isolar o gradiente de cada classe.
        """
        if len(self.val_labels.shape) > 1 and self.val_labels.shape[-1] > 1:
            labels_idx = tf.argmax(self.val_labels, axis=-1)
        else:
            labels_idx = tf.cast(tf.reshape(self.val_labels, [-1]), tf.int32)

        n_classes = int(tf.reduce_max(labels_idx).numpy()) + 1
        default_class_names = [f"cls_{cls}" for cls in range(n_classes)]
        if self.class_names is not None:
            provided_names = list(self.class_names)
        else:
            provided_names = ["N", "S", "V", "F"]
        class_names = provided_names + default_class_names[len(provided_names):]

        grad_per_class: Dict[str, float] = {}
        for cls in range(n_classes):
            mask = tf.cast(tf.equal(labels_idx, cls), tf.float32)
            weighted_loss = tf.reduce_mean(
                self._loss_for_labels(self.val_labels, predictions) * mask
            )
            grad_cls = tape.gradient(weighted_loss, weights)
            name = class_names[cls] if cls < len(class_names) else f"cls_{cls}"
            if grad_cls is not None:
                grad_per_class[name] = float(tf.reduce_mean(tf.abs(grad_cls)).numpy())
            else:
                grad_per_class[name] = 0.0

        return grad_per_class

    def _export_json(self) -> None:
        """Exporta histórico para arquivo JSON."""
        with open(self.log_path, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2, ensure_ascii=False)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Retorna resumo das métricas de gradiente.

        Returns
        -------
        dict
            Média das normas e razões por camada monitorada.
        """
        if not self.history:
            return {}

        summary: Dict[str, Dict[str, float]] = {}
        for layer_name in self.layer_names or []:
            norms = [
                entry["l2_norm_mean"]
                for epoch in self.history
                for entry in epoch["layers"]
                if entry["layer_name"] == layer_name
            ]
            ratios = [
                entry["norm_ratio"]
                for epoch in self.history
                for entry in epoch["layers"]
                if entry["layer_name"] == layer_name
            ]
            summary[layer_name] = {
                "mean_norm": float(np.mean(norms)) if norms else 0.0,
                "mean_norm_ratio": float(np.mean(ratios)) if ratios else 0.0,
                "min_norm_ratio": float(np.min(ratios)) if ratios else 0.0,
            }
        return summary
