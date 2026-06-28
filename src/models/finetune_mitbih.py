"""Fine-tuning em MIT-BIH+ com transfer learning.

Congela camadas convolucionais, retreina classifier com class weights.
Loss: sparse_categorical_crossentropy | Activation: softmax
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
from sklearn.utils.class_weight import compute_class_weight

from src.data.augmentation import oversample_class, oversample_per_class

from .backbone_1d import freeze_conv_layers, save_model_config

LOGGER = logging.getLogger("lewis.camada04.finetune")


def _maybe_import_slha():
    """Importa o SLHA apenas quando necessário (lazy)."""
    from src.models import slha

    return slha


def _set_seeds(seed: int = 42) -> None:
    """Fixa seeds para reprodutibilidade."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _compute_class_weights(
    y_train: np.ndarray,
    max_weight: float = 20.0,
) -> Dict[int, float]:
    """Calcula class weights suavizados para focal loss / referência.

    Parameters
    ----------
    y_train : np.ndarray
        Labels inteiros.
    max_weight : float
        Limite máximo para qualquer peso.

    Returns
    -------
    dict
        {class_index: weight}
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    weights = np.power(weights / weights.min(), 0.75)
    weights = np.minimum(weights, max_weight)
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """Focal loss para labels inteiros (sparse).

    Reduz o peso dos exemplos fáceis (classe N) e foca nos exemplos difíceis
    e nas classes minoritárias (S, V, F, Q).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[np.ndarray] = None,
        from_logits: bool = False,
        name: str = "sparse_categorical_focal_loss",
    ):
        super().__init__(name=name)
        self.gamma = gamma
        self.from_logits = from_logits
        self.alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.keras.losses.sparse_categorical_crossentropy(
            y_true,
            y_pred,
            from_logits=self.from_logits,
        )
        # Probabilidade atribuída à classe verdadeira
        prob_true = tf.gather(y_pred, y_true, batch_dims=1)
        focal_weight = tf.pow(1.0 - prob_true, self.gamma)
        if self.alpha is not None:
            alpha_t = tf.gather(self.alpha, y_true)
            return alpha_t * focal_weight * ce
        return focal_weight * ce

    def get_config(self) -> dict:
        config = super().get_config()
        alpha = None
        if self.alpha is not None:
            if isinstance(self.alpha, np.ndarray):
                alpha = self.alpha.tolist()
            else:
                alpha = float(self.alpha)
        config.update(
            {
                "gamma": self.gamma,
                "from_logits": self.from_logits,
                "alpha": alpha,
            }
        )
        return config


def _augment(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Data augmentation leve para ECG (aplicada no treino)."""
    # Escala de amplitude [0.9, 1.1]
    scale = tf.random.uniform([], minval=0.9, maxval=1.1, dtype=x.dtype)
    x = x * scale
    # Deslocamento temporal pequeno (até ±10 amostras)
    shift = tf.random.uniform([], minval=-10, maxval=10, dtype=tf.int32)
    x = tf.roll(x, shift, axis=0)
    # Ruído gaussiano pequeno
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.01, dtype=x.dtype)
    x = x + noise
    return x, y


def _build_train_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    augment: bool = True,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Cria dataset de treino com shuffle e augmentation opcional.

    Mantém a distribuição original das classes; o balanceamento é feito via
    ``class_weight`` no ``model.fit`` para não forçar o modelo a superestimar
    as minoritárias durante a validação imbalanciada.

    Parameters
    ----------
    X : np.ndarray
        Dados, shape (n, 500, 1).
    y : np.ndarray
        Labels inteiros.
    batch_size : int
        Tamanho do batch.
    augment : bool
        Se True, aplica augmentation leve.
    seed : int, optional
        Seed para reproducibilidade.

    Returns
    -------
    tf.data.Dataset
        Dataset shuffle + batched + prefetched.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(buffer_size=max(1, len(y)), seed=seed, reshuffle_each_iteration=True)
    ds = ds.repeat()
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class F1MacroCheckpoint(tf.keras.callbacks.Callback):
    """Salva o melhor modelo segundo F1-macro AAMI na validação.

    Substitui o ModelCheckpoint baseado em ``val_loss``, que é distorcido pelos
    class weights. Monitora diretamente a métrica alvo do QG5.
    """

    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        filepath: Path,
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
        from .evaluate import (
            evaluate_aami,
            find_best_threshold,
            find_best_thresholds_multiclass,
        )

        y_proba = self.model.predict(self.X_val, verbose=0)

        # Para classificação binária, busca threshold que maximize F1-macro/QG5.
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
            else (
                str(thresholds_dict)
                if thresholds_dict is not None
                else "argmax"
            )
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
        if self.filepath.exists():
            LOGGER.info("Restoring best weights (best %s=%.4f)", self.metric, self.best_score)
            self.model.load_weights(str(self.filepath))


def finetune_mitbih(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    seed: int = 42,
    experiment_dir: Optional[Path] = None,
    monitor: str = "val_loss",
    freeze_backbone: bool = True,
    class_names: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    class_weight: Optional[Dict[int, float]] = None,
    selection_metric: str = "F1_macro",
    use_slha: bool = False,
    augment_class: Optional[int] = None,
    augment_factor: int = 1,
    augment_config: Optional[Dict[str, Any]] = None,
    loss: str | tf.keras.losses.Loss = "sparse_categorical_crossentropy",
    optimize_thresholds: bool = False,
) -> Tuple[tf.keras.Model, dict]:
    """Fine-tuning com backbone opcionalmente congelado.

    Parameters
    ----------
    model : tf.keras.Model
        Backbone (opcionalmente pré-treinado).
    X_train : np.ndarray
        Dados de treino (shape: (n, 500, 1)).
    y_train : np.ndarray
        Labels inteiros (shape: (n,)).
    X_val : np.ndarray
        Dados de validação.
    y_val : np.ndarray
        Labels de validação.
    epochs : int
        Épocas máximas.
    batch_size : int
        Tamanho do batch.
    learning_rate : float
        Taxa de aprendizado (menor que pré-treino).
    seed : int
        Seed.
    experiment_dir : Path, optional
        Diretório do experimento.
    monitor : str
        Métrica para early stopping.
    freeze_backbone : bool
        Se True, congela camadas convolucionais antes do treino.
    class_names : list[str], optional
        Nomes das classes para avaliação AAMI no callback F1MacroCheckpoint.
    thresholds : dict, optional
        Thresholds configuráveis para ``evaluate_aami`` no callback.
    class_weight : dict, optional
        Pesos por classe para ``model.fit``. Se None, calcula automaticamente.
    selection_metric : str
        Métrica usada pelo callback para salvar/restaurar melhor modelo.
        Ex.: "F1_macro", "Se_Anormal", "F1_V".
    augment_class : int, optional
        Índice da classe a ser oversampled durante o treino (ex.: 2 para F).
        Mantido para compatibilidade; ``augment_config`` tem precedência.
    augment_factor : int
        Fator de oversampling. factor=1 desativa.
    augment_config : dict, optional
        Configuração de oversampling por classe para augmentation avançada.
        Exemplo: ``{"2": {"factor": 8, "methods": ["jitter", "baseline_wander"],
        "intensity": "high"}}``.
    loss : str or tf.keras.losses.Loss
        Função de perda. Pode ser ``"sparse_categorical_crossentropy"`` ou uma
        instância de ``SparseCategoricalFocalLoss``.
    optimize_thresholds : bool
        Se True, realiza threshold tuning one-vs-rest na validação para
        classificação multiclasse.

    Returns
    -------
    tuple
        (model, history_dict)
    """
    _set_seeds(seed)

    if experiment_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path("experiments") / f"exp_{ts}_finetune_mitbih"
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Fine-tuning MIT-BIH+ | experiment_dir=%s | epochs=%d | lr=%.1e | batch=%d",
        experiment_dir,
        epochs,
        learning_rate,
        batch_size,
    )

    # Congelar camadas conv apenas quando solicitado (transfer learning)
    if freeze_backbone:
        model = freeze_conv_layers(model)

    # Oversampling de classe minoritária (ex.: F) antes de construir dataset
    if augment_config is not None:
        LOGGER.info(
            "Class-specific augmentation | config=%s | train size before=%d",
            augment_config,
            len(X_train),
        )
        X_train, y_train = oversample_per_class(
            X_train, y_train, config=augment_config, seed=seed
        )
        LOGGER.info("Train size after class-specific augmentation=%d", len(X_train))
    elif augment_class is not None and augment_factor > 1:
        LOGGER.info(
            "Oversampling class %d by factor %d | train size before=%d",
            augment_class,
            augment_factor,
            len(X_train),
        )
        X_train, y_train = oversample_class(
            X_train, y_train, class_idx=augment_class, factor=augment_factor
        )
        LOGGER.info("Train size after oversampling=%d", len(X_train))

    # Dataset de treino com augmentation (mantém distribuição original)
    train_ds = _build_train_dataset(
        X_train, y_train, batch_size=batch_size, augment=True, seed=seed
    )
    steps_per_epoch = len(X_train) // batch_size
    LOGGER.info("Train dataset | steps_per_epoch=%d | augment=True", steps_per_epoch)

    # Pesos de classe para compensar desbalanceamento AAMI
    if class_weight is None:
        class_weight = _compute_class_weights(y_train)
    LOGGER.info("Class weights: %s", class_weight)

    # Compilar
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )

    # Callbacks: usar F1-macro AAMI como critério principal de seleção
    # (val_loss é distorcido pelos class weights).
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        F1MacroCheckpoint(
            X_val=X_val,
            y_val=y_val,
            filepath=experiment_dir / "best_weights.weights.h5",
            class_names=class_names,
            thresholds=thresholds,
            metric=selection_metric,
            patience=15,
            optimize_thresholds=optimize_thresholds,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(experiment_dir / "training.log"),
            separator=",",
            append=False,
        ),
    ]

    # SLHA opt-in: auto-configura batch size e adiciona monitor de recursos
    if use_slha:
        slha = _maybe_import_slha()
        n_sample = min(8, len(X_train))
        config = slha.auto_configure_training(
            X_sample=X_train[:n_sample],
            y_sample=y_train[:n_sample],
            model=model,
            reference_batch_size=batch_size,
        )
        batch_size = config.batch_size
        LOGGER.info("SLHA config: %s", config.model_dump_json())
        callbacks.append(slha.ResourceMonitor())

    # Treinar
    history = model.fit(
        train_ds,
        validation_data=(X_val, y_val),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # Salvar config
    save_model_config(
        model,
        experiment_dir / "config.json",
        extra={
            "stage": "finetune_mitbih",
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "class_weight": class_weight,
            "monitor": monitor,
        },
    )

    # Salvar metrics
    metrics = {
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history.get("val_loss", [np.nan])[-1]),
        "final_acc": float(history.history.get("accuracy", [np.nan])[-1]),
        "final_val_acc": float(history.history.get("val_accuracy", [np.nan])[-1]),
        "final_val_auc": float(history.history.get("val_auc", [np.nan])[-1]),
        "stopped_epoch": len(history.history["loss"]),
    }
    with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    # Salvar modelo completo (já com os melhores pesos restaurados pelo callback)
    model.save(str(experiment_dir / "model.keras"), save_format="keras")
    LOGGER.info("Full model saved to %s", experiment_dir / "model.keras")

    LOGGER.info(
        "Fine-tuning concluído | loss=%.4f | val_loss=%.4f | val_acc=%.4f",
        metrics["final_loss"],
        metrics["final_val_loss"],
        metrics["final_val_acc"],
    )
    return model, history.history
