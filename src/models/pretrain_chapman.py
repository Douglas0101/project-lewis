"""Pré-treino multi-label em Chapman-Shaoxing (SCP-ECG superclasses).

Labels: 5 superclasses (NORM, CD, MI, HYP, STTC) — one-hot multi-label.
Loss: binary_crossentropy | Activation: sigmoid
"""

from __future__ import annotations

import json
import logging
import os
import random
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
import yaml

from .backbone_1d import build_backbone_1d_multilabel, save_model_config

LOGGER = logging.getLogger("lewis.camada04.pretrain")


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


def _make_callbacks(
    experiment_dir: Path,
    patience_es: int = 5,
    patience_lr: int = 3,
) -> list:
    """Cria callbacks padrão para pré-treino."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(experiment_dir / "backbone_pretrained.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(experiment_dir / "training.log"),
            separator=",",
            append=False,
        ),
    ]


def pretrain_chapman(
    data_generator: Optional[Callable] = None,
    val_generator: Optional[Callable] = None,
    train_dataset: Optional[tf.data.Dataset] = None,
    val_dataset: Optional[tf.data.Dataset] = None,
    steps_per_epoch: int = 1000,
    validation_steps: Optional[int] = None,
    epochs: int = 30,
    batch_size: int = 64,
    input_len: int = 500,
    num_classes: int = 5,
    learning_rate: float = 1e-3,
    seed: int = 42,
    experiment_dir: Optional[Path] = None,
    use_slha: bool = False,
) -> Tuple[tf.keras.Model, dict]:
    """Pré-treina backbone em Chapman-Shaoxing (multi-label).

    Parameters
    ----------
    data_generator : Callable, optional
        Generator que yield (X_batch, y_batch).
    val_generator : Callable, optional
        Generator de validação.
    train_dataset : tf.data.Dataset, optional
        Dataset de treino (preferido, mais eficiente que generator).
    val_dataset : tf.data.Dataset, optional
        Dataset de validação (preferido).
    steps_per_epoch : int
        Passos por época.
    validation_steps : int, optional
        Passos de validação.
    epochs : int
        Épocas máximas.
    batch_size : int
        Tamanho do batch.
    input_len : int
        Comprimento do segmento.
    num_classes : int
        Número de superclasses SCP-ECG.
    learning_rate : float
        Taxa de aprendizado.
    seed : int
        Seed para reprodutibilidade.
    experiment_dir : Path, optional
        Diretório do experimento. Se None, cria um novo.

    Returns
    -------
    tuple
        (model, history_dict)
    """
    if train_dataset is None and data_generator is None:
        raise ValueError("Forneça train_dataset ou data_generator")
    _set_seeds(seed)

    if experiment_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path("experiments") / f"exp_{ts}_pretrain_chapman"
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Pré-treino Chapman | experiment_dir=%s | epochs=%d | lr=%.1e",
        experiment_dir,
        epochs,
        learning_rate,
    )

    # Criar modelo
    model = build_backbone_1d_multilabel(
        input_len=input_len,
        num_classes=num_classes,
        name="lewis_backbone_pretrain",
    )

    # Compilar
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc_roc", curve="ROC", multi_label=True),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR", multi_label=True),
        ],
    )

    # Salvar summary
    summary_path = experiment_dir / "model_summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        model.summary(print_fn=lambda x: fh.write(x + "\n"))

    # Callbacks
    callbacks = _make_callbacks(experiment_dir)

    # SLHA opt-in: auto-configura batch size e adiciona monitor de recursos
    if use_slha:
        slha = _maybe_import_slha()
        if train_dataset is not None:
            for X_sample, y_sample in train_dataset.take(1):
                X_sample = X_sample.numpy()
                y_sample = y_sample.numpy()
                break
        else:
            assert data_generator is not None
            gen = data_generator()
            X_sample, y_sample = next(gen)
        config = slha.auto_configure_training(
            X_sample=X_sample[:8],
            y_sample=y_sample[:8],
            model=model,
            reference_batch_size=batch_size,
            log_dir=experiment_dir / "slha",
        )
        batch_size = config.batch_size
        LOGGER.info("SLHA config: %s", config.model_dump_json())
        callbacks.append(
            slha.ResourceMonitor(log_path=experiment_dir / "slha" / "resource_logs.jsonl")
        )

    # Treinar
    fit_kwargs = {
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "validation_steps": validation_steps,
        "callbacks": callbacks,
        "verbose": 2,
    }
    if train_dataset is not None:
        fit_kwargs["x"] = train_dataset
        fit_kwargs["validation_data"] = val_dataset
    else:
        assert data_generator is not None
        fit_kwargs["x"] = data_generator()
        fit_kwargs["validation_data"] = val_generator() if val_generator else None

    history = model.fit(**fit_kwargs)

    # Salvar config
    save_model_config(
        model,
        experiment_dir / "config.json",
        extra={
            "stage": "pretrain_chapman",
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_classes": num_classes,
            "input_len": input_len,
        },
    )

    # Salvar metrics
    metrics = {
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history.get("val_loss", [np.nan])[-1]),
        "final_auc_roc": float(history.history.get("auc_roc", [np.nan])[-1]),
        "final_val_auc_roc": float(history.history.get("val_auc_roc", [np.nan])[-1]),
        "stopped_epoch": len(history.history["loss"]),
    }
    with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    LOGGER.info(
        "Pré-treino concluído | loss=%.4f | val_loss=%.4f",
        metrics["final_loss"],
        metrics["final_val_loss"],
    )
    return model, history.history


def load_pretrained_backbone(
    weights_path: Path,
    input_len: int = 500,
    num_classes: int = 5,
    for_finetune: bool = True,
) -> tf.keras.Model:
    """Carrega backbone pré-treinado e opcionalmente congela camadas conv.

    Parameters
    ----------
    weights_path : Path
        Caminho para .keras ou .weights.h5.
    input_len : int
        Comprimento do segmento.
    num_classes : int
        Número de classes (5 para AAMI).
    for_finetune : bool
        Se True, congela camadas convolucionais.

    Returns
    -------
    tf.keras.Model
        Modelo carregado.
    """
    from .backbone_1d import build_backbone_1d, freeze_conv_layers

    model = build_backbone_1d(input_len=input_len, num_classes=num_classes)
    model.load_weights(str(weights_path))

    if for_finetune:
        model = freeze_conv_layers(model)
        LOGGER.info("Backbone carregado e camadas conv congeladas para fine-tuning")
    else:
        LOGGER.info("Backbone carregado (todas as camadas treináveis)")

    return model


def _load_config(config_path: Path) -> dict:
    """Load YAML training config."""
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    """CLI entry point for Chapman pre-training."""
    parser = argparse.ArgumentParser(description="Pré-treino Project-Lewis em Chapman-Shaoxing")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pretrain_v1.0.yaml"),
        help="Caminho para config/pretrain_v*.yaml",
    )
    parser.add_argument(
        "--use-slha",
        action="store_true",
        help="Ativar Self-Learning Hardware Adapter (auto batch size + monitor)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limitar número de registros Chapman (smoke test)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Sobrescrever número de épocas do config",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Sobrescrever steps_per_epoch do config",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    cfg = _load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    from .chapman_dataset import chapman_train_val_split

    train_ds, val_ds = chapman_train_val_split(
        val_ratio=0.1,
        batch_size=train_cfg["batch_size"],
        segment_len=model_cfg["input_len"],
        seed=train_cfg["seed"],
    )

    if args.max_records:
        # Quick smoke test: limit catalog via environment override handled in dataset
        LOGGER.warning("--max-records não implementado para tf.data; use com cautela")

    # Estimate steps per epoch from dataset size if not in config
    n_train = sum(1 for _ in train_ds)
    n_val = sum(1 for _ in val_ds)
    if args.steps_per_epoch is not None:
        steps_per_epoch = args.steps_per_epoch
    else:
        steps_per_epoch = train_cfg.get("steps_per_epoch", n_train)
    validation_steps = train_cfg.get("validation_steps", n_val)
    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    LOGGER.info(
        "Steps per epoch: %d | Validation steps: %d | Epochs: %d",
        steps_per_epoch,
        validation_steps,
        epochs,
    )

    experiment_dir = Path("experiments") / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_pretrain_chapman"
    )

    model, history = pretrain_chapman(
        train_dataset=train_ds.repeat(),
        val_dataset=val_ds.repeat(),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        batch_size=train_cfg["batch_size"],
        input_len=model_cfg["input_len"],
        num_classes=model_cfg["num_classes"],
        learning_rate=train_cfg["learning_rate"],
        seed=train_cfg["seed"],
        experiment_dir=experiment_dir,
        use_slha=args.use_slha,
    )

    # QG4 validation
    final_val_auc = history.get("val_auc_roc", [np.nan])[-1]
    final_val_loss = history.get("val_loss", [np.nan])[-1]
    qg4_pass = (
        final_val_auc > cfg["quality_gate"]["qg4"]["min_val_auc_roc_macro"]
        and final_val_loss < cfg["quality_gate"]["qg4"]["max_val_loss"]
    )
    LOGGER.info(
        "QG4 | val_auc_roc=%.4f | val_loss=%.4f | pass=%s",
        final_val_auc,
        final_val_loss,
        qg4_pass,
    )

    if qg4_pass:
        # Copy best model to canonical path
        canonical = Path("models") / "backbone_pretrained_v1.0.keras"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        best_path = experiment_dir / "backbone_pretrained.keras"
        shutil.copy(str(best_path), str(canonical))
        LOGGER.info("Backbone copiado para %s", canonical)

    return 0 if qg4_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
