"""Pré-treino multi-label em Chapman-Shaoxing (SCP-ECG superclasses).

Labels: 5 superclasses (NORM, CD, MI, HYP, STTC) — one-hot multi-label.
Loss: binary_crossentropy | Activation: sigmoid
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
import yaml

from .backbone_1d import save_model_config

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


def build_datasets(
    val_ratio: float = 0.1,
    batch_size: int = 64,
    segment_len: int = 500,
    seed: int = 42,
    steps_per_epoch: Optional[int] = None,
    validation_steps: Optional[int] = None,
    catalog_path: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """Assemble train/val datasets with repeat + explicit step counts.

    Both datasets are repeated: Keras then stops each phase by step count and
    never relies on StopIteration — running a generator dataset to exhaustion
    makes Keras 3 emit a false ``Your input ran out of data`` warning.

    Returns ``(train_ds_repeated, val_ds_repeated, steps_per_epoch,
    validation_steps)`` with cardinality logging.
    """
    from .chapman_dataset import (
        chapman_split_record_sets,
        chapman_train_val_split,
        estimate_n_segments,
    )

    train_ds, val_ds = chapman_train_val_split(
        val_ratio=val_ratio,
        batch_size=batch_size,
        segment_len=segment_len,
        seed=seed,
        catalog_path=catalog_path,
        processed_dir=processed_dir,
    )
    train_set, val_set = chapman_split_record_sets(
        val_ratio=val_ratio, seed=seed, catalog_path=catalog_path
    )
    est_common = {"catalog_path": catalog_path, "processed_dir": processed_dir}
    est_train = estimate_n_segments(train_set, segment_len=segment_len, **est_common)
    est_val = estimate_n_segments(val_set, segment_len=segment_len, **est_common)
    n_train = -(-est_train // batch_size)
    n_val = -(-est_val // batch_size)

    steps = steps_per_epoch or n_train
    val_steps = validation_steps or n_val
    LOGGER.info(
        "Cardinality | train_batches=%d | val_batches=%d | batch_size=%d | "
        "steps_per_epoch=%d | validation_steps=%d",
        n_train,
        n_val,
        batch_size,
        steps,
        val_steps,
    )
    return train_ds.repeat(), val_ds.repeat(), steps, val_steps


def qg4_passes(best: dict, qg4_cfg: dict) -> bool:
    """QG4: best-epoch val_auc_roc > min AND val_loss < max (strict operators)."""
    return bool(
        best["val_auc_roc"] > qg4_cfg["min_val_auc_roc_macro"]
        and best["val_loss"] < qg4_cfg["max_val_loss"]
    )


def _best_epoch_metrics(history: dict, loss_key: str = "val_loss") -> dict:
    """Return metrics of the best epoch (lowest ``loss_key``).

    QG4 must judge the best checkpoint (restored by EarlyStopping and saved
    by ModelCheckpoint), not the final epoch. For runs trained with a
    non-BCE loss (focal/weighted), ``loss_key`` must point at the BCE
    monitor so the gate compares the same metric across variants.
    """
    val_loss = history.get(loss_key) or []
    if not val_loss:
        return {"best_epoch": 0, "val_loss": float("nan"), "val_auc_roc": float("nan")}
    best_idx = int(np.argmin(val_loss))
    val_auc = history.get("val_auc_roc") or [float("nan")] * len(val_loss)
    return {
        "best_epoch": best_idx + 1,
        "val_loss": float(val_loss[best_idx]),
        "val_auc_roc": float(val_auc[best_idx]),
    }


def _sample_one_batch(
    train_dataset: Optional[tf.data.Dataset],
    data_generator: Optional[Callable],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrai um batch de amostras para configuracao do SLHA."""
    if train_dataset is not None:
        for X_sample, y_sample in train_dataset.take(1):
            return X_sample.numpy(), y_sample.numpy()
    assert data_generator is not None
    gen = data_generator()
    X_sample, y_sample = next(gen)
    return X_sample, y_sample


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
    architecture: str = "a0",
    loss_name: str = "bce",
    pos_weight: Optional[np.ndarray] = None,
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

    # Criar modelo via factory (A0 congelada por padrão)
    from .backbones import BackboneSpec, build_backbone
    from .pretrain_losses import build_loss

    model = build_backbone(
        BackboneSpec(arch=architecture, input_len=input_len, num_classes=num_classes)
    )

    # Compilar: perdas não-BCE ganham um monitor BCE para o QG4 comparar
    # a mesma métrica entre variantes (gate é definido sobre BCE).
    metrics = [
        tf.keras.metrics.AUC(name="auc_roc", curve="ROC", multi_label=True),
        tf.keras.metrics.AUC(name="auc_pr", curve="PR", multi_label=True),
    ]
    if loss_name != "bce":
        metrics.append(tf.keras.losses.BinaryCrossentropy(name="bce_monitor"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=build_loss(loss_name, pos_weight=pos_weight),
        metrics=metrics,
    )

    # Salvar summary
    summary_path = experiment_dir / "model_summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:

        def _write_line(line: str) -> None:
            fh.write(line + "\n")

        model.summary(print_fn=_write_line)

    # Callbacks
    callbacks = _make_callbacks(experiment_dir)

    # SLHA opt-in: auto-configura batch size e adiciona monitor de recursos
    if use_slha:
        slha = _maybe_import_slha()
        X_sample, y_sample = _sample_one_batch(train_dataset, data_generator)
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
            "architecture": architecture,
            "loss": loss_name,
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
        "--promote",
        action="store_true",
        help="Copiar backbone para models/ quando QG4 passar (default: bloqueado)",
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
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=None,
        help="Sobrescrever validation_steps do config (default: validação completa)",
    )
    parser.add_argument(
        "--architecture",
        choices=["a0", "a1", "a2"],
        default=None,
        help="Variante de backbone (default: config ou a0)",
    )
    parser.add_argument(
        "--loss",
        choices=["bce", "bce_weighted", "focal"],
        default=None,
        help="Loss de treino (default: config ou bce)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sobrescrever seed do config",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    cfg = _load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    det_mode = str(cfg.get("deterministic", {}).get("mode", "fast"))
    from .chapman_dataset import chapman_split_record_sets
    from .pretrain_provenance import apply_deterministic_mode

    apply_deterministic_mode(det_mode)

    batch_size = train_cfg["batch_size"]
    segment_len = model_cfg["input_len"]
    seed = args.seed if args.seed is not None else train_cfg["seed"]
    architecture = args.architecture or str(cfg.get("architecture", "a0"))
    loss_name = args.loss or str(train_cfg.get("loss_variant", "bce"))

    if args.max_records:
        # Quick smoke test: limit catalog via environment override handled in dataset
        LOGGER.warning("--max-records não implementado para tf.data; use com cautela")

    steps_arg = (
        args.steps_per_epoch
        if args.steps_per_epoch is not None
        else train_cfg.get("steps_per_epoch")
    )
    # null/ausente no config → validação completa por época (= n_val estimado)
    val_steps_arg = (
        args.validation_steps
        if args.validation_steps is not None
        else train_cfg.get("validation_steps")
    )
    train_ds, val_ds, steps_per_epoch, validation_steps = build_datasets(
        val_ratio=0.1,
        batch_size=batch_size,
        segment_len=segment_len,
        seed=seed,
        steps_per_epoch=steps_arg,
        validation_steps=val_steps_arg,
    )
    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    LOGGER.info(
        "Epochs: %d | architecture=%s | loss=%s | seed=%d",
        epochs,
        architecture,
        loss_name,
        seed,
    )

    experiment_dir = Path("experiments") / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_pretrain_chapman"
    )

    # pos_weight SOMENTE do split de treino (regra 5 — nunca val/teste)
    pos_weight = None
    if loss_name == "bce_weighted":
        from src.data.chapman_labels import diagnosis_string_to_multihot
        from src.models.chapman_dataset import _load_catalog
        from src.models.pretrain_losses import estimate_pos_weights

        train_set_pw, _ = chapman_split_record_sets(val_ratio=0.1, seed=seed)
        multihots = {
            str(r["record_name"]): diagnosis_string_to_multihot(
                str(r.get("diagnosis", ""))
            )
            for r in _load_catalog()
            if r.get("dataset") == "chapman" and str(r["record_name"]) in train_set_pw
        }
        pos_weight = estimate_pos_weights(train_set_pw, multihots)

    model, history = pretrain_chapman(
        train_dataset=train_ds,
        val_dataset=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        batch_size=batch_size,
        input_len=segment_len,
        num_classes=model_cfg["num_classes"],
        learning_rate=train_cfg["learning_rate"],
        seed=seed,
        experiment_dir=experiment_dir,
        use_slha=args.use_slha,
        architecture=architecture,
        loss_name=loss_name,
        pos_weight=pos_weight,
    )

    # QG4 validation: judge the best checkpoint, not the final epoch.
    # Non-BCE losses are judged on the BCE monitor (same metric for all variants).
    gate_loss_key = "val_bce_monitor" if loss_name != "bce" else "val_loss"
    best = _best_epoch_metrics(history, loss_key=gate_loss_key)
    qg4_pass = qg4_passes(best, cfg["quality_gate"]["qg4"])
    LOGGER.info(
        "QG4 | best_epoch=%d | val_auc_roc=%.4f | val_loss=%.4f | pass=%s",
        best["best_epoch"],
        best["val_auc_roc"],
        best["val_loss"],
        qg4_pass,
    )

    # FASE 4: proveniência + métricas por classe (avaliação, nunca treino)
    from .pretrain_provenance import write_provenance_and_metrics

    train_set, val_set = chapman_split_record_sets(val_ratio=0.1, seed=seed)
    write_provenance_and_metrics(
        experiment_dir=experiment_dir,
        model=model,
        history=history,
        val_dataset=val_ds,
        validation_steps=validation_steps,
        seed=seed,
        deterministic_mode=det_mode,
        train_records=len(train_set),
        val_records=len(val_set),
        training_info={
            "epochs": epochs,
            "batch_size": batch_size,
            "steps_per_epoch": steps_per_epoch,
            "validation_steps": validation_steps,
            "lr_initial": train_cfg["learning_rate"],
            "optimizer": train_cfg["optimizer"],
            "loss": loss_name,
            "architecture": architecture,
        },
        best=best,
        qg4_pass=qg4_pass,
    )

    # run_status.json + qg4_result.json: execution_success separado de qg4_pass
    from .pretrain_provenance import write_gate_and_status

    write_gate_and_status(
        experiment_dir=experiment_dir,
        best=best,
        qg4_cfg=cfg["quality_gate"]["qg4"],
        qg4_pass=qg4_pass,
        model_promoted=bool(qg4_pass and args.promote),
        known_issues=["focal loss: QG4 julga val_bce_monitor"] if loss_name != "bce" else [],
    )

    if qg4_pass and args.promote:
        # Copy best model to canonical path (opt-in: E07R congela models/ por hash)
        canonical = Path("models") / "backbone_pretrained_v1.0.keras"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        best_path = experiment_dir / "backbone_pretrained.keras"
        shutil.copy(str(best_path), str(canonical))
        LOGGER.info("Backbone copiado para %s", canonical)
    elif qg4_pass:
        LOGGER.info(
            "QG4 passou; promoção para models/ bloqueada por padrão (use --promote "
            "somente com autorização de gate downstream)"
        )

    # Encerramento limpo: reduz a chance do erro de finalização do
    # GeneratorDataset ("Python interpreter state is not initialized").
    del train_ds, val_ds, model
    gc.collect()
    tf.keras.backend.clear_session()

    # Política SDD: execução bem-sucedida retorna 0 independentemente do QG4
    # (QG4 fail é resultado científico, registrado em run_status.json/qg4_result.json;
    # enforcement acontece no wrapper --enforce-qg4 / make pretrain-qg).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
