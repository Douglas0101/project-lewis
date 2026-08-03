"""Provenance and reproducibility helpers for Chapman pre-training (FASE 4).

Every run writes ``provenance.json`` (git, environment, dataset, model,
training, metrics, QG4, artifact SHA-256), ``history.json`` (full Keras
history) and ``metrics_per_class.json`` (per-superclass evaluation on the
validation split — evaluation only, never used for training decisions).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess  # nosec B404
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

LOGGER = logging.getLogger("lewis.camada04.provenance")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file (empty string if missing)."""
    path = Path(path)
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git_info() -> dict:
    """Current git commit/branch (best effort — never raises)."""

    def _git(*args: str) -> str:
        try:
            out = subprocess.run(  # nosec B603 B607
                ["git", *args],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return out.stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    return {"git_commit": _git("rev-parse", "HEAD"), "git_branch": _git("branch", "--show-current")}


def set_global_seeds(seed: int) -> None:
    """Fix every seed relevant to the pretrain pipeline."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def apply_deterministic_mode(mode: str) -> None:
    """Apply ``strict`` or ``fast`` determinism settings.

    strict: ``tf.config.experimental.enable_op_determinism()`` — the wrapper
    also exports ``TF_ENABLE_ONEDNN_OPTS=0``/``TF_DETERMINISTIC_OPS=1`` before
    launching the subprocess (they must be set before TF loads).
    fast: oneDNN/perf defaults; seeds still fixed.
    """
    if mode == "strict":
        tf.config.experimental.enable_op_determinism()
        LOGGER.info("deterministic mode: strict (op determinism enabled)")
    else:
        LOGGER.info("deterministic mode: fast (oneDNN/perf defaults, seeded)")


def onednn_enabled() -> bool:
    return os.environ.get("TF_ENABLE_ONEDNN_OPTS", "1") != "0"


def runtime_env_snapshot(profile: Optional[str] = None) -> dict:
    """Fotografia do ambiente de runtime EFETIVO para a proveniência (RF-PROV-003).

    Registra o perfil solicitado (quando propagado pelo wrapper via
    ``LEWIS_RUNTIME_PROFILE``) e as variáveis que realmente governam a
    numericidade do processo — sem depender de rótulos de config.
    """
    return {
        "profile": profile,
        "onednn": os.environ.get("TF_ENABLE_ONEDNN_OPTS", "1") != "0",
        "deterministic_ops": os.environ.get("TF_DETERMINISTIC_OPS") == "1",
        "intra_threads": os.environ.get("TF_NUM_INTRAOP_THREADS"),
        "inter_threads": os.environ.get("TF_NUM_INTEROP_THREADS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def compute_per_class_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Per-class ROC-AUC, PR-AUC, P/R/F1@0.5 and support (multi-label)."""
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    from src.data.chapman_labels import SCP_SUPERCLASSES

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class: dict[str, dict] = {}
    for idx, cls in enumerate(SCP_SUPERCLASSES[: y_true.shape[1]]):
        yt, ys = y_true[:, idx], y_score[:, idx]
        n_pos = int(yt.sum())
        per_class[cls] = {
            "support": n_pos,
            "auc_roc": float(roc_auc_score(yt, ys)) if 0 < n_pos < len(yt) else None,
            "auc_pr": float(average_precision_score(yt, ys)) if n_pos > 0 else None,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
        }
    return {"threshold": 0.5, "per_class": per_class}


def build_provenance(
    *,
    run_id: str,
    seed: int,
    deterministic_mode: str,
    train_records: int,
    val_records: int,
    model_info: dict,
    training_info: dict,
    metrics: dict,
    qg4: dict,
    artifacts: dict,
    hashes: dict,
    split_policy: str = "record_disjoint (val_ratio=0.1, seeded shuffle)",
    runtime: Optional[dict] = None,
) -> dict:
    """Assemble the provenance document for a run.

    ``runtime`` deve ser a fotografia do ambiente efetivo
    (``runtime_env_snapshot``); quando presente, substitui o rótulo legado
    como fonte de verdade sobre o perfil numérico da execução.
    """
    return {
        "run_id": run_id,
        "stage": "pretrain_chapman",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **git_info(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "gpu_available": bool(tf.config.list_physical_devices("GPU")),
        "onednn_enabled": onednn_enabled(),
        "deterministic_mode": deterministic_mode,
        "runtime": runtime,
        "seed": seed,
        "dataset": {
            "name": "Chapman",
            "train_records": train_records,
            "val_records": val_records,
            "split_policy": split_policy,
            "patient_disjoint": None,
            "reference": "Zheng et al., Scientific Data, 2020",
        },
        "model": model_info,
        "training": training_info,
        "metrics": metrics,
        "qg4": qg4,
        "artifacts": artifacts,
        "hashes": hashes,
        "paper_alignment": {
            "transfer_learning": True,
            "pr_auc_for_imbalance": True,
            "calibration": False,
            "patient_disjoint": None,
            "tinyml_constraint": True,
        },
    }


def write_json(path: Path, data: dict) -> str:
    """Write JSON and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    digest = sha256_file(path)
    LOGGER.info("Artefato salvo: %s (sha256=%s…)", path.name, digest[:12])
    return digest


def save_history(experiment_dir: Path, history: dict) -> str:
    """Persist the full Keras history; returns the file SHA-256."""
    serializable = {k: [float(v) for v in values] for k, values in history.items()}
    return write_json(Path(experiment_dir) / "history.json", {"history": serializable})


def evaluate_per_class(
    model: tf.keras.Model,
    val_dataset: tf.data.Dataset,
    validation_steps: int,
) -> dict:
    """Predict on the validation split and compute per-class metrics."""
    y_score = model.predict(val_dataset, steps=validation_steps, verbose=0)
    y_true_batches = [y.numpy() for _, y in val_dataset.take(validation_steps)]
    y_true = np.concatenate(y_true_batches, axis=0)
    return compute_per_class_metrics(y_true[: len(y_score)], y_score)


def write_gate_and_status(
    *,
    experiment_dir: Path,
    best: dict,
    qg4_cfg: dict,
    qg4_pass: bool,
    model_promoted: bool,
    known_issues: Optional[list] = None,
    gate_loss_metric: str = "val_loss",
    checkpoint_monitor: str = "val_loss",
) -> dict:
    """Write ``qg4_result.json`` and ``run_status.json`` (contracts 10.4/10.6).

    Separates execution_success from qg4_pass (DEF-010): QG4 fail is a
    scientific result, not a process failure.

    ``gate_loss_metric`` é o nome REAL da métrica de perda julgada
    (``val_bce_monitor`` em runs com loss não-BCE) — os braços do gate são
    rotulados por ela, nunca por um literal. ``best`` deve descrever a época
    do checkpoint salvo (monitor ``checkpoint_monitor`` do
    EarlyStopping/ModelCheckpoint).
    """
    experiment_dir = Path(experiment_dir)
    min_auc = qg4_cfg["min_val_auc_roc_macro"]
    max_loss = qg4_cfg["max_val_loss"]
    arms = {
        "val_auc_roc": {
            "threshold": min_auc,
            "observed": best["val_auc_roc"],
            "gap": best["val_auc_roc"] - min_auc,
            "pass": bool(best["val_auc_roc"] > min_auc),
        },
        gate_loss_metric: {
            "threshold": max_loss,
            "observed": best["val_loss"],
            "gap": best["val_loss"] - max_loss,
            "pass": bool(best["val_loss"] < max_loss),
        },
    }
    failing = [name for name, arm in arms.items() if not arm["pass"]]
    dominant = max(failing, key=lambda n: abs(arms[n]["gap"]), default=None)

    qg4_result = {
        "gate": "QG4",
        "pass": bool(qg4_pass),
        "metric": dominant or "all",
        "threshold": arms[dominant]["threshold"] if dominant else None,
        "observed": arms[dominant]["observed"] if dominant else None,
        "gap": arms[dominant]["gap"] if dominant else None,
        "decision_rule": (
            f"val_auc_roc > min AND {gate_loss_metric} < max at checkpoint epoch "
            f"(best {checkpoint_monitor})"
        ),
        "blocking": True,
        "arms": arms,
    }
    write_json(experiment_dir / "qg4_result.json", qg4_result)

    run_status = {
        "execution_success": True,
        "qg4": {
            "pass": bool(qg4_pass),
            "best_epoch": best["best_epoch"],
            "val_loss": best["val_loss"],
            "val_auc_roc": best["val_auc_roc"],
            "gate_loss_metric": gate_loss_metric,
            "checkpoint_monitor": checkpoint_monitor,
            "reason": (
                "all arms satisfied"
                if qg4_pass
                else f"failing arms: {', '.join(failing)} (dominant: {dominant})"
            ),
        },
        "model_promoted": bool(model_promoted),
        "artifacts_ok": True,
        "known_issues": known_issues or [],
    }
    write_json(experiment_dir / "run_status.json", run_status)
    return run_status


def write_provenance_and_metrics(
    *,
    experiment_dir: Path,
    model: tf.keras.Model,
    history: dict,
    val_dataset: tf.data.Dataset,
    validation_steps: int,
    seed: int,
    deterministic_mode: str,
    train_records: int,
    val_records: int,
    training_info: dict,
    best: dict,
    qg4_pass: bool,
    extra_artifacts: Optional[dict] = None,
    split_policy: Optional[str] = None,
    split_manifest_sha256: Optional[str] = None,
    runtime_profile: Optional[str] = None,
    gate_loss_metric: str = "val_loss",
    checkpoint_monitor: str = "val_loss",
) -> dict:
    """Write history.json, metrics_per_class.json and provenance.json.

    ``best`` deve descrever a época do checkpoint salvo (mesma usada pelo
    QG4), mantendo provenance.metrics consistente com o artefato implantável.
    Returns the provenance document (already persisted).
    """
    experiment_dir = Path(experiment_dir)
    run_id = experiment_dir.name

    history_sha = save_history(experiment_dir, history)
    per_class = evaluate_per_class(model, val_dataset, validation_steps)
    per_class_sha = write_json(experiment_dir / "metrics_per_class.json", per_class)

    artifacts = {
        "model": "backbone_pretrained.keras",
        "config": "config.json",
        "history": "history.json",
        **(extra_artifacts or {}),
    }
    hashes = {
        "model_sha256": sha256_file(experiment_dir / "backbone_pretrained.keras"),
        "config_sha256": sha256_file(experiment_dir / "config.json"),
        "history_sha256": history_sha,
        "metrics_per_class_sha256": per_class_sha,
    }
    if split_manifest_sha256:
        hashes["split_manifest_sha256"] = split_manifest_sha256
    val_auc_pr_series = history.get("val_auc_pr") or [float("nan")] * best["best_epoch"]
    provenance = build_provenance(
        run_id=run_id,
        seed=seed,
        deterministic_mode=deterministic_mode,
        train_records=train_records,
        val_records=val_records,
        model_info={
            "name": model.name,
            "params": int(model.count_params()),
            "estimated_flatbuffer_kb": int(model.count_params() * 1.3 / 1024),
            "input_shape": [None if s is None else int(s) for s in model.input_shape],
            "num_classes": int(model.output_shape[-1]),
        },
        training_info=training_info,
        metrics={
            "best_epoch": best["best_epoch"],
            "val_loss": best["val_loss"],
            "val_auc_roc": best["val_auc_roc"],
            "val_auc_pr": float(val_auc_pr_series[best["best_epoch"] - 1]),
            "gate_loss_metric": gate_loss_metric,
            "checkpoint_monitor": checkpoint_monitor,
        },
        qg4={
            "pass": bool(qg4_pass),
            "reason": (f"val_auc_roc > 0.85 and {gate_loss_metric} < 0.15 at checkpoint epoch"),
        },
        artifacts=artifacts,
        hashes=hashes,
        split_policy=split_policy or "record_disjoint (val_ratio=0.1, seeded shuffle)",
        runtime=runtime_env_snapshot(runtime_profile),
    )
    write_json(experiment_dir / "provenance.json", provenance)
    return provenance
