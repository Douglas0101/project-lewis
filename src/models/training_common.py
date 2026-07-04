"""Helpers compartilhados entre os scripts de treinamento do Project-Lewis.

Estas funções são usadas por ``scripts/run_stage1_training.py``,
``scripts/run_stage2_training.py`` e ``scripts/run_finetune_groupkfold.py`` para
eliminar duplicação de código em tarefas comuns: carregamento de config, dados,
grupos, parser CLI, resolução de loss e lineage.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from src.models.finetune_mitbih import SparseCategoricalFocalLoss

LOGGER = logging.getLogger("lewis.camada04.training_common")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_and_validate(path: Path) -> Path:
    """Resolve ``path`` e garante que ele permaneça dentro de ``PROJECT_ROOT``."""
    if not path.is_absolute():
        path = Path.cwd() / path
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes project root: {path}") from exc
    return resolved


def load_config(config_path: Path) -> dict:
    """Carrega um arquivo YAML de configuração."""
    config_path = _resolve_and_validate(config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_features(
    feature_npz: Path, feature_parquet: Path
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Carrega batimentos segmentados e metadados associados."""
    LOGGER.info("Loading features from %s", feature_npz)
    data = np.load(feature_npz)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    if X.ndim == 2:
        X = X[..., np.newaxis]

    LOGGER.info("Loading metadata from %s", feature_parquet)
    df = pd.read_parquet(feature_parquet)

    if len(X) != len(df) or len(y) != len(df):
        raise ValueError(f"Mismatch: X={len(X)}, y={len(y)}, df={len(df)}")

    LOGGER.info("Loaded %d beats | X shape=%s | classes=%d", len(X), X.shape, len(np.unique(y)))
    return X, y, df


def build_groups(df: pd.DataFrame) -> np.ndarray:
    """Monta array de grupos a partir de ``record_id`` para GroupKFold."""
    unique_records = df["record_id"].unique()
    record_to_group = {rec: idx for idx, rec in enumerate(unique_records)}
    groups = df["record_id"].map(record_to_group).to_numpy(dtype=np.int64)
    LOGGER.info("Built groups | n_patients=%d", len(unique_records))
    return groups


def build_base_arg_parser(
    description: str,
    default_config: Path,
    *,
    include_pretrained: bool = True,
    include_batch_size_lr: bool = False,
) -> argparse.ArgumentParser:
    """Parser CLI com argumentos comuns aos scripts de treinamento."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Caminho para {default_config.name}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Diretório para salvar modelo final",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Sobrescrever número de épocas",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        help="Sobrescrever número de folds",
    )
    if include_pretrained:
        parser.add_argument(
            "--pretrained",
            type=Path,
            default=PROJECT_ROOT / "models" / "finetuned_float32_v1.1.keras",
            help="Modelo pré-treinado para inicializar o backbone",
        )
        parser.add_argument(
            "--freeze-backbone",
            action="store_true",
            help="Congelar camadas convolucionais do backbone pré-treinado",
        )
    if include_batch_size_lr:
        parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help="Sobrescrever batch size do config",
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            default=None,
            help="Sobrescrever learning rate do config",
        )
    return parser


def resolve_loss(train_cfg: dict) -> Any:
    """Resolve configuração de loss para crossentropy ou focal loss."""
    loss_cfg = train_cfg.get("loss", "sparse_categorical_crossentropy")
    if loss_cfg == "sparse_categorical_crossentropy":
        return "sparse_categorical_crossentropy"
    if loss_cfg == "focal_loss":
        gamma = float(train_cfg.get("focal_gamma", 2.0))
        alpha = train_cfg.get("focal_alpha")
        if alpha is not None:
            alpha = np.array(alpha, dtype=np.float32)
        loss = SparseCategoricalFocalLoss(gamma=gamma, alpha=alpha)
        LOGGER.info("Using focal loss | gamma=%.2f | alpha=%s", gamma, alpha)
        return loss
    raise ValueError(f"Unsupported loss: {loss_cfg}")


def copy_best_fold_artifacts(
    summary: Dict[str, Any],
    experiment_dir: Path,
    output_dir: Path,
    artifact_map: Dict[str, Path],
) -> None:
    """Copia artefatos do melhor fold para o diretório de saída.

    ``artifact_map`` mapeia caminho relativo do artefato dentro do fold
    para o caminho absoluto de destino.
    """
    output_dir = _resolve_and_validate(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_fold_dir = experiment_dir / f"fold_{summary['best_fold']}"

    for src_rel, dst in artifact_map.items():
        src = best_fold_dir / src_rel
        if src.exists():
            shutil.copy(str(src), str(dst))
            LOGGER.info("Copied %s to %s", src_rel, dst)
        elif src_rel in ("best_weights.weights.threshold.json",):
            LOGGER.warning("Optional artifact not found at %s", src)
        else:
            LOGGER.error("Artifact not found at %s", src)


def write_lineage(lineage: Dict[str, Any], model_filename: str) -> Path:
    """Persiste lineage do modelo em ``data/lineage/models``."""
    lineage_dir = PROJECT_ROOT / "data" / "lineage" / "models"
    lineage_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_filename).stem
    lineage_path = lineage_dir / f"{model_name}.json"
    lineage["timestamp"] = datetime.now(timezone.utc).isoformat()
    with lineage_path.open("w", encoding="utf-8") as fh:
        json.dump(lineage, fh, indent=2, ensure_ascii=False)
    LOGGER.info("Lineage saved to %s", lineage_path)
    return lineage_path
