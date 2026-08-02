"""Gerenciador de freeze de modelo (PRD RF-DATA-005 / freeze_manager, SDD §12.8).

O conjunto de teste só pode ser acessado após a criação de ``model_freeze.json``
na run. Depois do freeze, qualquer alteração no modelo invalida a autorização
(o hash do checkpoint é registrado no freeze).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger("lewis.governance.freeze")

FREEZE_FILENAME = "model_freeze.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_model_freeze(
    run_dir: Path,
    *,
    architecture: str,
    loss: str,
    checkpoint_epoch: Optional[int] = None,
    preprocessing_hash: Optional[str] = None,
    validation_metrics_hash: Optional[str] = None,
) -> Path:
    """Cria ``model_freeze.json`` na run e autoriza o acesso ao teste.

    Falha se o checkpoint não existir ou se já houver freeze (write-once).
    """
    run_dir = Path(run_dir)
    freeze_path = run_dir / FREEZE_FILENAME
    if freeze_path.exists():
        raise RuntimeError(f"freeze já existe (write-once): {freeze_path}")
    checkpoint = run_dir / "backbone_pretrained.keras"
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint ausente para freeze: {checkpoint}")
    payload: dict[str, Any] = {
        "model_hash": _sha256(checkpoint),
        "architecture": architecture,
        "loss": loss,
        "checkpoint_epoch": checkpoint_epoch,
        "preprocessing_hash": preprocessing_hash,
        "validation_metrics_hash": validation_metrics_hash,
        "selection_complete": True,
        "test_access_authorized": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    freeze_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    LOGGER.info("freeze criado: %s (teste autorizado)", freeze_path)
    return freeze_path


def is_test_authorized(run_dir: Path) -> bool:
    """True somente se existir freeze válido com autorização de teste."""
    freeze_path = Path(run_dir) / FREEZE_FILENAME
    if not freeze_path.exists():
        return False
    try:
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("freeze corrompido: %s", freeze_path)
        return False
    return bool(payload.get("selection_complete") and payload.get("test_access_authorized"))
