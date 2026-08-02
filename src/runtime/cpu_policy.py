"""Política de runtime CPU-only (PRD+SDD CPU-First, RF-CPU-001..003).

Garante, para qualquer processo do pipeline:

- ``CUDA_VISIBLE_DEVICES=-1`` antes da importação do TensorFlow (RF-CPU-001);
- perfil numérico único por execução (RF-CPU-003): ``strict`` (avaliação oficial,
  oneDNN off, ops determinísticas) ou ``fast`` (exploração/benchmark, oneDNN on);
- falha se alguma GPU ficar visível numa execução CPU-only (RF-CPU-002).

Perfis em ``configs/runtime/<profile>.yaml``. ``apply()`` deve ser chamado ANTES
de importar TensorFlow; ``verify()`` DEPOIS do import.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger("lewis.runtime.cpu_policy")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILES_DIR = PROJECT_ROOT / "configs" / "runtime"
PROFILES = ("strict", "fast")

_TF_IMPORTED_ERROR = (
    "cpu_policy.apply() chamado após TensorFlow já importado — a política "
    "deve ser aplicada antes de qualquer `import tensorflow`"
)


def load_profile(profile: str) -> dict[str, Any]:
    """Carrega o perfil de runtime (configs/runtime/<profile>.yaml)."""
    if profile not in PROFILES:
        raise ValueError(f"perfil desconhecido '{profile}'; opções: {PROFILES}")
    path = PROFILES_DIR / f"{profile}.yaml"
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def apply(profile: str, *, _allow_imported_tf: bool = False) -> dict[str, Any]:
    """Aplica o perfil de runtime via variáveis de ambiente (pré-import TF).

    Deve ser chamado antes de qualquer ``import tensorflow``. Retorna o perfil
    resolvido (para logging/proveniência).
    """
    if not _allow_imported_tf and "tensorflow" in sys.modules:
        raise RuntimeError(_TF_IMPORTED_ERROR)
    cfg = load_profile(profile)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1" if cfg["onednn"] else "0"
    if cfg.get("deterministic_ops"):
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    else:
        os.environ.pop("TF_DETERMINISTIC_OPS", None)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(cfg["threading"]["intra_op"])
    os.environ["TF_NUM_INTEROP_THREADS"] = str(cfg["threading"]["inter_op"])
    LOGGER.info(
        "runtime profile=%s | CUDA_VISIBLE_DEVICES=-1 | oneDNN=%s | deterministic_ops=%s",
        profile,
        cfg["onednn"],
        bool(cfg.get("deterministic_ops")),
    )
    return cfg


def verify(profile: str) -> dict[str, Any]:
    """Verifica dispositivos pós-import TF; falha se GPU estiver visível."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    report = {
        "profile": profile,
        "physical_cpus": len(cpus),
        "physical_gpus": [str(g) for g in gpus],
        "cuda_disabled": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
    }
    if gpus:
        raise RuntimeError(
            f"execução CPU-only ({profile}) com GPU visível: {gpus} — abortando"
        )
    if not report["cuda_disabled"]:
        LOGGER.warning("CUDA_VISIBLE_DEVICES != -1 (perfil %s)", profile)
    LOGGER.info("dispositivos verificados: %s", report)
    return report


def environment_report(profile: str) -> dict[str, Any]:
    """Relatório de ambiente para proveniência (RF-PROV-003)."""
    import tensorflow as tf

    return {
        "hardware": {
            "cpu_model": platform.processor() or platform.machine(),
            "physical_cores": os.cpu_count(),
            "ram_gib_total": None,  # preenchido pelo chamador quando disponível
        },
        "software": {
            "os": platform.platform(),
            "kernel": platform.release(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": tf.keras.__version__ if hasattr(tf.keras, "__version__") else "bundled",
        },
        "runtime": {
            "profile": profile,
            "onednn": os.environ.get("TF_ENABLE_ONEDNN_OPTS") == "1",
            "deterministic_ops": os.environ.get("TF_DETERMINISTIC_OPS") == "1",
            "intra_threads": os.environ.get("TF_NUM_INTRAOP_THREADS"),
            "inter_threads": os.environ.get("TF_NUM_INTEROP_THREADS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
