"""QG4 gate semantics (FASE 9) — pinned thresholds, strict operators.

The gate is defined in ``config/pretrain_v1.0.yaml`` and evaluated on the
best epoch (min val_loss). These tests pin the thresholds so they cannot be
loosened accidentally.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.models.pretrain_chapman import qg4_passes

CONFIG = Path("config/pretrain_v1.0.yaml")


def test_qg4_thresholds_are_not_loosened():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    qg4 = cfg["quality_gate"]["qg4"]
    assert qg4["min_val_auc_roc_macro"] == 0.85
    assert qg4["max_val_loss"] == 0.15


@pytest.mark.parametrize(
    "auc,loss,expected",
    [
        (0.86, 0.14, True),   # passa: ambos satisfeitos
        (0.85, 0.14, False),  # AUC no limiar: operador estrito >
        (0.86, 0.15, False),  # loss no limiar: operador estrito <
        (0.8333, 0.3907, False),  # run histórico A0
        (0.90, 0.50, False),  # AUC ok, loss ruim
        (0.80, 0.10, False),  # loss ok, AUC ruim
    ],
)
def test_qg4_passes_strict_operators(auc, loss, expected):
    best = {"best_epoch": 1, "val_auc_roc": auc, "val_loss": loss}
    qg4_cfg = {"min_val_auc_roc_macro": 0.85, "max_val_loss": 0.15}
    assert qg4_passes(best, qg4_cfg) is expected
