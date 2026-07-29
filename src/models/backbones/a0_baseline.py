"""A0 baseline — congelada (FASE 5/6).

Wrapper fino sobre ``build_backbone_1d_multilabel``: a definição arquitetural
é pinada por ``tests/test_backbone_budget.py`` (params == 19.933). Qualquer
mudança arquitetural acidental na A0 quebra o teste de orçamento.
"""

from __future__ import annotations

import tensorflow as tf

from src.models.backbone_1d import build_backbone_1d_multilabel

from .spec import BackboneSpec

A0_BASELINE_PARAMS = 19_933


def build_a0_baseline(spec: BackboneSpec) -> tf.keras.Model:
    """Build the frozen A0 baseline (sigmoid multi-label head)."""
    return build_backbone_1d_multilabel(
        input_len=spec.input_len,
        num_classes=spec.num_classes,
        dropout_rate=spec.dropout_rate,
        name="a0_baseline",
    )
