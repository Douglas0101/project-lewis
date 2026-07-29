"""Backbone factory — pretrain architecture variants (FASE 6).

Variants:
- ``a0``: frozen baseline (19.933 params, pinned).
- ``a1``: A1_stable residual (no BatchNorm — project TFLM constraint).
- ``a2``: A1 architecture trained with an imbalance-aware loss (pos_weight /
  focal). A2 is a *training* variant, not a new architecture, so it maps to
  the same builder as ``a1``; the loss is selected in the training layer.
"""

from __future__ import annotations

from typing import Callable, Dict

import tensorflow as tf

from .a0_baseline import A0_BASELINE_PARAMS, build_a0_baseline
from .a1_stable import build_a1_stable
from .spec import BackboneSpec

MAX_PARAMS = 2 * A0_BASELINE_PARAMS
MAX_FLATBUFFER_KB = 64

BUILDERS: Dict[str, Callable[[BackboneSpec], tf.keras.Model]] = {
    "a0": build_a0_baseline,
    "a1": build_a1_stable,
    "a2": build_a1_stable,
}


def build_backbone(spec: BackboneSpec) -> tf.keras.Model:
    """Build a backbone variant by name (``a0`` | ``a1`` | ``a2``)."""
    try:
        builder = BUILDERS[spec.arch]
    except KeyError:
        raise ValueError(
            f"unknown architecture '{spec.arch}'; options: {sorted(BUILDERS)}"
        ) from None
    return builder(spec)


__all__ = [
    "A0_BASELINE_PARAMS",
    "BUILDERS",
    "BackboneSpec",
    "MAX_FLATBUFFER_KB",
    "MAX_PARAMS",
    "build_backbone",
]
