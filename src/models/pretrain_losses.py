"""Loss variants for imbalance-aware pre-training (FASE 6, A2).

``pos_weight`` is computed exclusively from the TRAIN split (never from
validation/test), per mission rule 5.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence, Set

import numpy as np
import tensorflow as tf

LOGGER = logging.getLogger("lewis.camada04.losses")

NUM_CLASSES = 5
POS_WEIGHT_CLIP = 10.0


def estimate_pos_weights(
    records: Set[str],
    diagnosis: Mapping[str, Sequence[int]],
    clip_max: float = POS_WEIGHT_CLIP,
) -> np.ndarray:
    """neg/pos ratio per class over ``records`` (clipped to [1, clip_max]).

    Parameters
    ----------
    records:
        Train-split record names (ONLY train — rule 5).
    diagnosis:
        Mapping record name → multi-hot vector.
    """
    sums = np.zeros(NUM_CLASSES, dtype=np.float64)
    n = 0
    for name in records:
        vec = diagnosis.get(name)
        if vec is None:
            continue
        sums += np.asarray(vec, dtype=np.float64)
        n += 1
    if n == 0:
        LOGGER.warning("pos_weight: no labeled records; defaulting to 1.0")
        return np.ones(NUM_CLASSES, dtype=np.float32)
    pos = np.maximum(sums, 1.0)
    neg = np.maximum(n - sums, 1.0)
    weights = np.clip(neg / pos, 1.0, clip_max).astype(np.float32)
    LOGGER.info("pos_weight (train split, n=%d): %s", n, weights.tolist())
    return weights


def weighted_bce(pos_weight: np.ndarray):
    """Binary cross-entropy with per-class positive weighting."""
    pw = tf.constant(np.asarray(pos_weight, dtype=np.float32))

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(pw * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(tf.reduce_sum(bce, axis=-1))

    return loss


def build_loss(name: str, pos_weight: Optional[np.ndarray] = None):
    """Return the loss for ``bce`` | ``bce_weighted`` | ``focal``."""
    if name == "bce":
        return "binary_crossentropy"
    if name == "bce_weighted":
        if pos_weight is None:
            raise ValueError("bce_weighted requires pos_weight")
        return weighted_bce(pos_weight)
    if name == "focal":
        return tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)
    raise ValueError(f"unknown loss '{name}'; options: bce, bce_weighted, focal")
