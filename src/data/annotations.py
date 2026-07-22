"""Annotation loader for WFDB beat annotations (.atr)."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import wfdb

from src.features.ontology_v3 import BEAT_MAP_V3, CANONICAL_TO_LEGACY

logger = logging.getLogger(__name__)

# AAMI EC57 mapping for the five heartbeat classes (visão legada).
# Fonte única: src/features/ontology_v3.py (v3.0.0). Símbolos não listados
# (incluindo '|', '~', '+', 'x' e desconhecidos) são excluídos — nunca → Q.
AAMI_BEAT_MAP = {
    sym: CANONICAL_TO_LEGACY[canonical] for sym, (canonical, _, _) in BEAT_MAP_V3.items()
}

# Symbols that are explicitly beat annotations. Anything else (e.g. rhythm
# changes, noise, etc.) is dropped so we only return actual beats.
_BEAT_SYMBOLS = set(AAMI_BEAT_MAP.keys())


def load_annotations(record_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load WFDB beat annotations for a record.

    Parameters
    ----------
    record_path : str or Path
        Full path to the record *without* extension. The ``.atr`` file must
        exist alongside it.

    Returns
    -------
    samples : np.ndarray
        1-D array of sample indices where a beat occurs.
    labels : np.ndarray
        1-D array of AAMI class labels (str) for each beat.

    Raises
    ------
    FileNotFoundError
        If the ``.atr`` annotation file does not exist.
    """
    record_path = Path(record_path)
    atr_path = record_path.with_suffix(".atr")

    if not atr_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {atr_path}")

    logger.debug("Reading annotations for %s", record_path)
    ann = wfdb.rdann(str(record_path), extension="atr")

    # ann.symbol holds the annotation symbol for each entry.
    # ann.sample holds the corresponding sample index.
    symbols = np.array(ann.symbol)
    samples = np.array(ann.sample)

    # Keep only known beat symbols.
    mask = np.isin(symbols, list(_BEAT_SYMBOLS))
    n_dropped = (~mask).sum()
    if n_dropped:
        logger.debug("Dropped %d non-beat annotations from %s", n_dropped, record_path.name)

    samples = samples[mask]
    symbols = symbols[mask]

    # Map to AAMI classes.
    labels = np.array([AAMI_BEAT_MAP[s] for s in symbols])

    logger.info("Loaded %d beat annotations from %s", len(samples), record_path.name)
    return samples, labels
