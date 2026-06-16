"""Annotation loader for WFDB beat annotations (.atr)."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import wfdb

logger = logging.getLogger(__name__)

# AAMI EC57 mapping for the five heartbeat classes.
# Symbols not listed here are mapped to 'Q' (unknown / unclassifiable)
# or ignored if they are non-beat markers.
AAMI_BEAT_MAP = {
    # Normal
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    # Supraventricular ectopic
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    # Ventricular ectopic
    "V": "V",
    "E": "V",
    # Fusion
    "F": "F",
    # Unknown / unclassifiable (paced, etc.)
    "/": "Q",
    "f": "Q",
    "Q": "Q",
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
