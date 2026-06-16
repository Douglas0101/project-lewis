"""AAMI EC57 annotation mapping — WFDB symbols → AAMI classes.

Regras mandatórias (ecg-preprocessing-pipeline + Camada-02/03 spec):
- Apenas beat annotations (códigos 0-29 no formato MIT)
- Mapeamento canônico: N, S, V, F, Q
- Stats: n_total, n_unmapped, n_by_class, n_by_symbol
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

LOGGER = logging.getLogger("lewis.camada03.aami_mapper")

# AAMI EC57 mapping: WFDB symbol → AAMI class
AAMI_MAP: Dict[str, str] = {
    # Normal
    "N": "N",
    "L": "N",  # Left bundle branch block
    "R": "N",  # Right bundle branch block
    "e": "N",  # Atrial escape beat
    "j": "N",  # Nodal (junctional) escape beat
    # Supraventricular ectopic
    "A": "S",  # Atrial premature contraction
    "a": "S",  # Aberrated atrial premature beat
    "J": "S",  # Nodal (junctional) premature beat
    "S": "S",  # Premature/ectopic supraventricular beat
    # Ventricular ectopic
    "V": "V",  # Premature ventricular contraction
    "E": "V",  # Ventricular escape beat
    # Fusion
    "F": "F",  # Fusion of ventricular and normal beat
    # Unknown / unclassifiable / paced
    "/": "Q",  # Paced beat
    "f": "Q",  # Fusion of paced and normal beat
    "Q": "Q",  # Unclassifiable beat
    "|": "Q",  # Isolated QRS-like artifact
}

AAMI_CLASSES: List[str] = ["N", "S", "V", "F", "Q"]

AAMI_DESCRIPTION: Dict[str, str] = {
    "N": "Normal / Bundle branch block / Escape",
    "S": "Supraventricular ectopic",
    "V": "Ventricular ectopic",
    "F": "Fusion beat",
    "Q": "Paced / Unclassifiable / Artifact",
}

# Symbols explicitly excluded (non-beat annotations)
_EXCLUDED_SYMBOLS: set[str] = {"~", "+", "x"}


def map_annotations(
    symbols: List[str],
) -> Tuple[List[str], Dict[str, int]]:
    """Map WFDB beat symbols to AAMI EC57 classes.

    Parameters
    ----------
    symbols : List[str]
        WFDB annotation symbols.

    Returns
    -------
    labels_aami : List[str]
        Mapped AAMI labels (only for known beat symbols).
    stats : Dict[str, int]
        {
            "n_total": int,
            "n_mapped": int,
            "n_unmapped": int,
            "n_by_class": Dict[str, int],
            "n_by_symbol": Dict[str, int],
        }
    """
    labels_aami: List[str] = []
    n_unmapped = 0
    n_by_class: Dict[str, int] = {c: 0 for c in AAMI_CLASSES}
    n_by_symbol: Dict[str, int] = {}

    for sym in symbols:
        # Skip non-beat annotations
        if sym in _EXCLUDED_SYMBOLS:
            continue

        if sym in AAMI_MAP:
            aami = AAMI_MAP[sym]
            labels_aami.append(aami)
            n_by_class[aami] = n_by_class.get(aami, 0) + 1
        else:
            n_unmapped += 1
            LOGGER.debug("Símbolo WFDB não mapeado: '%s' → classificado como 'Q'", sym)
            # Unknown symbols map to Q (unclassifiable)
            labels_aami.append("Q")
            n_by_class["Q"] = n_by_class.get("Q", 0) + 1

        n_by_symbol[sym] = n_by_symbol.get(sym, 0) + 1

    stats: Dict[str, Any] = {
        "n_total": len(labels_aami),
        "n_mapped": len(labels_aami) - n_unmapped,
        "n_unmapped": n_unmapped,
        "n_by_class": n_by_class,
        "n_by_symbol": n_by_symbol,
    }

    LOGGER.info(
        "AAMI mapping: %d total | N=%d S=%d V=%d F=%d Q=%d | %d unmapped",
        stats["n_total"],
        n_by_class.get("N", 0),
        n_by_class.get("S", 0),
        n_by_class.get("V", 0),
        n_by_class.get("F", 0),
        n_by_class.get("Q", 0),
        n_unmapped,
    )
    return labels_aami, stats


def map_annotations_array(
    symbols: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """NumPy array version of map_annotations.

    Returns
    -------
    labels_aami : np.ndarray
        Array of AAMI labels (dtype=str).
    stats : Dict[str, int]
        Same as map_annotations.
    """
    labels, stats = map_annotations([str(s) for s in symbols])
    return np.array(labels), stats
