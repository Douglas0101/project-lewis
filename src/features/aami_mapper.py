"""AAMI EC57 annotation mapping — WFDB symbols → AAMI classes (vista legada v2.x).

.. deprecated:: 3.0.0
    A fonte única de mapeamento é ``src/features/ontology_v3.py`` (ontologia
    clínica versionada v3.0.0, ver ``docs/rebuild_spec/01_clinical_ontology_decision.md``).
    Este módulo permanece como camada de compatibilidade: delega à ontologia v3 e
    devolve labels na visão legada (FUSION→F, Q_OR_UNKNOWN→Q).

    Mudanças semânticas v3:
    - símbolos desconhecidos são **excluídos**, nunca mapeados para Q;
    - ``|`` (QRS-like artifact) é excluído;
    - ``~``, ``+``, ``x`` permanecem excluídos.

Regras mandatórias (ecg-preprocessing-pipeline + Camada-02/03 spec):
- Apenas beat annotations (códigos 0-29 no formato MIT)
- Mapeamento canônico: N, S, V, F, Q (visão legada)
- Stats: n_total, n_unmapped, n_by_class, n_by_symbol
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from src.features.ontology_v3 import (
    BEAT_MAP_V3,
    CANONICAL_TO_LEGACY,
    EXCLUDED_SYMBOLS_V3,
    map_symbols_v3_legacy,
)

LOGGER = logging.getLogger("lewis.camada03.aami_mapper")

# AAMI EC57 mapping: WFDB symbol → AAMI class (visão legada derivada da ontologia v3)
AAMI_MAP: Dict[str, str] = {
    sym: CANONICAL_TO_LEGACY[canonical] for sym, (canonical, _, _) in BEAT_MAP_V3.items()
}

AAMI_CLASSES: List[str] = ["N", "S", "V", "F", "Q"]

AAMI_DESCRIPTION: Dict[str, str] = {
    "N": "Normal / Bundle branch block / Escape",
    "S": "Supraventricular ectopic",
    "V": "Ventricular ectopic",
    "F": "Fusion beat (somente fusão V+N; nunca fibrilação atrial)",
    "Q": "Paced / Unclassifiable (classe de rejeição — fora dos alvos clínicos)",
}

# Symbols explicitly excluded (non-beat annotations + QRS-like artifact)
_EXCLUDED_SYMBOLS: set[str] = set(EXCLUDED_SYMBOLS_V3)


def map_annotations(
    symbols: List[str],
) -> Tuple[List[str], Dict[str, Any]]:
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
    labels_aami, _keep_mask, v3_stats = map_symbols_v3_legacy(symbols)
    n_excluded = int(v3_stats["n_excluded"])
    n_by_class: Dict[str, int] = v3_stats["n_by_class_legacy"]
    n_by_symbol: Dict[str, int] = v3_stats["n_by_symbol"]

    stats: Dict[str, Any] = {
        "n_total": len(labels_aami),
        "n_mapped": len(labels_aami),
        "n_unmapped": 0,
        "n_excluded": n_excluded,
        "n_by_class": n_by_class,
        "n_by_symbol": n_by_symbol,
    }

    LOGGER.info(
        "AAMI mapping (v3): %d total | N=%d S=%d V=%d F=%d Q=%d | %d excluídos",
        stats["n_total"],
        n_by_class.get("N", 0),
        n_by_class.get("S", 0),
        n_by_class.get("V", 0),
        n_by_class.get("F", 0),
        n_by_class.get("Q", 0),
        n_excluded,
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
