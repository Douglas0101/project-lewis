"""Ontologia clínica única e versionada — Project-Lewis v3.0.0.

Fonte ÚNICA de mapeamento símbolo WFDB → classe canônica, conforme
``docs/rebuild_spec/01_clinical_ontology_decision.md``.

Mudanças semânticas em relação à v2.x:

- ``F`` passa a se chamar ``FUSION`` (somente fusão V+N; nunca fibrilação atrial);
- ``Q`` passa a se chamar ``Q_OR_UNKNOWN`` (classe de rejeição/abstenção, fora dos
  alvos clínicos — decisão D4);
- símbolos desconhecidos são **excluídos com registro**, nunca mapeados para Q
  (encerra a política de ``aami_mapper.py:96-99`` da v2.x);
- ``|`` (QRS-like artifact) é excluído (documentação C02 sempre disse que deveria ser);
- ``~``, ``+``, ``x`` permanecem excluídos (non-beat).

Qualquer mudança futura exige nova ``ONTOLOGY_VERSION`` e novo hash no bundle.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

ONTOLOGY_VERSION = "3.0.0"

# Níveis semânticos (docs/rebuild_spec/01)
SEMANTIC_LEVELS = ("quality", "beat", "rhythm", "diagnosis")

# ---------------------------------------------------------------------------
# Nível 2 — morfologia do batimento (escopo: batimento)
# ---------------------------------------------------------------------------

BEAT_CLASSES_V3: List[str] = ["N", "S", "V", "FUSION", "Q_OR_UNKNOWN"]

# Símbolo WFDB → (canonicalCode, mappingRule, ambiguity)
BEAT_MAP_V3: Dict[str, Tuple[str, str, str]] = {
    # Normal / BBB / escapes
    "N": ("N", "AAMI-EC57 beat-level v3", "none"),
    "L": ("N", "AAMI-EC57 beat-level v3", "none"),
    "R": ("N", "AAMI-EC57 beat-level v3", "none"),
    "e": ("N", "AAMI-EC57 beat-level v3", "none"),
    "j": ("N", "AAMI-EC57 beat-level v3", "none"),
    # Supraventricular
    "A": ("S", "AAMI-EC57 beat-level v3", "none"),
    "a": ("S", "AAMI-EC57 beat-level v3", "none"),
    "J": ("S", "AAMI-EC57 beat-level v3", "none"),
    "S": ("S", "AAMI-EC57 beat-level v3", "none"),
    # Ventricular
    "V": ("V", "AAMI-EC57 beat-level v3", "none"),
    "E": ("V", "AAMI-EC57 beat-level v3", "none"),
    # Fusão V+N — SOMENTE fusão, nunca fibrilação atrial
    "F": ("FUSION", "AAMI-EC57 beat-level v3", "none"),
    # Paced / fusão paced / não-classificável — classe de rejeição (D4)
    "/": ("Q_OR_UNKNOWN", "AAMI-EC57 beat-level v3 (rejeição)", "none"),
    "f": ("Q_OR_UNKNOWN", "AAMI-EC57 beat-level v3 (rejeição)", "none"),
    "Q": ("Q_OR_UNKNOWN", "AAMI-EC57 beat-level v3 (rejeição)", "none"),
}

# Símbolos explicitamente excluídos (non-beat ou artefato) — nunca geram label.
EXCLUDED_SYMBOLS_V3: frozenset[str] = frozenset({"~", "+", "x", "|"})

# Visão legada (v2.x) usada internamente pelo pipeline existente
# (AAMI_TO_INT em pipeline.py e modelos já treinados em pesquisa).
CANONICAL_TO_LEGACY: Dict[str, str] = {
    "N": "N",
    "S": "S",
    "V": "V",
    "FUSION": "F",
    "Q_OR_UNKNOWN": "Q",
}
LEGACY_TO_CANONICAL: Dict[str, str] = {v: k for k, v in CANONICAL_TO_LEGACY.items()}

# Nível 3 — ritmo (escopo: episódio) — usado por src/features/afdb_rhythm.py
RHYTHM_CLASSES_V3: List[str] = [
    "SINUS",
    "AFIB",
    "AFL",
    "JUNCTIONAL",
    "OTHER_RHYTHM",
    "UNKNOWN_RHYTHM",
]
AFDB_RHYTHM_MAP_V3: Dict[str, str] = {
    "N": "SINUS",
    "AFIB": "AFIB",
    "AFL": "AFL",
    "J": "JUNCTIONAL",
}


def map_symbols_v3(
    symbols: List[str],
) -> Tuple[List[str], List[bool], Dict[str, Any]]:
    """Mapeia símbolos WFDB para classes canônicas v3.

    Parameters
    ----------
    symbols : List[str]
        Símbolos de anotação WFDB (qualquer símbolo).

    Returns
    -------
    labels : List[str]
        Classes canônicas v3 apenas para símbolos mantidos.
    keep_mask : List[bool]
        Máscara booleana alinhada a ``symbols`` (True = mantido).
    stats : Dict[str, Any]
        Contagens por classe, por símbolo e de exclusões.
    """
    labels: List[str] = []
    keep_mask: List[bool] = []
    n_by_class: Dict[str, int] = {c: 0 for c in BEAT_CLASSES_V3}
    n_by_symbol: Dict[str, int] = {}
    n_excluded = 0
    n_unknown_excluded = 0

    for sym in symbols:
        sym_str = str(sym)
        n_by_symbol[sym_str] = n_by_symbol.get(sym_str, 0) + 1
        if sym_str in BEAT_MAP_V3:
            canonical, _, _ = BEAT_MAP_V3[sym_str]
            labels.append(canonical)
            keep_mask.append(True)
            n_by_class[canonical] += 1
        else:
            keep_mask.append(False)
            n_excluded += 1
            if sym_str not in EXCLUDED_SYMBOLS_V3:
                n_unknown_excluded += 1

    stats: Dict[str, Any] = {
        "ontology_version": ONTOLOGY_VERSION,
        "n_input": len(symbols),
        "n_kept": len(labels),
        "n_excluded": n_excluded,
        "n_unknown_excluded": n_unknown_excluded,
        "n_by_class": n_by_class,
        "n_by_symbol": n_by_symbol,
    }
    return labels, keep_mask, stats


def map_symbols_v3_legacy(
    symbols: List[str],
) -> Tuple[List[str], List[bool], Dict[str, Any]]:
    """Igual a :func:`map_symbols_v3`, mas devolve labels na visão legada v2.x.

    ``FUSION→F`` e ``Q_OR_UNKNOWN→Q``; mantém compatibilidade com o pipeline e
    com os encodings inteiros existentes (``AAMI_TO_INT``).
    """
    labels, keep_mask, stats = map_symbols_v3(symbols)
    legacy = [CANONICAL_TO_LEGACY[label] for label in labels]
    legacy_by_class: Dict[str, int] = {}
    for label in legacy:
        legacy_by_class[label] = legacy_by_class.get(label, 0) + 1
    stats["n_by_class_legacy"] = legacy_by_class
    return legacy, keep_mask, stats
