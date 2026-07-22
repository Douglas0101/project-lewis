"""Contrato da ontologia clínica única v3.0.0 (docs/rebuild_spec/01).

Garante: tabela única, FUSION≠AFIB, desconhecidos excluídos (nunca → Q),
Q_OR_UNKNOWN como classe de rejeição, e delegação dos mappers legados.
"""

from __future__ import annotations

import numpy as np

from src.features.aami_mapper import AAMI_CLASSES, map_annotations, map_annotations_array
from src.features.ontology_v3 import (
    AFDB_RHYTHM_MAP_V3,
    BEAT_CLASSES_V3,
    BEAT_MAP_V3,
    CANONICAL_TO_LEGACY,
    EXCLUDED_SYMBOLS_V3,
    ONTOLOGY_VERSION,
    RHYTHM_CLASSES_V3,
    map_symbols_v3,
    map_symbols_v3_legacy,
)


class TestOntologyV3Contract:
    def test_version_pinned(self):
        assert ONTOLOGY_VERSION == "3.0.0"

    def test_beat_classes(self):
        assert BEAT_CLASSES_V3 == ["N", "S", "V", "FUSION", "Q_OR_UNKNOWN"]

    def test_fusion_is_not_afib(self):
        """D2: F (beat) nunca significa fibrilação atrial."""
        canonical, _, _ = BEAT_MAP_V3["F"]
        assert canonical == "FUSION"
        assert "AFIB" not in BEAT_CLASSES_V3
        assert "AFIB" in RHYTHM_CLASSES_V3
        assert AFDB_RHYTHM_MAP_V3["AFIB"] == "AFIB"

    def test_unknown_symbols_excluded_never_q(self):
        """D2/D4: símbolo desconhecido é excluído, nunca mapeado para Q."""
        labels, keep, stats = map_symbols_v3(["N", "UNKNOWN", "V"])
        assert labels == ["N", "V"]
        assert keep == [True, False, True]
        assert stats["n_unknown_excluded"] == 1
        assert "Q_OR_UNKNOWN" not in labels

    def test_pipe_artifact_excluded(self):
        labels, keep, stats = map_symbols_v3(["N", "|", "V"])
        assert labels == ["N", "V"]
        assert keep == [True, False, True]
        assert "|" in EXCLUDED_SYMBOLS_V3

    def test_non_beat_symbols_excluded(self):
        labels, keep, _ = map_symbols_v3(["N", "~", "+", "x", "V"])
        assert labels == ["N", "V"]
        assert keep == [True, False, False, False, True]

    def test_q_symbols_are_rejection_class(self):
        for sym in ["/", "f", "Q"]:
            canonical, rule, _ = BEAT_MAP_V3[sym]
            assert canonical == "Q_OR_UNKNOWN"
            assert "rejeição" in rule

    def test_table_integrity(self):
        for sym, (canonical, rule, ambiguity) in BEAT_MAP_V3.items():
            assert canonical in BEAT_CLASSES_V3
            assert rule
            assert ambiguity in {"none", "symbol_conflict", "border_region", "rhythm_overlap"}

    def test_legacy_view(self):
        labels, keep, _ = map_symbols_v3_legacy(["N", "F", "/", "V"])
        assert labels == ["N", "F", "Q", "V"]
        assert CANONICAL_TO_LEGACY["FUSION"] == "F"
        assert CANONICAL_TO_LEGACY["Q_OR_UNKNOWN"] == "Q"


class TestAamiMapperDelegation:
    def test_unknown_excluded_in_legacy_api(self):
        labels, stats = map_annotations(["N", "UNKNOWN", "V"])
        assert labels == ["N", "V"]
        assert stats["n_excluded"] == 1
        assert stats["n_unmapped"] == 0

    def test_pipe_excluded_in_legacy_api(self):
        labels, stats = map_annotations(["N", "|", "V"])
        assert labels == ["N", "V"]
        assert stats["n_excluded"] == 1

    def test_legacy_classes_unchanged(self):
        assert AAMI_CLASSES == ["N", "S", "V", "F", "Q"]

    def test_array_version(self):
        labels, _ = map_annotations_array(np.array(["N", "V", "S", "F", "Q"]))
        assert list(labels) == ["N", "V", "S", "F", "Q"]

    def test_stats_counts(self):
        labels, stats = map_annotations(["N", "N", "V", "V", "V", "S", "F", "Q"])
        assert stats["n_by_class"]["N"] == 2
        assert stats["n_by_class"]["V"] == 3
        assert stats["n_by_class"]["S"] == 1
        assert stats["n_by_class"]["F"] == 1
        assert stats["n_by_class"]["Q"] == 1
