"""E07R PD workflow ordering and frozen protocol contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.stage2_research.e07r_integrity import E07RIntegrityError
from src.stage2_research.pd_workflows import (
    build_pd_protocol_manifest,
    load_valid_e065_pd_selection,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pd_protocol_freezes_exact_matrix_and_gates() -> None:
    protocol = build_pd_protocol_manifest(PROJECT_ROOT)

    assert len(protocol.candidates) * len(protocol.folds) * len(protocol.seeds) == 100
    assert len(protocol.samplers) * len(protocol.folds) * len(protocol.seeds) == 150
    assert protocol.f1_f_gate == 0.15
    assert protocol.primary_target == 0.50
    assert protocol.deterministic is True
    assert protocol.device == "cpu"


def test_e07_pd_is_blocked_before_e065_pd_completion(tmp_path: Path) -> None:
    with pytest.raises(E07RIntegrityError, match="PD protocol is absent"):
        load_valid_e065_pd_selection(tmp_path)
