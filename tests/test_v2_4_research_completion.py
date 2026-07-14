"""Testes de conclusão da research branch v2.4 (E09)."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = PROJECT_ROOT / "experiments" / "stage2_v2.4_research"
MODELS_DIR = PROJECT_ROOT / "models"


def test_research_report_exists():
    assert (PROJECT_ROOT / "docs" / "stage2_v2.4_research_report.md").exists()


def test_all_manifests_exist():
    for e in range(9):
        dirs = list(RESEARCH_DIR.glob(f"E{e:02d}_*"))
        assert dirs, f"E{e:02d} nao encontrado"
        assert (dirs[0] / f"E{e:02d}_manifest.json").exists(), f"Manifesto E{e:02d} ausente"


def test_no_v2_4_artifacts_in_models():
    """Garante que nenhum artefato v2.4 foi publicado acidentalmente."""
    for path in MODELS_DIR.iterdir():
        if path.is_file() and "v2.4" in path.name:
            pytest.fail(f"Artefato v2.4 encontrado em models/: {path}")


def test_v2_3_artifacts_preserved():
    assert (MODELS_DIR / "stage1_float32_v2.3.keras").exists()
    assert (MODELS_DIR / "stage2_float32_v2.3.keras").exists()
    assert (MODELS_DIR / "input_scaler_stage1_v2.3.pkl").exists()
    assert (MODELS_DIR / "input_scaler_stage2_v2.3.pkl").exists()
    assert (MODELS_DIR / "stage1_threshold_v2.3.json").exists()
    assert (MODELS_DIR / "stage2_thresholds_v2.3.json").exists()
