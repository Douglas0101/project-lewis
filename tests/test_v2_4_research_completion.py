"""Testes de conclusão da research branch v2.4 (E09)."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = PROJECT_ROOT / "experiments" / "stage2_v2.4_research"
MODELS_DIR = PROJECT_ROOT / "models"
CANONICAL_STAGE_DIRS = {
    "E00": "E00_baseline_snapshot",
    "E01": "E01_patient_distribution",
    "E02": "E02_manifest_immutable",
    "E03": "E03_split_protocol",
    "E04": "E04_qg5_gates",
    "E05": "E05_feature_separability",
    "E06": "E06_feature_engineering",
    "E07": "E07_label_audit",
    "E08": "E08_resampled_focal_mlp",
}
E06_NON_COMPLETION_DIRS = ("E06_reopened", "E06_5")


def test_research_report_exists():
    assert (PROJECT_ROOT / "docs" / "stage2_v2.4_research_report.md").exists()


@pytest.mark.requires_artifacts
def test_all_canonical_manifests_exist():
    for stage, directory_name in CANONICAL_STAGE_DIRS.items():
        stage_dir = RESEARCH_DIR / directory_name
        assert stage_dir.is_dir(), f"Diretorio canonico {stage} ausente: {directory_name}"
        assert (stage_dir / f"{stage}_manifest.json").is_file(), f"Manifesto {stage} ausente"


@pytest.mark.requires_artifacts
@pytest.mark.parametrize("directory_name", E06_NON_COMPLETION_DIRS)
def test_reopened_e06_paths_are_not_historical_completion(directory_name: str):
    stage_dir = RESEARCH_DIR / directory_name
    assert stage_dir.is_dir(), f"Diretorio E06 esperado ausente: {directory_name}"
    assert not (
        stage_dir / "E06_manifest.json"
    ).exists(), f"{directory_name} nao pode declarar conclusao historica E06"


def test_no_v2_4_artifacts_in_models():
    """Garante que nenhum artefato v2.4 foi publicado acidentalmente."""
    for path in MODELS_DIR.iterdir():
        if path.is_file() and "v2.4" in path.name:
            pytest.fail(f"Artefato v2.4 encontrado em models/: {path}")


@pytest.mark.requires_artifacts
def test_v2_3_artifacts_preserved():
    assert (MODELS_DIR / "stage1_float32_v2.3.keras").exists()
    assert (MODELS_DIR / "stage2_float32_v2.3.keras").exists()
    assert (MODELS_DIR / "input_scaler_stage1_v2.3.pkl").exists()
    assert (MODELS_DIR / "input_scaler_stage2_v2.3.pkl").exists()
    assert (MODELS_DIR / "stage1_threshold_v2.3.json").exists()
    assert (MODELS_DIR / "stage2_thresholds_v2.3.json").exists()
