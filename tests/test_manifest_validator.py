"""Testes de validação de manifests para modelos e scalers v2.4 (E02)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _feature_schema_hash(names: list[str]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(names, ensure_ascii=True).encode()).hexdigest()


def test_manifest_validator_ok():
    from src.inference.manifest_validator import (
        compute_feature_schema_hash,
        load_and_validate_manifest,
        write_manifest,
    )

    features = ["a", "b", "c"]
    expected_hash = compute_feature_schema_hash(features)
    dataset_hash = "abc123"

    tmp = Path("/tmp/manifest_test_ok")
    tmp.mkdir(parents=True, exist_ok=True)
    write_manifest(tmp / "m.json", features, dataset_hash)
    manifest = load_and_validate_manifest(tmp / "m.json", expected_hash, dataset_hash, "model")
    assert manifest["feature_schema_hash"] == expected_hash
    assert manifest["dataset_manifest_hash"] == dataset_hash


def test_feature_order_mismatch_fails():
    from src.inference.manifest_validator import (
        ManifestValidationError,
        compute_feature_schema_hash,
        write_manifest,
    )

    features = ["a", "b", "c"]
    wrong_hash = compute_feature_schema_hash(["c", "b", "a"])
    tmp = Path("/tmp/manifest_test_order")
    tmp.mkdir(parents=True, exist_ok=True)
    write_manifest(tmp / "m.json", features, "abc123")
    with pytest.raises(ManifestValidationError):
        from src.inference.manifest_validator import load_and_validate_manifest

        load_and_validate_manifest(tmp / "m.json", wrong_hash, "abc123", "model")


def test_feature_missing_fails():
    from src.inference.manifest_validator import (
        ManifestValidationError,
        compute_feature_schema_hash,
        write_manifest,
    )

    features = ["a", "b", "c"]
    wrong_hash = compute_feature_schema_hash(["a", "b"])
    tmp = Path("/tmp/manifest_test_missing")
    tmp.mkdir(parents=True, exist_ok=True)
    write_manifest(tmp / "m.json", features, "abc123")
    with pytest.raises(ManifestValidationError):
        from src.inference.manifest_validator import load_and_validate_manifest

        load_and_validate_manifest(tmp / "m.json", wrong_hash, "abc123", "model")


def test_feature_extra_fails():
    from src.inference.manifest_validator import (
        ManifestValidationError,
        compute_feature_schema_hash,
        write_manifest,
    )

    features = ["a", "b"]
    wrong_hash = compute_feature_schema_hash(["a", "b", "c"])
    tmp = Path("/tmp/manifest_test_extra")
    tmp.mkdir(parents=True, exist_ok=True)
    write_manifest(tmp / "m.json", features, "abc123")
    with pytest.raises(ManifestValidationError):
        from src.inference.manifest_validator import load_and_validate_manifest

        load_and_validate_manifest(tmp / "m.json", wrong_hash, "abc123", "model")


def test_dataset_hash_mismatch_fails():
    from src.inference.manifest_validator import (
        ManifestValidationError,
        compute_feature_schema_hash,
        write_manifest,
    )

    features = ["a", "b", "c"]
    expected_hash = compute_feature_schema_hash(features)
    tmp = Path("/tmp/manifest_test_dataset")
    tmp.mkdir(parents=True, exist_ok=True)
    write_manifest(tmp / "m.json", features, "abc123")
    with pytest.raises(ManifestValidationError):
        from src.inference.manifest_validator import load_and_validate_manifest

        load_and_validate_manifest(tmp / "m.json", expected_hash, "xyz789", "model")


@pytest.mark.requires_artifacts
def test_pipeline_v2_3_loads_without_manifest():
    """Pipeline v2.3 continua carregando sem manifest quando strict=False."""
    from src.inference.two_stage_mlp_pipeline import TwoStageMLPPipeline

    pipeline = TwoStageMLPPipeline.from_directory(PROJECT_ROOT / "models", strict_manifest=False)
    pipeline.load()
    assert pipeline.stage1_model is not None
    assert pipeline.stage2_model is not None


@pytest.mark.requires_artifacts
def test_pipeline_strict_manifest_rejects_missing_manifest():
    """strict=True rejeita modelos v2.3 sem manifest."""
    from src.inference.manifest_validator import ManifestValidationError
    from src.inference.two_stage_mlp_pipeline import TwoStageMLPPipeline

    pipeline = TwoStageMLPPipeline.from_directory(PROJECT_ROOT / "models", strict_manifest=True)
    with pytest.raises(ManifestValidationError):
        pipeline.load()


@pytest.mark.requires_artifacts
def test_pipeline_rejects_incompatible_manifest(tmp_path: Path) -> None:
    """Pipeline rejeita manifest com feature schema incompatível."""
    from src.inference.manifest_validator import ManifestValidationError
    from src.inference.two_stage_mlp_pipeline import TwoStageMLPPipeline

    # Copia modelo v2.3 para tmp e cria manifest com schema errado
    src_model = PROJECT_ROOT / "models" / "stage2_float32_v2.3.keras"
    dst_model = tmp_path / "stage2_float32_v2.4.keras"
    dst_scaler = tmp_path / "input_scaler_stage2_v2.4.pkl"
    shutil.copy2(src_model, dst_model)
    shutil.copy2(PROJECT_ROOT / "models" / "input_scaler_stage2_v2.3.pkl", dst_scaler)

    try:
        feature_names = json.load(
            open(PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features.json")
        )["feature_names"]
    except Exception as exc:
        raise AssertionError(f"Falha ao carregar feature names: {exc}") from exc
    from src.inference.manifest_validator import compute_feature_schema_hash

    expected_feature_hash = compute_feature_schema_hash(feature_names)
    manifest = {
        "feature_schema_hash": "wrong_hash",
        "dataset_manifest_hash": "also_wrong",
        "feature_names": ["wrong", "feature", "names"],
    }
    (tmp_path / "stage2_float32_v2.4.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    # Para stage1, reutilizamos v2.3 sem manifest (strict=False permite)
    stage1_model = PROJECT_ROOT / "models" / "stage1_float32_v2.3.keras"
    stage1_scaler = PROJECT_ROOT / "models" / "input_scaler_stage1_v2.3.pkl"
    stage1_threshold = PROJECT_ROOT / "models" / "stage1_threshold_v2.3.json"

    pipeline = TwoStageMLPPipeline(
        stage1_model_path=stage1_model,
        stage1_scaler_path=stage1_scaler,
        stage2_model_path=dst_model,
        stage2_scaler_path=dst_scaler,
        stage1_threshold_path=stage1_threshold,
        strict_manifest=False,  # stage1 sem manifest, stage2 com manifest invalido
        expected_feature_schema_hash=expected_feature_hash,
        expected_dataset_hash="expected_dataset_hash",
    )
    with pytest.raises(ManifestValidationError):
        pipeline.load()


@pytest.mark.requires_artifacts
def test_pipeline_accepts_compatible_manifest(tmp_path: Path) -> None:
    """Pipeline carrega modelo quando manifest é compatível."""
    from src.inference.manifest_validator import compute_feature_schema_hash
    from src.inference.two_stage_mlp_pipeline import TwoStageMLPPipeline

    src_model = PROJECT_ROOT / "models" / "stage2_float32_v2.3.keras"
    dst_model = tmp_path / "stage2_float32_v2.4.keras"
    dst_scaler = tmp_path / "input_scaler_stage2_v2.4.pkl"
    shutil.copy2(src_model, dst_model)
    shutil.copy2(PROJECT_ROOT / "models" / "input_scaler_stage2_v2.3.pkl", dst_scaler)

    try:
        with open(PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features.json") as f:
            feature_names = json.load(f)["feature_names"]
    except Exception as exc:
        raise AssertionError(f"Falha ao carregar feature names: {exc}") from exc
    dataset_hash = "placeholder_dataset_hash"
    manifest = {
        "feature_schema_hash": compute_feature_schema_hash(feature_names),
        "dataset_manifest_hash": dataset_hash,
        "feature_names": feature_names,
    }
    (tmp_path / "stage2_float32_v2.4.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    stage1_model = PROJECT_ROOT / "models" / "stage1_float32_v2.3.keras"
    stage1_scaler = PROJECT_ROOT / "models" / "input_scaler_stage1_v2.3.pkl"
    stage1_threshold = PROJECT_ROOT / "models" / "stage1_threshold_v2.3.json"

    pipeline = TwoStageMLPPipeline(
        stage1_model_path=stage1_model,
        stage1_scaler_path=stage1_scaler,
        stage2_model_path=dst_model,
        stage2_scaler_path=dst_scaler,
        stage1_threshold_path=stage1_threshold,
        strict_manifest=False,
        expected_feature_schema_hash=compute_feature_schema_hash(feature_names),
        expected_dataset_hash=dataset_hash,
    )
    pipeline.load()
    assert pipeline.stage2_manifest is not None
