"""Testes para a API canônica de inferência em duas etapas.

Valida:
* Carregamento de modelos Keras float32 e scalers .pkl
* Aplicação de threshold JSON no Estágio 1
* Encaminhamento correto para o Estágio 2
* Retorno padronizado com classe final, confianças e thresholds
* Modo quantizado TFLite INT8
* Compatibilidade de shape (batch e amostra única)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.inference.two_stage_pipeline import TwoStageInferencePipeline
from src.models.backbone_1d import build_backbone_1d
from src.quantization.export_tflite import export_tflite
from src.quantization.ptq import representative_dataset_random


def _make_float32_artifacts(tmp_path: Path, n_samples: int = 30) -> tuple[Path, Path]:
    """Cria modelos Keras float32, scalers e threshold dummy para testes."""
    rng = np.random.default_rng(42)

    stage1_model = build_backbone_1d(input_len=500, num_classes=2)
    stage1_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    X1 = rng.standard_normal((n_samples, 500, 1)).astype(np.float32)
    y1 = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    stage1_model.fit(X1, y1, epochs=1, batch_size=8, verbose=0)

    stage2_model = build_backbone_1d(input_len=500, num_classes=3)
    stage2_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    X2 = rng.standard_normal((n_samples, 500, 1)).astype(np.float32)
    y2 = np.tile(np.arange(3, dtype=int), n_samples // 3 + 1)[:n_samples]
    stage2_model.fit(X2, y2, epochs=1, batch_size=8, verbose=0)

    scaler1 = StandardScaler()
    scaler1.fit(X1.reshape(-1, 1))
    scaler2 = StandardScaler()
    scaler2.fit(X2.reshape(-1, 1))

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    stage1_model.save(model_dir / "stage1_float32_v2.0.keras")
    stage2_model.save(model_dir / "stage2_float32_v2.0.keras")
    joblib.dump(scaler1, model_dir / "input_scaler_stage1_v2.0.pkl")
    joblib.dump(scaler2, model_dir / "input_scaler_stage2_v2.0.pkl")

    threshold_path = model_dir / "stage1_threshold_v2.0.json"
    threshold_path.write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")

    return model_dir, threshold_path


def _make_quantized_artifacts(tmp_path: Path, n_samples: int = 30) -> tuple[Path, Path]:
    """Cria modelos TFLite INT8, scalers e threshold dummy para testes."""
    rng = np.random.default_rng(42)

    stage1_model = build_backbone_1d(input_len=500, num_classes=2)
    stage1_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    X1 = rng.standard_normal((n_samples, 500, 1)).astype(np.float32)
    y1 = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    stage1_model.fit(X1, y1, epochs=1, batch_size=8, verbose=0)

    stage2_model = build_backbone_1d(input_len=500, num_classes=3)
    stage2_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    X2 = rng.standard_normal((n_samples, 500, 1)).astype(np.float32)
    y2 = np.tile(np.arange(3, dtype=int), n_samples // 3 + 1)[:n_samples]
    stage2_model.fit(X2, y2, epochs=1, batch_size=8, verbose=0)

    scaler1 = StandardScaler()
    scaler1.fit(X1.reshape(-1, 1))
    scaler2 = StandardScaler()
    scaler2.fit(X2.reshape(-1, 1))

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler1, model_dir / "input_scaler_stage1_v2.0.pkl")
    joblib.dump(scaler2, model_dir / "input_scaler_stage2_v2.0.pkl")

    threshold_path = model_dir / "stage1_threshold_v2.0.json"
    threshold_path.write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")

    rep1 = representative_dataset_random(X1, n_samples=10, seed=42)
    export_tflite(
        model=stage1_model,
        representative_data=rep1,
        output_dir=model_dir / "quantized",
        model_name="stage1_int8_v2.0",
        version="2.0.0",
        allow_float=False,
    )

    rep2 = representative_dataset_random(X2, n_samples=10, seed=42)
    export_tflite(
        model=stage2_model,
        representative_data=rep2,
        output_dir=model_dir / "quantized",
        model_name="stage2_int8_v2.0",
        version="2.0.0",
        allow_float=False,
    )

    return model_dir, threshold_path


class TestTwoStageInferencePipeline:
    """Testes da API canônica de inferência."""

    def test_float32_pipeline_batch(self, tmp_path):
        """Pipeline float32 deve classificar batch e retornar estrutura correta."""
        model_dir, threshold_path = _make_float32_artifacts(tmp_path)
        pipeline = TwoStageInferencePipeline(
            stage1_model_path=model_dir / "stage1_float32_v2.0.keras",
            stage1_scaler_path=model_dir / "input_scaler_stage1_v2.0.pkl",
            stage2_model_path=model_dir / "stage2_float32_v2.0.keras",
            stage2_scaler_path=model_dir / "input_scaler_stage2_v2.0.pkl",
            stage1_threshold_path=threshold_path,
            use_quantized=False,
        ).load()

        rng = np.random.default_rng(123)
        X = rng.standard_normal((5, 500, 1)).astype(np.float32)
        result = pipeline.predict(X)

        assert isinstance(result, dict)
        assert "class" in result
        assert "stage1_score" in result
        assert "stage2_scores" in result
        assert "stage1_threshold" in result
        assert "stage2_labels" in result
        assert len(result["class"]) == 5
        assert len(result["stage1_score"]) == 5
        assert len(result["stage2_scores"]) == 5
        assert all(cls in {"N", "S", "V", "F"} for cls in result["class"])

    def test_float32_pipeline_single_sample(self, tmp_path):
        """Pipeline deve aceitar amostra única com shape (500, 1)."""
        model_dir, threshold_path = _make_float32_artifacts(tmp_path)
        pipeline = TwoStageInferencePipeline(
            stage1_model_path=model_dir / "stage1_float32_v2.0.keras",
            stage1_scaler_path=model_dir / "input_scaler_stage1_v2.0.pkl",
            stage2_model_path=model_dir / "stage2_float32_v2.0.keras",
            stage2_scaler_path=model_dir / "input_scaler_stage2_v2.0.pkl",
            stage1_threshold_path=threshold_path,
            use_quantized=False,
        ).load()

        X = np.zeros((500, 1), dtype=np.float32)
        result = pipeline.predict(X)

        assert len(result["class"]) == 1
        assert len(result["stage1_score"]) == 1
        assert len(result["stage2_scores"]) == 1

    def test_threshold_from_json(self, tmp_path):
        """Deve carregar o threshold do arquivo JSON fornecido."""
        model_dir, _ = _make_float32_artifacts(tmp_path)
        threshold_path = model_dir / "custom_threshold.json"
        threshold_path.write_text(json.dumps({"threshold": 0.75}), encoding="utf-8")

        pipeline = TwoStageInferencePipeline(
            stage1_model_path=model_dir / "stage1_float32_v2.0.keras",
            stage1_scaler_path=model_dir / "input_scaler_stage1_v2.0.pkl",
            stage2_model_path=model_dir / "stage2_float32_v2.0.keras",
            stage2_scaler_path=model_dir / "input_scaler_stage2_v2.0.pkl",
            stage1_threshold_path=threshold_path,
            use_quantized=False,
        ).load()

        assert pipeline.stage1_threshold == pytest.approx(0.75)

    def test_from_directory_float32(self, tmp_path):
        """Factory from_directory deve localizar artefatos padrão v2.0."""
        model_dir, _ = _make_float32_artifacts(tmp_path)
        pipeline = TwoStageInferencePipeline.from_directory(model_dir, use_quantized=False).load()

        rng = np.random.default_rng(7)
        X = rng.standard_normal((2, 500, 1)).astype(np.float32)
        result = pipeline.predict(X)

        assert len(result["class"]) == 2

    def test_from_directory_quantized(self, tmp_path):
        """Factory from_directory deve localizar artefatos quantizados."""
        model_dir, _ = _make_quantized_artifacts(tmp_path)
        pipeline = TwoStageInferencePipeline.from_directory(model_dir, use_quantized=True).load()

        rng = np.random.default_rng(7)
        X = rng.standard_normal((2, 500, 1)).astype(np.float32)
        result = pipeline.predict(X)

        assert len(result["class"]) == 2
        assert all(cls in {"N", "S", "V", "F"} for cls in result["class"])

    def test_invalid_input_shape_raises(self, tmp_path):
        """Deve levantar ValueError para shape inesperado."""
        model_dir, threshold_path = _make_float32_artifacts(tmp_path)
        pipeline = TwoStageInferencePipeline(
            stage1_model_path=model_dir / "stage1_float32_v2.0.keras",
            stage1_scaler_path=model_dir / "input_scaler_stage1_v2.0.pkl",
            stage2_model_path=model_dir / "stage2_float32_v2.0.keras",
            stage2_scaler_path=model_dir / "input_scaler_stage2_v2.0.pkl",
            stage1_threshold_path=threshold_path,
            use_quantized=False,
        ).load()

        with pytest.raises(ValueError):
            pipeline.predict(np.zeros((5, 500), dtype=np.float32))
