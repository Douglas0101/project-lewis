"""API canônica de inferência em duas etapas para Project-Lewis v2.0.

Estágio 1: classificação binária N vs Anormal.
Estágio 2: classificação multiclasse S vs V vs F (executado apenas quando
o Estágio 1 indica Anormal).

Suporta modelos Keras float32 ou modelos TFLite INT8 quantizados.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
from src.models.keras_loader import load_keras_model

from src.inference.quantized_runner import QuantizedModelRunner

LOGGER = logging.getLogger("lewis.inference.two_stage_pipeline")

AAMI_FINAL_CLASSES = ["N", "S", "V", "F"]
STAGE2_CLASS_NAMES = ["S", "V", "F"]
STAGE2_TO_AAMI = {0: "S", 1: "V", 2: "F"}


class TwoStageInferencePipeline:
    """Pipeline de inferência canônico de duas etapas.

    Parameters
    ----------
    stage1_model_path : Path | str
        Caminho do modelo do Estágio 1 (``.keras`` ou ``.tflite``).
    stage1_scaler_path : Path | str
        Caminho do scaler serializado do Estágio 1 (``.pkl``).
    stage2_model_path : Path | str
        Caminho do modelo do Estágio 2 (``.keras`` ou ``.tflite``).
    stage2_scaler_path : Path | str
        Caminho do scaler serializado do Estágio 2 (``.pkl``).
    stage1_threshold_path : Path | str | None, optional
        Caminho do JSON com o threshold do Estágio 1. Se None, usa 0.5.
    use_quantized : bool, optional
        Se True, carrega modelos ``.tflite`` INT8; caso contrário, Keras float32.
    """

    def __init__(
        self,
        stage1_model_path: Path | str,
        stage1_scaler_path: Path | str,
        stage2_model_path: Path | str,
        stage2_scaler_path: Path | str,
        stage1_threshold_path: Path | str | None = None,
        use_quantized: bool = False,
    ) -> None:
        self.stage1_model_path = Path(stage1_model_path)
        self.stage1_scaler_path = Path(stage1_scaler_path)
        self.stage2_model_path = Path(stage2_model_path)
        self.stage2_scaler_path = Path(stage2_scaler_path)
        self.stage1_threshold_path = (
            Path(stage1_threshold_path) if stage1_threshold_path is not None else None
        )
        self.use_quantized = use_quantized

        self.stage1_threshold = 0.5
        self.stage1_model: tf.keras.Model | QuantizedModelRunner | None = None
        self.stage2_model: tf.keras.Model | QuantizedModelRunner | None = None
        self.stage1_scaler: Any = None
        self.stage2_scaler: Any = None

    @classmethod
    def from_directory(
        cls,
        model_dir: Path | str,
        use_quantized: bool = False,
    ) -> TwoStageInferencePipeline:
        """Cria pipeline a partir de um diretório com artefatos v2.0 padrão.

        Parameters
        ----------
        model_dir : Path | str
            Diretório que contém os artefatos do modelo.
        use_quantized : bool, optional
            Se True, usa ``models/quantized/stage{1,2}_int8_v2.0.tflite``;
            caso contrário, usa ``stage{1,2}_float32_v2.0.keras``.

        Returns
        -------
        TwoStageInferencePipeline
            Instância configurada (ainda não carregada).
        """
        model_dir = Path(model_dir)

        if use_quantized:
            stage1_model = model_dir / "quantized" / "stage1_int8_v2.0.tflite"
            stage2_model = model_dir / "quantized" / "stage2_int8_v2.0.tflite"
        else:
            stage1_model = model_dir / "stage1_float32_v2.0.keras"
            stage2_model = model_dir / "stage2_float32_v2.0.keras"

        return cls(
            stage1_model_path=stage1_model,
            stage1_scaler_path=model_dir / "input_scaler_stage1_v2.0.pkl",
            stage2_model_path=stage2_model,
            stage2_scaler_path=model_dir / "input_scaler_stage2_v2.0.pkl",
            stage1_threshold_path=model_dir / "stage1_threshold_v2.0.json",
            use_quantized=use_quantized,
        )

    def load(self) -> TwoStageInferencePipeline:
        """Carrega scalers, modelos e threshold.

        Returns
        -------
        TwoStageInferencePipeline
            A própria instância, permitindo encadeamento.
        """
        LOGGER.info("Carregando pipeline v2.0 (quantizado=%s)", self.use_quantized)

        self.stage1_scaler = joblib.load(self.stage1_scaler_path)
        self.stage2_scaler = joblib.load(self.stage2_scaler_path)

        if self.use_quantized:
            self.stage1_model = QuantizedModelRunner(self.stage1_model_path).allocate()
            self.stage2_model = QuantizedModelRunner(self.stage2_model_path).allocate()
        else:
            self.stage1_model = load_keras_model(str(self.stage1_model_path), compile=False)
            self.stage2_model = load_keras_model(str(self.stage2_model_path), compile=False)

        if self.stage1_threshold_path is not None:
            data = json.loads(Path(self.stage1_threshold_path).read_text(encoding="utf-8"))
            self.stage1_threshold = float(data.get("threshold", 0.5))

        LOGGER.info("Threshold Estágio 1: %.4f", self.stage1_threshold)
        return self

    def _normalize(self, X: np.ndarray, scaler: Any) -> np.ndarray:
        """Aplica normalização z-score usando o scaler fornecido."""
        n, seq_len, channels = X.shape
        return scaler.transform(X.reshape(-1, channels)).reshape(n, seq_len, channels)

    def _run_stage1(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Executa o Estágio 1 e retorna predições + probabilidade Anormal."""
        assert self.stage1_model is not None
        X_norm = self._normalize(X, self.stage1_scaler)
        logits = self._forward(self.stage1_model, X_norm)
        score_anormal = logits[:, 1]
        y_pred = (score_anormal >= self.stage1_threshold).astype(np.int64)
        return y_pred, score_anormal

    def _run_stage2(self, X: np.ndarray) -> np.ndarray:
        """Executa o Estágio 2 e retorna labels S/V/F."""
        assert self.stage2_model is not None
        X_norm = self._normalize(X, self.stage2_scaler)
        logits = self._forward(self.stage2_model, X_norm)
        return np.argmax(logits, axis=1).astype(np.int64)

    def _stage2_scores(self, X: np.ndarray) -> np.ndarray:
        """Retorna os logits/probabilidades do Estágio 2 para as amostras."""
        assert self.stage2_model is not None
        X_norm = self._normalize(X, self.stage2_scaler)
        return self._forward(self.stage2_model, X_norm)

    def _forward(
        self,
        model: tf.keras.Model | QuantizedModelRunner,
        X: np.ndarray,
    ) -> np.ndarray:
        """Executa forward pass no modelo, abstraindo Keras ou TFLite."""
        if isinstance(model, QuantizedModelRunner):
            outputs = [model.run(x[np.newaxis, ...]) for x in X]
            return np.concatenate(outputs, axis=0)

        return model.predict(X, verbose=0)

    @staticmethod
    def _build_integrated_predictions(
        stage1_pred: np.ndarray,
        stage2_pred: np.ndarray,
    ) -> np.ndarray:
        """Combina predições: N=0, S=1, V=2, F=3."""
        integrated = np.zeros_like(stage1_pred)
        abnormal_mask = stage1_pred == 1
        n_abnormal = int(abnormal_mask.sum())

        if n_abnormal > 0:
            if len(stage2_pred) != n_abnormal:
                raise ValueError(
                    f"stage2_pred deve ter {n_abnormal} amostras (Anormal), "
                    f"mas tem {len(stage2_pred)}"
                )
            integrated[abnormal_mask] = stage2_pred + 1

        return integrated

    def predict(self, X: np.ndarray) -> dict[str, Any]:
        """Executa o pipeline completo e retorna resultado estruturado.

        Parameters
        ----------
        X : np.ndarray
            Batimento(s) ECG com shape ``(500, 1)`` ou ``(n, 500, 1)``.

        Returns
        -------
        dict
            Dicionário com:
            - ``class``: lista de classes finais (N, S, V, F);
            - ``stage1_score``: probabilidade/logit da classe Anormal;
            - ``stage2_scores``: matriz (n, 3) de scores do Estágio 2;
            - ``stage1_threshold``: threshold usado no Estágio 1;
            - ``stage2_labels``: rótulos do Estágio 2 (S, V, F).
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        if X.ndim != 3:
            raise ValueError(f"X deve ter shape (n, 500, 1) ou (500, 1); recebido {X.shape}")

        y_stage1, score_anormal = self._run_stage1(X)
        abnormal_mask = y_stage1 == 1
        n_abnormal = int(abnormal_mask.sum())

        stage2_scores = np.zeros((X.shape[0], 3), dtype=np.float32)
        if n_abnormal > 0:
            X_abnormal = X[abnormal_mask]
            y_stage2 = self._run_stage2(X_abnormal)
            integrated = self._build_integrated_predictions(y_stage1, y_stage2)
            stage2_scores[abnormal_mask] = self._stage2_scores(X_abnormal)
        else:
            integrated = y_stage1.copy()

        final_classes = [AAMI_FINAL_CLASSES[idx] for idx in integrated.tolist()]

        return {
            "class": final_classes,
            "stage1_score": score_anormal.tolist(),
            "stage2_scores": stage2_scores.tolist(),
            "stage1_threshold": self.stage1_threshold,
            "stage2_labels": STAGE2_CLASS_NAMES,
        }
