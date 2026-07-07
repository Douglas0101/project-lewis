"""API de inferência two-stage para MLPs sobre features (Project-Lewis v2.3).

Diferente do pipeline CNN de sinais (``TwoStageInferencePipeline``), este
pipeline:

1. Extrai 16 features morfológicas/time-domain de cada segmento ECG.
2. Aplica o scaler treinado.
3. Executa MLP do Estágio 1 (N vs Anormal).
4. Para amostras classificadas como Anormal, executa MLP do Estágio 2 (S vs V vs F).

Suporta modelos Keras float32 e TFLite INT8 quantizados.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import tensorflow as tf

from src.inference.feature_extractor import FeatureExtractor, FEATURE_NAMES
from src.inference.quantized_runner import QuantizedModelRunner

LOGGER = logging.getLogger("lewis.inference.two_stage_mlp_pipeline")

AAMI_FINAL_CLASSES = ["N", "S", "V", "F"]
STAGE2_CLASS_NAMES = ["S", "V", "F"]
STAGE2_TO_AAMI = {0: "S", 1: "V", 2: "F"}


class TwoStageMLPPipeline:
    """Pipeline two-stage baseado em features para os modelos MLP v2.3."""

    def __init__(
        self,
        stage1_model_path: Path | str,
        stage1_scaler_path: Path | str,
        stage2_model_path: Path | str,
        stage2_scaler_path: Path | str,
        stage1_threshold_path: Path | str | None = None,
        use_quantized: bool = False,
        fs: float = 500.0,
    ) -> None:
        self.stage1_model_path = Path(stage1_model_path)
        self.stage1_scaler_path = Path(stage1_scaler_path)
        self.stage2_model_path = Path(stage2_model_path)
        self.stage2_scaler_path = Path(stage2_scaler_path)
        self.stage1_threshold_path = (
            Path(stage1_threshold_path) if stage1_threshold_path is not None else None
        )
        self.use_quantized = use_quantized
        self.fs = fs

        self.stage1_threshold = 0.5
        self.stage1_model: tf.keras.Model | QuantizedModelRunner | None = None
        self.stage2_model: tf.keras.Model | QuantizedModelRunner | None = None
        self.stage1_scaler: Any = None
        self.stage2_scaler: Any = None
        self.feature_extractor = FeatureExtractor(fs=fs)

    @classmethod
    def from_directory(
        cls,
        model_dir: Path | str,
        use_quantized: bool = False,
    ) -> TwoStageMLPPipeline:
        """Cria pipeline a partir dos artefatos v2.3 padrão em ``model_dir``."""
        model_dir = Path(model_dir)

        if use_quantized:
            stage1_model = model_dir / "quantized" / "stage1_int8_v2.3.tflite"
            stage2_model = model_dir / "quantized" / "stage2_int8_v2.3.tflite"
        else:
            stage1_model = model_dir / "stage1_float32_v2.3.keras"
            stage2_model = model_dir / "stage2_float32_v2.3.keras"

        return cls(
            stage1_model_path=stage1_model,
            stage1_scaler_path=model_dir / "input_scaler_stage1_v2.3.pkl",
            stage2_model_path=stage2_model,
            stage2_scaler_path=model_dir / "input_scaler_stage2_v2.3.pkl",
            stage1_threshold_path=model_dir / "stage1_threshold_v2.3.json",
            use_quantized=use_quantized,
        )

    def load(self) -> TwoStageMLPPipeline:
        """Carrega scalers, modelos e threshold."""
        LOGGER.info("Carregando pipeline MLP v2.3 (quantizado=%s)", self.use_quantized)

        self.stage1_scaler = joblib.load(self.stage1_scaler_path)
        self.stage2_scaler = joblib.load(self.stage2_scaler_path)

        if self.use_quantized:
            self.stage1_model = QuantizedModelRunner(self.stage1_model_path).allocate()
            self.stage2_model = QuantizedModelRunner(self.stage2_model_path).allocate()
        else:
            self.stage1_model = tf.keras.models.load_model(
                str(self.stage1_model_path), compile=False
            )
            self.stage2_model = tf.keras.models.load_model(
                str(self.stage2_model_path), compile=False
            )

        if self.stage1_threshold_path is not None:
            data = json.loads(Path(self.stage1_threshold_path).read_text(encoding="utf-8"))
            self.stage1_threshold = float(data.get("threshold", 0.5))

        LOGGER.info("Threshold Estágio 1: %.4f", self.stage1_threshold)
        return self

    def _extract_features(
        self,
        X: np.ndarray,
        r_peaks: np.ndarray | None = None,
        temporal_features: List[Dict[str, float]] | None = None,
    ) -> np.ndarray:
        """Extrai e empilha features dos segmentos ECG."""
        feats = self.feature_extractor.extract_from_segments(
            X,
            r_peaks=r_peaks,
            precomputed_temporal=temporal_features,
        )
        return FeatureExtractor.features_to_array(feats, feature_names=FEATURE_NAMES)

    def _run_stage1(self, X_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Executa o Estágio 1 e retorna predições + probabilidade Anormal."""
        X_scaled = self.stage1_scaler.transform(X_features)
        logits = self._forward(self.stage1_model, X_scaled)
        score_anormal = logits[:, 1]
        y_pred = (score_anormal >= self.stage1_threshold).astype(np.int64)
        return y_pred, score_anormal

    def _run_stage2(self, X_features: np.ndarray) -> np.ndarray:
        """Executa o Estágio 2 e retorna labels S/V/F."""
        X_scaled = self.stage2_scaler.transform(X_features)
        logits = self._forward(self.stage2_model, X_scaled)
        return np.argmax(logits, axis=1).astype(np.int64)

    def _stage2_scores(self, X_features: np.ndarray) -> np.ndarray:
        """Retorna scores do Estágio 2 para as amostras."""
        X_scaled = self.stage2_scaler.transform(X_features)
        return self._forward(self.stage2_model, X_scaled)

    def _forward(
        self,
        model: tf.keras.Model | QuantizedModelRunner,
        X: np.ndarray,
    ) -> np.ndarray:
        """Forward pass abstraindo Keras ou TFLite."""
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

    def predict_from_features(self, X_features: np.ndarray) -> dict[str, Any]:
        """Executa o pipeline completo a partir de features pré-extraídas.

        Parameters
        ----------
        X_features : np.ndarray
            Features de shape ``(n, n_features)`` já escalonadas ou não.
            O scaler do Estágio 1/2 é aplicado internamente.

        Returns
        -------
        dict
            Mesmo formato de ``predict``.
        """
        X_features = np.asarray(X_features, dtype=np.float32)
        if X_features.ndim != 2:
            shape = X_features.shape
            raise ValueError(f"X_features deve ter shape (n, n_features); recebido {shape}")

        y_stage1, score_anormal = self._run_stage1(X_features)
        abnormal_mask = y_stage1 == 1
        n_abnormal = int(abnormal_mask.sum())

        stage2_scores = np.zeros((X_features.shape[0], 3), dtype=np.float32)
        if n_abnormal > 0:
            X_abnormal_features = X_features[abnormal_mask]
            y_stage2 = self._run_stage2(X_abnormal_features)
            integrated = self._build_integrated_predictions(y_stage1, y_stage2)
            stage2_scores[abnormal_mask] = self._stage2_scores(X_abnormal_features)
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

    def predict(
        self,
        X: np.ndarray,
        r_peaks: np.ndarray | None = None,
        temporal_features: List[Dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        """Executa o pipeline completo sobre segmentos ECG.

        Parameters
        ----------
        X : np.ndarray
            Batimento(s) ECG com shape ``(n, 500)`` ou ``(n, 500, 1)``.
        r_peaks : np.ndarray | None
            Índices globais dos R-peaks no sinal completo, shape ``(n,)``.
            Usado apenas se ``temporal_features`` não for fornecido.
        temporal_features : list[dict] | None
            Features time-domain pré-computadas para cada batimento. Se
            fornecido, tem precedência sobre ``r_peaks``.

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
        if X.ndim == 3:
            X = X[..., 0]
        if X.ndim != 2:
            raise ValueError(f"X deve ter shape (n, 500) ou (n, 500, 1); recebido {X.shape}")

        X_features = self._extract_features(
            X,
            r_peaks=r_peaks,
            temporal_features=temporal_features,
        )
        return self.predict_from_features(X_features)
