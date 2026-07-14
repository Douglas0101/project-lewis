"""MLPStage1Runner — inferência do Estágio 1 baseada em features.

Este runner substitui a CNN raw-signal pelo MLP treinado sobre features
morfológicas e time-domain, conforme resultado do fallback v2.1.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from src.models.keras_loader import load_keras_model


class MLPStage1Runner:
    """Executa inferência do Estágio 1 (N vs Anormal) via MLP sobre features.

    Parameters
    ----------
    model_path : Path | str
        Caminho para o modelo Keras (ex.: models/stage1_mlp_features_v2.1.keras).
    scaler_path : Path | str
        Caminho para o scaler sklearn.
    config_path : Path | str
        Caminho para o JSON com os nomes das features.
    """

    def __init__(
        self,
        model_path: Path | str,
        scaler_path: Path | str,
        config_path: Path | str,
    ) -> None:
        self.model = load_keras_model(str(model_path), compile=False)
        self.scaler = joblib.load(scaler_path)
        try:
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid Stage 1 config: {config_path}") from error
        if not isinstance(config, dict) or not isinstance(config.get("feature_names"), list):
            raise ValueError("Stage 1 config must contain a feature_names list")
        self.config = config
        self.feature_names: list[str] = [str(name) for name in config["feature_names"]]

    def predict(
        self,
        features: dict[str, np.ndarray],
        threshold: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Retorna classes e probabilidades para um batch de batimentos.

        Parameters
        ----------
        features : dict
            Dicionário com arrays numpy por feature (mesmos nomes do config).
        threshold : float, optional
            Threshold para classe Anormal. Se None, usa argmax.

        Returns
        -------
        dict
            {"y_pred": array, "y_proba": array shape (n, 2)}.
        """
        X = np.column_stack([features[name] for name in self.feature_names])
        X = X.astype(np.float32)

        # Trata NaN/Inf da mesma forma do treinamento (mediana não é stateful;
        # aqui usamos 0.0 para simplificar a inferência embarcada).
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(X)

        # Keras 3 accepts integer verbosity modes at runtime; its unannotated
        # signature makes Pyright infer ``str`` from the default ``"auto"``.
        y_proba = self.model.predict(
            X, batch_size=1024, verbose=0  # pyright: ignore[reportArgumentType]
        )

        if threshold is None:
            y_pred = np.argmax(y_proba, axis=1)
        else:
            y_pred = (y_proba[:, 1] >= threshold).astype(int)

        return {"y_pred": y_pred, "y_proba": y_proba}
