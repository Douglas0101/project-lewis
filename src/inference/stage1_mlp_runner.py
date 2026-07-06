"""MLPStage1Runner — inferência do Estágio 1 baseada em features.

Este runner substitui a CNN raw-signal pelo MLP treinado sobre features
morfológicas e time-domain, conforme resultado do fallback v2.1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import tensorflow as tf


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
        model_path: Union[Path, str],
        scaler_path: Union[Path, str],
        config_path: Union[Path, str],
    ):
        self.model = tf.keras.models.load_model(str(model_path), compile=False)
        self.scaler = joblib.load(scaler_path)
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.feature_names: List[str] = self.config["feature_names"]

    def predict(
        self,
        features: Dict[str, np.ndarray],
        threshold: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
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

        y_proba = self.model.predict(X, batch_size=1024, verbose=0)

        if threshold is None:
            y_pred = np.argmax(y_proba, axis=1)
        else:
            y_pred = (y_proba[:, 1] >= threshold).astype(int)

        return {"y_pred": y_pred, "y_proba": y_proba}
