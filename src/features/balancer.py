"""ECG class balancing — SMOTE / ADASYN in feature space.

Regras mandatórias (ecg-preprocessing-pipeline + Camada-03 spec §3.8):
- SMOTE apenas no espaço de features (nunca no sinal bruto)
- Estratégias: "smote", "smote+rus", "adasyn"
- k_neighbors=5 (SMOTE), n_neighbors=5 (ADASYN)
- Aplicar apenas no treino do fine-tuning
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

LOGGER = logging.getLogger("lewis.camada03.balancer")


class ECGBalancer:
    """Balance dataset using SMOTE/ADASYN in feature space.

    Parameters
    ----------
    strategy : str
        One of "smote", "smote+rus", "adasyn".
    random_state : int
        Random seed.
    """

    def __init__(self, strategy: str = "smote+rus", random_state: int = 42):
        self.strategy = strategy
        self.random_state = random_state

    def balance(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balance features using selected strategy.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix shape (n_samples, n_features).
        y : np.ndarray
            Labels shape (n_samples,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Balanced X, y.

        Raises
        ------
        ImportError
            If imbalanced-learn is not installed.
        ValueError
            If strategy is unknown.
        """
        try:
            from imblearn.over_sampling import ADASYN, SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError as exc:
            raise ImportError(
                "imbalanced-learn is required for ECGBalancer. "
                "Install: pip install imbalanced-learn"
            ) from exc

        if self.strategy == "smote":
            sampler = SMOTE(random_state=self.random_state, k_neighbors=5)
        elif self.strategy == "smote+rus":
            sampler = ImbPipeline(
                [
                    ("over", SMOTE(random_state=self.random_state, k_neighbors=5)),
                    ("under", RandomUnderSampler(random_state=self.random_state)),
                ]
            )
        elif self.strategy == "adasyn":
            sampler = ADASYN(random_state=self.random_state, n_neighbors=5)
        else:
            raise ValueError(f"Estratégia desconhecida: {self.strategy}")

        X_bal, y_bal = sampler.fit_resample(X, y)

        # Log class distribution
        from collections import Counter

        orig_counts = Counter(y)
        bal_counts = Counter(y_bal)
        LOGGER.info(
            "Balanceamento: %s | antes=%s | depois=%s",
            self.strategy,
            dict(orig_counts),
            dict(bal_counts),
        )
        return X_bal, y_bal
