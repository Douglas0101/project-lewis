"""SMOTE sequence-aware para dados de séries temporais 3D.

Preserva a forma (timesteps, channels) ao aplicar SMOTE no espaço
achatado, reconstruindo os tensores após o oversampling.
"""

import numpy as np
from imblearn.over_sampling import SMOTE


class SmoteSequence:
    """Oversampling SMOTE para tensores 3D (amostras, timesteps, channels)."""

    def __init__(self, ratio: dict[int, float], random_state: int = 42):
        """Inicializa com o dicionário de ratios por classe.

        Args:
            ratio: Mapeamento classe -> multiplicador desejado. O número
                final de amostras nunca é menor que a contagem original.
            random_state: Semente para reprodutibilidade do SMOTE.
        """
        self.ratio = ratio
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray):
        """Aplica SMOTE preservando a forma 3D dos dados.

        Args:
            X: Array 3D com shape (n_samples, timesteps, channels).
            y: Array 1D com os rótulos inteiros.

        Returns:
            Tuple (X_res, y_res) com shapes (n_res, timesteps, channels)
            e (n_res,) respectivamente.

        Raises:
            ValueError: Se X não tiver exatamente 3 dimensões.
        """
        if X.ndim != 3:
            raise ValueError(f"Esperado X com 3 dimensões, recebido {X.ndim}")

        n_samples, timesteps, channels = X.shape
        X_2d = X.reshape(n_samples, -1)

        counts = {
            cls: max(int(np.sum(y == cls) * ratio), int(np.sum(y == cls)))
            for cls, ratio in self.ratio.items()
        }

        smote = SMOTE(sampling_strategy=counts, random_state=self.random_state)
        X_res, y_res = smote.fit_resample(X_2d, y)

        return X_res.reshape(-1, timesteps, channels), y_res
