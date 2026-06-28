"""StratifiedGroupKFold wrapper com exportação JSON de folds.

Garante validação cruzada por grupo (paciente) preservando a proporção
estratificada das classes em cada fold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold as _SKSGKF


class StratifiedGroupKFold:
    """Wrapper reprodutível ao redor do scikit-learn StratifiedGroupKFold.

    Parameters
    ----------
    n_splits : int, default=5
        Número de folds.
    shuffle : bool, default=True
        Embaralha os grupos antes de dividir.
    random_state : int, default=42
        Semente para reprodutibilidade.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self,
        X: Iterable[Any],
        y: Iterable[Any],
        groups: Iterable[Any],
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Retorna gerador de (train_idx, test_idx) para cada fold."""
        sgkf = _SKSGKF(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        return sgkf.split(X=X, y=y, groups=groups)

    def export_json(
        self,
        groups: Iterable[Any],
        y: Iterable[Any],
        output_dir: Path,
    ) -> pd.DataFrame:
        """Exporta índices de treino/teste de cada fold e retorna proporções.

        Parameters
        ----------
        groups : iterable
            Identificador de grupo (paciente) para cada amostra.
        y : iterable
            Rótulo de classe para cada amostra.
        output_dir : Path
            Diretório onde os JSONs serão salvos.

        Returns
        -------
        pd.DataFrame
            DataFrame com colunas ``fold``, ``class`` e ``prop`` representando
            a proporção de cada classe no conjunto de teste do fold.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        y_list = list(y)
        groups_list = list(groups)
        rows: list[dict[str, Any]] = []

        for fold, (train_idx, test_idx) in enumerate(
            self.split(X=[0] * len(y_list), y=y_list, groups=groups_list)
        ):
            path = output_dir / f"fold_{fold}.json"
            path.write_text(
                json.dumps(
                    {"train": train_idx.tolist(), "test": test_idx.tolist()},
                    indent=2,
                ),
                encoding="utf-8",
            )

            test_y = [y_list[i] for i in test_idx]
            n_test = len(test_y)
            for cls in sorted(set(y_list)):
                rows.append(
                    {
                        "fold": fold,
                        "class": cls,
                        "prop": test_y.count(cls) / n_test,
                    }
                )

        return pd.DataFrame(rows)
