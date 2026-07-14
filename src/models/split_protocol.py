"""Protocolos de split versionados para validação cruzada inter-paciente.

Garante que experimentos futuros declarem explicitamente qual splitter usam
e que o split seja auditável por manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold as _GroupKFold

from src.data.stratified_group_split import StratifiedGroupKFold as _StratifiedGroupKFold


class SplitterName(str, Enum):
    """Splitter suportados pela research branch v2.4."""

    GROUP_K_FOLD = "GroupKFold"
    STRATIFIED_GROUP_K_FOLD = "StratifiedGroupKFold"


@dataclass
class SplitConfig:
    """Configuração imutável de split."""

    splitter: SplitterName
    n_splits: int
    shuffle: bool = False
    random_state: int | None = None
    version: str = "v2.4"

    def to_dict(self) -> dict[str, Any]:
        return {
            "splitter": self.splitter.value,
            "n_splits": self.n_splits,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "version": self.version,
        }

    def manifest_hash(self) -> str:
        """Hash canônico da configuração de split."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True).encode()
        ).hexdigest()


class SplitProtocol:
    """Abstração versionada de split inter-paciente.

    Parameters
    ----------
    config : SplitConfig
        Configuração de split.
    """

    def __init__(self, config: SplitConfig) -> None:
        self.config = config
        if config.splitter == SplitterName.GROUP_K_FOLD:
            if config.shuffle:
                raise ValueError("GroupKFold nao suporta shuffle")
            self._splitter = _GroupKFold(n_splits=config.n_splits)
        elif config.splitter == SplitterName.STRATIFIED_GROUP_K_FOLD:
            self._splitter = _StratifiedGroupKFold(
                n_splits=config.n_splits,
                shuffle=config.shuffle,
                random_state=config.random_state or 42,
            )
        else:
            raise ValueError(f"Splitter nao suportado: {config.splitter}")

    def split(
        self,
        X: Iterable[Any],
        y: Iterable[Any],
        groups: Iterable[Any],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Retorna lista de (train_idx, test_idx) para cada fold."""
        X_arr = np.asarray(list(X))
        y_arr = np.asarray(list(y))
        groups_arr = np.asarray(list(groups))
        return list(self._splitter.split(X=X_arr, y=y_arr, groups=groups_arr))

    def export_manifest(
        self,
        X: Iterable[Any],
        y: Iterable[Any],
        groups: Iterable[Any],
        output_path: Path,
    ) -> dict[str, Any]:
        """Exporta split manifest com índices e estatísticas por fold."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        y_arr = np.asarray(list(y))
        groups_arr = np.asarray(list(groups))
        splits = self.split(X=X, y=y, groups=groups)

        folds = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_groups = set(groups_arr[train_idx])
            test_groups = set(groups_arr[test_idx])
            train_y = y_arr[train_idx]
            test_y = y_arr[test_idx]
            try:
                counts_train = {int(c): int((train_y == c).sum()) for c in np.unique(train_y)}
                counts_test = {int(c): int((test_y == c).sum()) for c in np.unique(test_y)}
                train_n = int(len(train_idx))
                test_n = int(len(test_idx))
            except Exception as exc:
                raise ValueError(
                    f"Falha ao computar estatisticas do fold {fold_idx}: {exc}"
                ) from exc
            try:
                train_groups_sorted = [int(g) for g in sorted(train_groups)]
                test_groups_sorted = [int(g) for g in sorted(test_groups)]
            except Exception as exc:
                raise ValueError(f"Falha ao converter group IDs: {exc}") from exc
            folds.append(
                {
                    "fold": fold_idx,
                    "train_idx_hash": hashlib.sha256(train_idx.tobytes()).hexdigest(),
                    "test_idx_hash": hashlib.sha256(test_idx.tobytes()).hexdigest(),
                    "train_groups": train_groups_sorted,
                    "test_groups": test_groups_sorted,
                    "train_counts": counts_train,
                    "test_counts": counts_test,
                    "train_n": train_n,
                    "test_n": test_n,
                    "overlap_groups": sorted(train_groups & test_groups),
                }
            )

        try:
            n_samples = int(len(y_arr))
            n_groups = int(len(np.unique(groups_arr)))
        except Exception as exc:
            raise ValueError(f"Falha ao computar estatisticas globais: {exc}") from exc

        manifest = {
            "split_config": self.config.to_dict(),
            "split_config_hash": self.config.manifest_hash(),
            "n_samples": n_samples,
            "n_groups": n_groups,
            "folds": folds,
        }
        output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest
