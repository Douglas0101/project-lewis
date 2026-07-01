"""Prepara dataset de features morfológicas + time-domain para o Estágio 1.

As features já foram extraídas durante o pré-processamento e estão disponíveis
em `data/features/stage1_binary.parquet`. Este script apenas as consolida em
um formato NPZ adequado para treinamento de MLP.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("prepare_stage1_features")


FEATURE_COLUMNS = [
    # Time-domain (ritmo)
    "rr_prev",
    "rr_next",
    "rr_ratio",
    "rr_local_mean",
    "rr_local_std",
    "rmssd",
    "heart_rate",
    # Morphological (morfologia do complexo QRS/T)
    "r_amplitude",
    "q_depth",
    "t_amplitude",
    "qrs_width_ms",
    "qrs_area",
    "st_slope_mV_s",
]


def main() -> int:
    parquet_path = Path("data/features/stage1_binary.parquet")
    npz_path = Path("data/features/stage1_binary.npz")
    output_path = Path("data/features/stage1_binary_features.npz")

    if not parquet_path.exists():
        LOGGER.error("Parquet não encontrado: %s", parquet_path)
        return 1
    if not npz_path.exists():
        LOGGER.error("NPZ não encontrado: %s", npz_path)
        return 1

    LOGGER.info("Carregando metadados e labels...")
    df = pd.read_parquet(parquet_path)
    npz = np.load(npz_path)
    y = npz["y"]
    groups = df["record_id"].values

    if len(df) != len(y):
        LOGGER.error(
            "Inconsistência: parquet tem %d linhas, npz tem %d",
            len(df),
            len(y),
        )
        return 1

    # Verifica NaN/Inf nas features
    features_df = df[FEATURE_COLUMNS].copy()
    n_nan = features_df.isna().sum().sum()
    if n_nan > 0:
        LOGGER.warning("%d valores NaN encontrados; preenchendo com mediana", n_nan)
        for col in FEATURE_COLUMNS:
            median = features_df[col].median()
            features_df[col] = features_df[col].fillna(median)

    X_features = features_df.values.astype(np.float32)

    # Sanity check: distribuição de classes
    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    LOGGER.info("Classes: %s", class_counts)
    LOGGER.info("Features shape: %s", X_features.shape)

    # Mapeia record_id (string) para inteiros para compatibilidade com NPZ
    # sem object arrays e com GroupKFold.
    unique_records = sorted(set(groups))
    record_to_idx = {r: i for i, r in enumerate(unique_records)}
    group_ids = np.array([record_to_idx[r] for r in groups], dtype=np.int64)

    np.savez(
        output_path,
        X=X_features,
        y=y,
        groups=group_ids,
    )
    # Salva nomes das features e mapeamento de grupos em JSON.
    import json
    (output_path.parent / "stage1_binary_features.json").write_text(
        json.dumps(
            {
                "feature_names": FEATURE_COLUMNS,
                "group_mapping": record_to_idx,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    LOGGER.info("Dataset salvo em %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
