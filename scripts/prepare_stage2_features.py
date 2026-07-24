"""Prepara dataset de features morfológicas + time-domain para o Estágio 2.

As features já foram extraídas durante o pré-processamento e estão disponíveis
em `data/features/stage2_multiclass.parquet`. Este script as consolida em um
formato NPZ adequado para treinamento de MLP (S vs V vs F).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Ajustar PYTHONPATH implicitamente quando rodado como script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("prepare_stage2_features")


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
    "qrs_asymmetry_index",
    "t_r_ratio",
    "qrs_raggedness",
]

assert len(FEATURE_COLUMNS) == 16, (
    f"FEATURE_COLUMNS deve ter 16 features, tem {len(FEATURE_COLUMNS)}"
)


def main() -> int:
    parquet_path = Path("data/features/stage2_multiclass.parquet")
    npz_path = Path("data/features/stage2_multiclass.npz")
    output_path = Path("data/features/stage2_multiclass_features.npz")
    scaler_path = Path("data/features/stage2_multiclass_features_scaler.pkl")

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

    # Verifica existência das 16 features e NaN/Inf.
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        LOGGER.error("Features ausentes no parquet: %s", missing)
        return 1

    features_df = df[FEATURE_COLUMNS].copy()
    x_features = features_df.values.astype(np.float32)

    # O treinamento aplica seu próprio StandardScaler por fold; portanto o NPZ
    # deve conter features RAW (não escaladas) para evitar dupla normalização.
    # O scaler abaixo é apenas uma referência de sanity check/backup e não é
    # usado pelo pipeline de treinamento.
    from src.features.scaler_utils import fit_feature_scaler_on_train

    scaler, train_idx, _ = fit_feature_scaler_on_train(x_features, groups, n_splits=5)
    if scaler.n_features_in_ != 16:
        LOGGER.error("Scaler foi ajustado com %d features, esperado 16", scaler.n_features_in_)
        return 1

    # Imputa NaNs usando medianas por classe (quando disponível) ou globais do treino.
    n_nan = features_df.isna().sum().sum()
    if n_nan > 0:
        LOGGER.warning(
            "%d valores NaN encontrados; preenchendo com mediana por classe do treino", n_nan
        )
        y_train_arr = y[train_idx]
        for col in FEATURE_COLUMNS:
            col_median = features_df[col].iloc[train_idx].median()
            for cls in np.unique(y_train_arr):
                cls_median = features_df[col].iloc[train_idx][y_train_arr == cls].median()
                if not np.isnan(cls_median):
                    mask = (features_df[col].isna()) & (y == cls)
                    features_df.loc[mask, col] = features_df.loc[mask, col].fillna(cls_median)
            # Fallback global para valores ainda ausentes
            features_df[col] = features_df[col].fillna(col_median)
        x_features = features_df.values.astype(np.float32)

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    LOGGER.info("Scaler de referência salvo em %s", scaler_path)

    # Sanity check: distribuição de classes
    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    LOGGER.info("Classes: %s", class_counts)
    LOGGER.info("Features shape: %s", x_features.shape)

    # Mapeia record_id (string) para inteiros para compatibilidade com NPZ
    unique_records = sorted(set(groups))
    record_to_idx = {r: i for i, r in enumerate(unique_records)}
    group_ids = np.array([record_to_idx[r] for r in groups], dtype=np.int64)

    np.savez(
        output_path,
        X=x_features,
        y=y,
        groups=group_ids,
    )
    # Salva nomes das features e mapeamento de grupos em JSON.
    (output_path.parent / "stage2_multiclass_features.json").write_text(
        json.dumps(
            {
                "feature_names": FEATURE_COLUMNS,
                "group_mapping": record_to_idx,
                "scaler_path": str(scaler_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    LOGGER.info("Dataset salvo em %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
