"""Prepara datasets para o pipeline de duas etapas (Project-Lewis v2.0).

A partir do dataset de fine-tuning original (5 classes AAMI: N, S, V, F, Q),
gera:

- ``stage1_binary``: todos os batimentos com label binário
  0 = Normal (N), 1 = Anormal (S, V, F, Q).
- ``stage2_multiclass``: apenas os batimentos S, V, F, remapeados para
  0 = S, 1 = V, 2 = F.

A classe Q é excluída da classificação final, mas ainda é utilizada no Estágio 1
como "Anormal" para evitar que o modelo a confunda com N.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("lewis.camada04.prepare_two_stage")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"

SOURCE_NPZ = FEATURE_DIR / "finetuning_mitbih_family.npz"
SOURCE_PARQUET = FEATURE_DIR / "finetuning_mitbih_family.parquet"

STAGE1_NPZ = FEATURE_DIR / "stage1_binary.npz"
STAGE1_PARQUET = FEATURE_DIR / "stage1_binary.parquet"
STAGE2_NPZ = FEATURE_DIR / "stage2_multiclass.npz"
STAGE2_PARQUET = FEATURE_DIR / "stage2_multiclass.parquet"

# Mapeamento canônico AAMI: int -> string
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]


def _aami_int_to_str(y: np.ndarray) -> np.ndarray:
    return np.array([AAMI_CLASSES[int(v)] for v in y], dtype=object)


def _prepare_stage1(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    npz_path: Path,
    parquet_path: Path,
    exclude_q: bool = False,
    feature_columns: tuple[str, ...] = ("rr_prev", "qrs_width_ms"),
) -> None:
    """Cria dataset binário N vs Anormal.

    Parameters
    ----------
    exclude_q : bool
        Se True, remove amostras da classe Q do treino do Estágio 1.
        A classe Anormal passa a ser formada apenas por S, V e F.
    feature_columns : tuple[str, ...]
        Colunas do DataFrame fonte a serem incluídas como features
        morfológicas/contextuais no Estágio 1.
    """
    if exclude_q:
        keep_mask = y != 4
        X = X[keep_mask]
        y = y[keep_mask]
        df = df.iloc[np.nonzero(keep_mask)[0]].copy()
        stage_name = "stage1_binary_no_q"
    else:
        df = df.copy()
        stage_name = "stage1_binary"

    y_bin = np.where(y == 0, 0, 1).astype(np.int64)

    # Features auxiliares já computadas na Camada 3
    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Feature columns missing in source DataFrame: {missing}")
    features = df[list(feature_columns)].to_numpy(dtype=np.float32)

    np.savez(
        npz_path,
        X=X.astype(np.float32),
        y=y_bin,
        features=features,
        feature_columns=np.array(feature_columns),
    )

    df_out = df.copy()
    df_out["y"] = y_bin
    df_out["stage"] = stage_name
    df_out.to_parquet(parquet_path, index=False)

    n_normal = int((y_bin == 0).sum())
    n_abnormal = int((y_bin == 1).sum())
    LOGGER.info(
        "Stage1 binary saved: n=%d | Normal=%d | Anormal=%d | "
        "features=%s | exclude_q=%s | path=%s",
        len(y_bin),
        n_normal,
        n_abnormal,
        list(feature_columns),
        exclude_q,
        npz_path,
    )


def _prepare_stage2(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    npz_path: Path,
    parquet_path: Path,
) -> None:
    """Cria dataset S/V/F (exclui N e Q)."""
    mask = np.isin(y, [1, 2, 3])
    X_sub = X[mask]
    y_sub = y[mask]
    df_sub = df.iloc[np.nonzero(mask)[0]].copy()

    # Remapear: S=1->0, V=2->1, F=3->2
    remap = {1: 0, 2: 1, 3: 2}
    y_remapped = np.vectorize(remap.get)(y_sub).astype(np.int64)

    np.savez(
        npz_path,
        X=X_sub.astype(np.float32),
        y=y_remapped,
    )

    df_sub["y"] = y_remapped
    df_sub["stage"] = "stage2_multiclass"
    df_sub.to_parquet(parquet_path, index=False)

    counts = {cls: int((y_remapped == i).sum()) for i, cls in enumerate(["S", "V", "F"])}
    LOGGER.info(
        "Stage2 multiclass saved: n=%d | S=%d | V=%d | F=%d | path=%s",
        len(y_remapped),
        counts["S"],
        counts["V"],
        counts["F"],
        npz_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare two-stage datasets.")
    parser.add_argument(
        "--exclude-q-from-stage1",
        action="store_true",
        help="Remove class Q from the Stage 1 binary training set (N vs S/V/F).",
    )
    args = parser.parse_args()

    LOGGER.info("Loading source dataset from %s", SOURCE_NPZ)
    data = np.load(SOURCE_NPZ)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    if X.ndim == 2:
        X = X[..., np.newaxis]

    LOGGER.info("Loading metadata from %s", SOURCE_PARQUET)
    df = pd.read_parquet(SOURCE_PARQUET)

    if len(X) != len(df) or len(y) != len(df):
        raise ValueError(f"Mismatch: X={len(X)}, y={len(y)}, df={len(df)}")

    _prepare_stage1(
        X,
        y,
        df,
        npz_path=STAGE1_NPZ,
        parquet_path=STAGE1_PARQUET,
        exclude_q=args.exclude_q_from_stage1,
    )
    _prepare_stage2(X, y, df, npz_path=STAGE2_NPZ, parquet_path=STAGE2_PARQUET)

    LOGGER.info("Two-stage datasets prepared successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
