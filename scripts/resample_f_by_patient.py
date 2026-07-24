"""Reamostragem de batimentos F por paciente (E07).

Aumenta a representacao de records com F, especialmente nao-208/213,
para melhorar a generalizacao inter-paciente.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("resample_f_by_patient")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resample_f_by_patient(
    input_npz: Path,
    output_npz: Path,
    output_json: Path,
    min_per_record: int = 64,
    noise_std: float = 0.01,
    seed: int = 42,
) -> None:
    """Bootstrap de F por record com ruido gaussiano aditivo."""
    npz = np.load(input_npz)
    X = np.asarray(npz["X"], dtype=np.float32)
    y = np.asarray(npz["y"], dtype=np.int64)
    groups = np.asarray(npz["groups"])

    rng = np.random.default_rng(seed)
    f_mask = y == 2

    f_X = X[f_mask]
    f_groups = groups[f_mask]
    f_y = y[f_mask]  # preservado para referencia futura de consistencia
    _ = f_y

    new_X: list[np.ndarray] = []
    new_y: list[np.ndarray] = []
    new_groups: list[np.ndarray] = []

    for g in np.unique(f_groups):
        try:
            g_mask = f_groups == g
            g_X = f_X[g_mask]
            n = g_X.shape[0]
            if n >= min_per_record:
                continue
            n_needed = min_per_record - n
            indices = rng.choice(n, size=n_needed, replace=True)
            sampled = g_X[indices]
            noise = rng.normal(0, noise_std, size=sampled.shape).astype(np.float32)
            new_X.append(sampled + noise)
            new_y.append(np.full(n_needed, 2, dtype=np.int64))
            new_groups.append(np.full(n_needed, g, dtype=f_groups.dtype))
        except Exception as exc:
            raise ValueError(f"Falha ao reamostrar grupo {g}: {exc}") from exc

    if new_X:
        X_aug = np.vstack([X] + new_X)
        y_aug = np.concatenate([y] + new_y)
        groups_aug = np.concatenate([groups] + new_groups)
    else:
        X_aug, y_aug, groups_aug = X, y, groups

    try:
        np.savez(output_npz, X=X_aug, y=y_aug, groups=groups_aug)
    except Exception as exc:
        raise ValueError(f"Falha ao salvar NPZ: {exc}") from exc

    try:
        meta = {
            "n_samples": int(X_aug.shape[0]),
            "n_features": int(X_aug.shape[1]),
            "class_counts": {int(c): int((y_aug == c).sum()) for c in sorted(np.unique(y_aug))},
            "f_augmented_samples": int(sum(len(arr) for arr in new_X)) if new_X else 0,
            "source": str(input_npz),
            "min_per_record": min_per_record,
            "noise_std": noise_std,
        }
    except Exception as exc:
        raise ValueError(f"Falha ao construir metadata: {exc}") from exc

    try:
        with open(output_json, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise ValueError(f"Falha ao salvar JSON: {exc}") from exc

    try:
        LOGGER.info(
            "Resampled F: %d original + %d augmented = %d total",
            int(X.shape[0]),
            meta["f_augmented_samples"],
            meta["n_samples"],
        )
    except Exception as exc:
        raise ValueError(f"Falha no log: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Reamostragem de F por paciente.")
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features.npz",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "features"
        / "stage2_multiclass_features_resampled_e07_v1.npz",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "features"
        / "stage2_multiclass_features_resampled_e07_v1.json",
    )
    parser.add_argument("--min-per-record", type=int, default=64)
    parser.add_argument("--noise-std", type=float, default=0.01)
    args = parser.parse_args()

    try:
        _resample_f_by_patient(
            args.input_npz,
            args.output_npz,
            args.output_json,
            min_per_record=args.min_per_record,
            noise_std=args.noise_std,
        )
    except Exception as exc:
        LOGGER.error("Falha na reamostragem: %s", exc)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
