"""Engenharia de features focada na classe F (E06).

Adiciona features derivadas de HRV/RR e interacoes morfologicas para melhorar
a separabilidade da classe F em cenarios inter-paciente.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("engineer_stage2_features_for_class_f")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_derived_features(X: np.ndarray, base_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Constroi features derivadas a partir do vetor base."""
    df = pd.DataFrame(X, columns=cast(list[str], base_cols))  # type: ignore[arg-type]
    eps = 1e-9

    new_features: list[tuple[str, np.ndarray]] = []

    # Indices conhecidos
    rr_prev = df["rr_prev"].to_numpy()
    rr_next = df["rr_next"].to_numpy()
    rr_ratio = df["rr_ratio"].to_numpy()
    rr_local_mean = df["rr_local_mean"].to_numpy()
    rr_local_std = df["rr_local_std"].to_numpy()
    rmssd = df["rmssd"].to_numpy()
    heart_rate = df["heart_rate"].to_numpy()
    r_amp = df["r_amplitude"].to_numpy()
    q_depth = df["q_depth"].to_numpy()
    t_amp = df["t_amplitude"].to_numpy()
    qrs_width = df["qrs_width_ms"].to_numpy()
    qrs_area = df["qrs_area"].to_numpy()
    st_slope = df["st_slope_mV_s"].to_numpy()
    qrs_ragged = df["qrs_raggedness"].to_numpy()
    t_r_ratio = df["t_r_ratio"].to_numpy()

    # Variabilidade e deltas de RR
    new_features.append(("rr_delta", rr_next - rr_prev))
    new_features.append(("rr_delta_abs", np.abs(rr_next - rr_prev)))
    new_features.append(("rr_local_cv", rr_local_std / (rr_local_mean + eps)))
    new_features.append(("rr_rmssd_ratio", rmssd / (rr_local_mean + eps)))
    new_features.append(("rr_prev_next_ratio", rr_prev / (rr_next + eps)))

    # Entropia aproximada / regularidade de RR
    rr_local_entropy = np.zeros_like(rr_local_mean)
    for i in range(rr_local_mean.shape[0]):
        vals = np.array([rr_prev[i], rr_next[i], rr_local_mean[i], rr_local_std[i]])
        vals = vals[vals > 0]
        if vals.sum() > 0:
            p = vals / vals.sum()
            p = p[p > 0]
            rr_local_entropy[i] = -np.sum(p * np.log(p))
    new_features.append(("rr_local_entropy", rr_local_entropy))

    # Interacoes ritmo x morfologia
    new_features.append(("hr_var", rr_local_std * heart_rate))
    new_features.append(("hr_rmssd", heart_rate * rmssd))
    new_features.append(("qrs_area_per_ms", qrs_area / (qrs_width + eps)))
    new_features.append(("st_t_product", st_slope * t_amp))
    new_features.append(("r_q_ratio", r_amp / (q_depth + eps)))
    new_features.append(("t_q_ratio", t_amp / (q_depth + eps)))
    new_features.append(("qrs_ragged_x_ratio", qrs_ragged * rr_ratio))
    new_features.append(("t_r_x_rr_ratio", t_r_ratio * rr_ratio))

    # Log-transformacoes para features com cauda longa
    new_features.append(("log_qrs_width", np.log1p(np.abs(qrs_width))))
    new_features.append(("log_rr_local_std", np.log1p(np.abs(rr_local_std))))
    new_features.append(("log_rmssd", np.log1p(np.abs(rmssd))))

    new_arrays = np.column_stack([arr for _, arr in new_features])
    new_names = [name for name, _ in new_features]
    return new_arrays, new_names


def _engineer_features(
    input_npz: Path,
    input_json: Path,
    output_dir: Path,
    version_tag: str = "v1",
) -> Path:
    """Carrega features base, adiciona derivadas e salva novo artefato."""
    output_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(input_npz)
    X = np.asarray(npz["X"], dtype=np.float32)
    y = np.asarray(npz["y"], dtype=np.int64)
    groups = np.asarray(npz["groups"])

    try:
        with open(input_json) as f:
            meta = json.load(f)
        base_cols = meta["feature_names"]
    except Exception as exc:
        raise ValueError(f"Falha ao carregar manifesto {input_json}: {exc}") from exc

    X_new, derived_cols = _build_derived_features(X, base_cols)
    X_enhanced = np.hstack([X, X_new])
    all_cols = base_cols + derived_cols

    try:
        out_npz = output_dir / f"stage2_multiclass_features_enhanced_{version_tag}.npz"
        out_json = output_dir / f"stage2_multiclass_features_enhanced_{version_tag}.json"

        np.savez(
            out_npz,
            X=X_enhanced,
            y=y,
            groups=groups,
        )

        meta_out = {
            "version": version_tag,
            "base_features": base_cols,
            "derived_features": derived_cols,
            "feature_names": all_cols,
            "n_features": len(all_cols),
            "n_samples": int(X_enhanced.shape[0]),
            "class_counts": {int(c): int((y == c).sum()) for c in sorted(np.unique(y))},
            "source": str(input_npz),
        }
        with open(out_json, "w") as f:
            json.dump(meta_out, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise ValueError(f"Falha ao salvar features enhanced: {exc}") from exc

    LOGGER.info(
        "Features enhanced: %d base + %d derived = %d total. Saved to %s",
        len(base_cols),
        len(derived_cols),
        len(all_cols),
        out_npz,
    )
    return out_npz


def main() -> int:
    parser = argparse.ArgumentParser(description="Engenharia de features focada na classe F.")
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features.npz",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "stage2_multiclass_features.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "features",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
    )
    args = parser.parse_args()

    try:
        _engineer_features(
            args.input_npz,
            args.input_json,
            args.output_dir,
            args.version,
        )
    except Exception as exc:
        LOGGER.error("Falha na engenharia de features: %s", exc)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
