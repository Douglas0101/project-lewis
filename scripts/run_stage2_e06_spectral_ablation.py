"""Run E06R-H9: spectral QRS features ablation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "0")

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_stage2_e06_representation_ablation import (  # noqa: E402
    BASE_FEATURE_NAMES,
    E06RunConfig,
    _aggregate_feature_set,
    _class_metrics,
    _column,
    _fit_predict_outer,
    _git_identity,
    _load_data,
    _select_epoch_count,
    _sha256_file,
    _to_int,
)
from src.features.e06_spectral import (  # noqa: E402
    SPECTRAL_QRS_FEATURE_NAMES,
    build_spectral_qrs_schema,
    extract_spectral_qrs_features,
)
from src.models.e06_protocol import build_outer_splits  # noqa: E402

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("run_stage2_e06_spectral_ablation")
FEATURE_SETS = ("baseline16", "h9_spectral")


class H9Verdict(BaseModel):
    """Machine-readable H9 outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis: str
    result: Literal[
        "REPRESENTATION_HYPOTHESIS_SUPPORTED",
        "H9_SPECTRAL_NOT_SUFFICIENT",
    ]
    checkpoint: Literal["PASS", "PASS_HYPOTHESIS_REJECTED"]
    target_f1_f_met: bool
    material_gain_outside_208_213: bool
    macro_gate_met: bool


def _verdict(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    config: E06RunConfig,
) -> H9Verdict:
    target_met = candidate["mean"]["f1_F"] >= config.target_f1_f
    remaining_gain = (
        candidate["mean_remaining_f_groups_f1_F"] - baseline["mean_remaining_f_groups_f1_F"]
    )
    material_gain = remaining_gain >= config.material_gain
    macro_gate = candidate["mean"]["f1_macro"] >= config.minimum_macro_f1
    supported = target_met and material_gain and macro_gate
    return H9Verdict(
        hypothesis=(
            "Spectral QRS features materially improve patient-wise fusion-beat "
            "classification outside records 208/213."
        ),
        result=(
            "REPRESENTATION_HYPOTHESIS_SUPPORTED" if supported else "H9_SPECTRAL_NOT_SUFFICIENT"
        ),
        checkpoint="PASS" if supported else "PASS_HYPOTHESIS_REJECTED",
        target_f1_f_met=target_met,
        material_gain_outside_208_213=material_gain,
        macro_gate_met=macro_gate,
    )


def run(config: E06RunConfig) -> dict[str, Any]:
    """Execute paired baseline/H9 folds."""
    config.output_dir.mkdir(parents=True, exist_ok=False)
    frame, stage2_signals, labels, base_features, dataset = _load_data(
        config.stage2_npz,
        config.stage2_parquet,
    )
    del frame
    spectral = extract_spectral_qrs_features(stage2_signals)
    candidate = np.column_stack([base_features, spectral]).astype(np.float32, copy=False)
    schema = build_spectral_qrs_schema()
    (config.output_dir / "h9_feature_schema.json").write_text(
        schema.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    groups = _column(pd.read_parquet(config.stage2_parquet), "record_id").astype(str).to_numpy()
    outer_splits = build_outer_splits(labels, groups, config.protocol)
    prediction_rows: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold_index, (outer_train, outer_test) in enumerate(outer_splits):
        for seed in config.seeds:
            model_seed = seed + fold_index * 100
            baseline_epoch = _select_epoch_count(
                base_features,
                labels,
                groups,
                outer_train,
                config.protocol,
                config,
                fold_index=fold_index,
                model_seed=model_seed,
            )
            baseline_pred, baseline_prob = _fit_predict_outer(
                base_features,
                labels,
                outer_train,
                outer_test,
                config,
                epochs=baseline_epoch,
                model_seed=model_seed,
            )
            candidate_epoch = _select_epoch_count(
                candidate,
                labels,
                groups,
                outer_train,
                config.protocol,
                config,
                fold_index=fold_index,
                model_seed=model_seed,
            )
            candidate_pred, candidate_prob = _fit_predict_outer(
                candidate,
                labels,
                outer_train,
                outer_test,
                config,
                epochs=candidate_epoch,
                model_seed=model_seed,
            )
            for feature_set, best_epoch, predictions, probabilities in (
                ("baseline16", baseline_epoch, baseline_pred, baseline_prob),
                ("h9_spectral", candidate_epoch, candidate_pred, candidate_prob),
            ):
                metrics = _class_metrics(labels[outer_test], predictions)
                metrics.update(
                    {
                        "seed": _to_int(seed, "seed"),
                        "fold": _to_int(fold_index + 1, "fold"),
                        "feature_set": feature_set,
                        "best_epoch": _to_int(best_epoch, "best epoch"),
                        "n_train": _to_int(outer_train.size, "train sample count"),
                        "n_test": _to_int(outer_test.size, "test sample count"),
                    }
                )
                fold_metrics.append(metrics)
                rows = (
                    pd.read_parquet(config.stage2_parquet)
                    .iloc[outer_test]
                    .loc[:, ["dataset", "record_id", "beat_idx", "r_peak_sample"]]
                    .copy()
                )
                rows["seed"] = _to_int(seed, "seed")
                rows["fold"] = _to_int(fold_index + 1, "fold")
                rows["feature_set"] = feature_set
                rows["y_true"] = labels[outer_test]
                rows["y_pred"] = predictions
                rows["p_S"] = probabilities[:, 0]
                rows["p_V"] = probabilities[:, 1]
                rows["p_F"] = probabilities[:, 2]
                prediction_rows.append(rows)

    predictions_frame = pd.concat(prediction_rows, ignore_index=True)
    predictions_frame.to_parquet(config.output_dir / "predictions.parquet", index=False)
    (config.output_dir / "fold_metrics.json").write_text(
        json.dumps(fold_metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    feature_sets = _column(predictions_frame, "feature_set").astype(str).to_numpy()
    baseline_summary = _aggregate_feature_set(predictions_frame.loc[feature_sets == "baseline16"])
    candidate_summary = _aggregate_feature_set(predictions_frame.loc[feature_sets == "h9_spectral"])
    verdict = _verdict(baseline_summary, candidate_summary, config)
    summary = {
        "baseline16": baseline_summary,
        "h9_spectral": candidate_summary,
        "delta": {
            "mean_f1_F": (candidate_summary["mean"]["f1_F"] - baseline_summary["mean"]["f1_F"]),
            "mean_f1_macro": (
                candidate_summary["mean"]["f1_macro"] - baseline_summary["mean"]["f1_macro"]
            ),
            "remaining_f_groups_f1_F": (
                candidate_summary["mean_remaining_f_groups_f1_F"]
                - baseline_summary["mean_remaining_f_groups_f1_F"]
            ),
            "records_208_213_f1_F": (
                candidate_summary["mean_208_213_f1_F"] - baseline_summary["mean_208_213_f1_F"]
            ),
        },
        "verdict": verdict.model_dump(mode="json"),
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "stage": "E06_REOPENED",
        "hypothesis_id": "H9_SPECTRAL",
        "config": config.model_dump(mode="json"),
        "dataset": dataset.model_dump(mode="json"),
        "source_hashes": {
            "stage2_npz": _sha256_file(config.stage2_npz),
            "stage2_parquet": _sha256_file(config.stage2_parquet),
        },
        "feature_schema_sha256": schema.schema_sha256,
        "base_feature_names": list(BASE_FEATURE_NAMES),
        "candidate_feature_names": list(BASE_FEATURE_NAMES) + list(SPECTRAL_QRS_FEATURE_NAMES),
        "git": _git_identity(),
        "verdict": verdict.model_dump(mode="json"),
    }
    (config.output_dir / "E06R_H9_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage2-npz",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "stage2_multiclass.npz",
    )
    parser.add_argument(
        "--stage2-parquet",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "stage2_multiclass.parquet",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=(42,))
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    random.seed(args.seeds[0])
    np.random.seed(args.seeds[0])
    config = E06RunConfig(
        stage2_npz=args.stage2_npz,
        stage2_parquet=args.stage2_parquet,
        output_dir=args.output_dir,
        seeds=args.seeds,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    summary = run(config)
    LOGGER.info("E06R-H9 verdict: %s", summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
