"""Run E06R-H3: causal multi-beat RR context ablation."""

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
from src.features.e06_context import (  # noqa: E402
    CAUSAL_RR_FEATURE_NAMES,
    build_causal_rr_schema,
    extract_causal_rr_context,
)
from src.models.e06_protocol import build_outer_splits  # noqa: E402

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("run_stage2_e06_context_ablation")


class H3RunConfig(E06RunConfig):
    """Serializable H3 configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    full_parquet: Path


class H3Verdict(BaseModel):
    """Machine-readable H3 outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis: str
    result: Literal[
        "REPRESENTATION_HYPOTHESIS_SUPPORTED",
        "H3_CAUSAL_RR_CONTEXT_NOT_SUFFICIENT",
    ]
    checkpoint: Literal["PASS", "PASS_HYPOTHESIS_REJECTED"]
    target_f1_f_met: bool
    material_gain_outside_208_213: bool
    macro_gate_met: bool


def _verdict(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    config: H3RunConfig,
) -> H3Verdict:
    target_met = candidate["mean"]["f1_F"] >= config.target_f1_f
    remaining_gain = (
        candidate["mean_remaining_f_groups_f1_F"] - baseline["mean_remaining_f_groups_f1_F"]
    )
    material_gain = remaining_gain >= config.material_gain
    macro_gate = candidate["mean"]["f1_macro"] >= config.minimum_macro_f1
    supported = target_met and material_gain and macro_gate
    return H3Verdict(
        hypothesis=(
            "Prior-only multi-beat RR context materially improves patient-wise "
            "fusion-beat classification outside records 208/213."
        ),
        result=(
            "REPRESENTATION_HYPOTHESIS_SUPPORTED"
            if supported
            else "H3_CAUSAL_RR_CONTEXT_NOT_SUFFICIENT"
        ),
        checkpoint="PASS" if supported else "PASS_HYPOTHESIS_REJECTED",
        target_f1_f_met=target_met,
        material_gain_outside_208_213=material_gain,
        macro_gate_met=macro_gate,
    )


def run(config: H3RunConfig) -> dict[str, Any]:
    """Execute paired baseline/H3 folds with identical training protocol."""
    config.output_dir.mkdir(parents=True, exist_ok=False)
    stage2_frame, signals, labels, base_features, dataset = _load_data(
        config.stage2_npz,
        config.stage2_parquet,
    )
    del signals
    full_frame = pd.read_parquet(config.full_parquet)
    context_features = extract_causal_rr_context(full_frame, stage2_frame)
    candidate_features = np.column_stack([base_features, context_features]).astype(
        np.float32,
        copy=False,
    )
    schema = build_causal_rr_schema()
    (config.output_dir / "h3_feature_schema.json").write_text(
        schema.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    groups = _column(stage2_frame, "record_id").astype(str).to_numpy()
    outer_splits = build_outer_splits(labels, groups, config.protocol)
    feature_values = {
        "baseline16": base_features,
        "h3_causal_rr_context": candidate_features,
    }
    prediction_rows: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for seed in config.seeds:
        for fold_index, (outer_train, outer_test) in enumerate(outer_splits):
            for feature_set, values in feature_values.items():
                model_seed = seed + fold_index * 100
                best_epoch = _select_epoch_count(
                    values,
                    labels,
                    groups,
                    outer_train,
                    config.protocol,
                    config,
                    fold_index=fold_index,
                    model_seed=model_seed,
                )
                predictions, probabilities = _fit_predict_outer(
                    values,
                    labels,
                    outer_train,
                    outer_test,
                    config,
                    epochs=best_epoch,
                    model_seed=model_seed,
                )
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
                    stage2_frame.iloc[outer_test]
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
    candidate_summary = _aggregate_feature_set(
        predictions_frame.loc[feature_sets == "h3_causal_rr_context"]
    )
    verdict = _verdict(baseline_summary, candidate_summary, config)
    summary = {
        "baseline16": baseline_summary,
        "h3_causal_rr_context": candidate_summary,
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
        "hypothesis_id": "H3_CAUSAL_MULTI_BEAT_RR_CONTEXT",
        "config": config.model_dump(mode="json"),
        "dataset": dataset.model_dump(mode="json"),
        "source_hashes": {
            "stage2_npz": _sha256_file(config.stage2_npz),
            "stage2_parquet": _sha256_file(config.stage2_parquet),
            "full_parquet": _sha256_file(config.full_parquet),
        },
        "feature_schema_sha256": schema.schema_sha256,
        "base_feature_names": list(BASE_FEATURE_NAMES),
        "candidate_feature_names": list(BASE_FEATURE_NAMES) + list(CAUSAL_RR_FEATURE_NAMES),
        "git": _git_identity(),
        "verdict": verdict.model_dump(mode="json"),
    }
    (config.output_dir / "E06R_H3_manifest.json").write_text(
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
    parser.add_argument(
        "--full-parquet",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "finetuning_mitbih_family.parquet",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=(42,))
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    random.seed(args.seeds[0])
    np.random.seed(args.seeds[0])
    config = H3RunConfig(
        stage2_npz=args.stage2_npz,
        stage2_parquet=args.stage2_parquet,
        full_parquet=args.full_parquet,
        output_dir=args.output_dir,
        seeds=args.seeds,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    summary = run(config)
    LOGGER.info("E06R-H3 verdict: %s", summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
