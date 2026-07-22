"""Run a protocol-correct E06 representation-only patient-wise ablation.

The script compares the canonical 16 raw Stage-2 features against the same
features plus one stateless direct-QRS morphology family.  Sampling, loss,
architecture, decision rule, folds and seeds remain locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, cast

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "0")

import keras
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.e06_fusion import (  # noqa: E402
    DIRECT_MORPHOLOGY_FEATURE_NAMES,
    E06FeatureSchema,
    build_direct_morphology_schema,
    extract_direct_morphology,
)
from src.features.e06_sampled import (  # noqa: E402
    SampledQRSSchema,
    build_sampled_qrs_schema,
    extract_sampled_qrs_morphology,
)
from src.models.e06_protocol import (  # noqa: E402
    E06EvaluationContract,
    build_outer_splits,
    select_inner_split,
)

BASE_FEATURE_NAMES = (
    "rr_prev",
    "rr_next",
    "rr_ratio",
    "rr_local_mean",
    "rr_local_std",
    "rmssd",
    "heart_rate",
    "r_amplitude",
    "q_depth",
    "t_amplitude",
    "qrs_width_ms",
    "qrs_area",
    "st_slope_mV_s",
    "qrs_asymmetry_index",
    "t_r_ratio",
    "qrs_raggedness",
)
LABEL_TO_INDEX = {"S": 0, "V": 1, "F": 2}
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("run_stage2_e06_representation_ablation")


def _to_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} is not an integer") from error


def _to_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} is not numeric") from error


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    value = frame.loc[:, name]
    if isinstance(value, pd.DataFrame):
        raise ValueError(f"duplicate DataFrame column: {name}")
    return cast(pd.Series, value)


class E06RunConfig(BaseModel):
    """Serializable run configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage2_npz: Path
    stage2_parquet: Path
    output_dir: Path
    seeds: tuple[int, ...] = (42,)
    max_epochs: int = Field(default=30, ge=1)
    patience: int = Field(default=5, ge=0)
    batch_size: int = Field(default=256, ge=1)
    material_gain: float = Field(default=0.05, ge=0.0)
    target_f1_f: float = Field(default=0.50, ge=0.0, le=1.0)
    minimum_macro_f1: float = Field(default=0.45, ge=0.0, le=1.0)
    context_mode: Literal["offline_base16_plus_causal_h1"] = "offline_base16_plus_causal_h1"
    candidate_mode: Literal[
        "append",
        "replace_legacy_morphology",
        "sampled_qrs_append",
    ] = "append"
    protocol: E06EvaluationContract = E06EvaluationContract()


class DatasetBundle(BaseModel):
    """Validated metadata for arrays retained outside Pydantic."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_samples: int
    signal_shape: tuple[int, ...]
    class_counts: dict[int, int]
    n_groups: int
    composite_key_unique: bool
    labels_aligned: bool
    r_peak_metadata_center_mismatch_count: int


class ExperimentVerdict(BaseModel):
    """Machine-readable H1 decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis: str
    result: Literal[
        "REPRESENTATION_HYPOTHESIS_SUPPORTED",
        "H1_DIRECT_MORPHOLOGY_NOT_SUFFICIENT",
        "H4_SAMPLED_QRS_NOT_SUFFICIENT",
    ]
    checkpoint: Literal["PASS", "PASS_HYPOTHESIS_REJECTED"]
    target_f1_f_met: bool
    material_gain_outside_208_213: bool
    macro_gate_met: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_identity() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    return {"head": head, "working_tree_dirty": bool(status.strip())}


def _load_data(
    npz_path: Path,
    parquet_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, DatasetBundle]:
    frame = pd.read_parquet(parquet_path).reset_index(drop=True)
    with np.load(npz_path, allow_pickle=False) as archive:
        signals = np.asarray(archive["X"], dtype=np.float32)
        labels = np.asarray(archive["y"], dtype=np.int64)
    if not (signals.ndim == 2 or (signals.ndim == 3 and signals.shape[-1] == 1)):
        raise ValueError(f"unexpected Stage-2 signal shape: {signals.shape}")
    signal_shape = tuple(_to_int(value, "signal shape dimension") for value in signals.shape)
    if len(frame) != signals.shape[0] or labels.shape[0] != signals.shape[0]:
        raise ValueError("Stage-2 parquet, signals and labels have different lengths")

    missing = [name for name in BASE_FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise ValueError(f"Stage-2 parquet is missing base features: {missing}")
    label_names = _column(frame, "label_aami").astype(str).to_numpy()
    try:
        expected_labels = np.asarray(
            [LABEL_TO_INDEX[label] for label in label_names],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("Stage-2 parquet contains a non S/V/F label") from error
    labels_aligned = np.array_equal(expected_labels, labels)
    if not labels_aligned:
        raise ValueError("Stage-2 parquet labels do not align with the waveform NPZ")

    key_columns = ["dataset", "record_id", "beat_idx"]
    composite_key_unique = not frame.duplicated(key_columns).any()
    if not composite_key_unique:
        raise ValueError("Stage-2 composite beat keys are not unique")
    if not np.isfinite(signals).all():
        raise ValueError("Stage-2 signals contain NaN or Inf")

    base_features = frame.loc[:, list(BASE_FEATURE_NAMES)].to_numpy(dtype=np.float32)
    class_values, class_counts = np.unique(labels, return_counts=True)
    center_index = signals.shape[1] // 2
    metadata_r_peaks = _column(frame, "r_peak_in_segment").to_numpy(dtype=np.int64)
    bundle = DatasetBundle(
        n_samples=_to_int(labels.shape[0], "sample count"),
        signal_shape=signal_shape,
        class_counts={
            _to_int(label, "class label"): _to_int(count, "class count")
            for label, count in zip(class_values, class_counts, strict=True)
        },
        n_groups=_to_int(_column(frame, "record_id").nunique(), "group count"),
        composite_key_unique=composite_key_unique,
        labels_aligned=labels_aligned,
        r_peak_metadata_center_mismatch_count=_to_int(
            np.sum(metadata_r_peaks != center_index),
            "R-peak metadata mismatch count",
        ),
    )
    return frame, signals, labels, base_features, bundle


def _build_model(input_dim: int, seed: int) -> keras.Model:
    keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _fit_preprocessor(
    train_values: np.ndarray,
) -> tuple[SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
        keep_empty_features=True,
    )
    imputed = imputer.fit_transform(train_values)
    scaler = StandardScaler()
    scaler.fit(imputed)
    return imputer, scaler


def _transform(
    values: np.ndarray,
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    transformed = np.asarray(
        scaler.transform(imputer.transform(values)),
        dtype=np.float64,
    )
    if not np.isfinite(transformed).all():
        raise ValueError("preprocessed E06 features contain NaN or Inf")
    return transformed.astype(np.float32, copy=False)


def _select_epoch_count(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    contract: E06EvaluationContract,
    config: E06RunConfig,
    *,
    fold_index: int,
    model_seed: int,
) -> int:
    inner_train, inner_val = select_inner_split(
        outer_train,
        labels,
        groups,
        contract,
        fold_index=fold_index,
    )
    imputer, scaler = _fit_preprocessor(values[inner_train])
    train_values = _transform(values[inner_train], imputer, scaler)
    val_values = _transform(values[inner_val], imputer, scaler)
    model = _build_model(train_values.shape[1], model_seed)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        restore_best_weights=True,
        verbose=0,
    )
    history = model.fit(
        train_values,
        labels[inner_train],
        validation_data=(val_values, labels[inner_val]),
        epochs=config.max_epochs,
        batch_size=config.batch_size,
        callbacks=[early_stopping],
        verbose=0,  # pyright: ignore[reportArgumentType]
    )
    losses = np.asarray(history.history["val_loss"], dtype=np.float64)
    if losses.size == 0 or not np.isfinite(losses).all():
        raise RuntimeError("inner validation did not produce finite losses")
    best_epoch = _to_int(np.argmin(losses), "best epoch index") + 1
    keras.backend.clear_session()
    return best_epoch


def _fit_predict_outer(
    values: np.ndarray,
    labels: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    config: E06RunConfig,
    *,
    epochs: int,
    model_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    imputer, scaler = _fit_preprocessor(values[outer_train])
    train_values = _transform(values[outer_train], imputer, scaler)
    test_values = _transform(values[outer_test], imputer, scaler)
    model = _build_model(train_values.shape[1], model_seed)
    model.fit(
        train_values,
        labels[outer_train],
        epochs=epochs,
        batch_size=config.batch_size,
        verbose=0,  # pyright: ignore[reportArgumentType]
    )
    probabilities = np.asarray(
        model.predict(
            test_values,
            verbose=0,  # pyright: ignore[reportArgumentType]
        )
    )
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    keras.backend.clear_session()
    return predictions, probabilities.astype(np.float32, copy=False)


def _class_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    f1_score_untyped: Any = f1_score
    per_class = np.asarray(
        f1_score_untyped(
            labels,
            predictions,
            labels=[0, 1, 2],
            average=None,
            zero_division=0.0,
        )
    )
    macro = _to_float(
        f1_score_untyped(
            labels,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0.0,
        ),
        "macro F1",
    )
    return {
        "f1_macro": macro,
        "f1_S": _to_float(per_class[0], "F1 S"),
        "f1_V": _to_float(per_class[1], "F1 V"),
        "f1_F": _to_float(per_class[2], "F1 F"),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=[0, 1, 2],
        ).tolist(),
    }


def _f_group_scopes(predictions: pd.DataFrame) -> dict[str, Any]:
    record_ids = _column(predictions, "record_id").astype(str).to_numpy()
    y_true = _column(predictions, "y_true").to_numpy(dtype=np.int64)
    y_pred = _column(predictions, "y_pred").to_numpy(dtype=np.int64)
    f_groups = set(record_ids[y_true == 2].tolist())
    special = {"208", "213"}
    remaining = f_groups - special
    f1_score_untyped: Any = f1_score

    def scope_metrics(group_ids: set[str]) -> dict[str, Any]:
        mask = np.isin(record_ids, sorted(group_ids))
        if not np.any(mask):
            return {"n_samples": 0, "f_support": 0, "f1_F": 0.0}
        per_class = np.asarray(
            f1_score_untyped(
                y_true[mask],
                y_pred[mask],
                labels=[0, 1, 2],
                average=None,
                zero_division=0.0,
            )
        )
        return {
            "n_samples": _to_int(np.sum(mask), "scope sample count"),
            "f_support": _to_int(np.sum(y_true[mask] == 2), "scope F support"),
            "f1_F": _to_float(per_class[2], "scope F1 F"),
        }

    per_group = {group: scope_metrics({group}) for group in sorted(f_groups)}
    return {
        "records_208_213": scope_metrics(f_groups & special),
        "remaining_f_groups": scope_metrics(remaining),
        "per_f_group": per_group,
    }


def _aggregate_feature_set(predictions: pd.DataFrame) -> dict[str, Any]:
    seed_metrics: list[dict[str, Any]] = []
    for seed, seed_frame in predictions.groupby("seed", sort=True):
        metrics = _class_metrics(
            _column(seed_frame, "y_true").to_numpy(dtype=np.int64),
            _column(seed_frame, "y_pred").to_numpy(dtype=np.int64),
        )
        metrics["seed"] = _to_int(seed, "seed")
        metrics["group_scopes"] = _f_group_scopes(seed_frame)
        seed_metrics.append(metrics)

    metric_names = ("f1_macro", "f1_S", "f1_V", "f1_F")
    return {
        "seed_metrics": seed_metrics,
        "mean": {
            name: _to_float(
                np.mean([entry[name] for entry in seed_metrics]),
                f"mean {name}",
            )
            for name in metric_names
        },
        "std": {
            name: _to_float(
                np.std([entry[name] for entry in seed_metrics]),
                f"std {name}",
            )
            for name in metric_names
        },
        "mean_remaining_f_groups_f1_F": _to_float(
            np.mean(
                [entry["group_scopes"]["remaining_f_groups"]["f1_F"] for entry in seed_metrics]
            ),
            "mean remaining-groups F1 F",
        ),
        "mean_208_213_f1_F": _to_float(
            np.mean([entry["group_scopes"]["records_208_213"]["f1_F"] for entry in seed_metrics]),
            "mean 208/213 F1 F",
        ),
    }


def _verdict(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    config: E06RunConfig,
) -> ExperimentVerdict:
    target_met = candidate["mean"]["f1_F"] >= config.target_f1_f
    remaining_gain = (
        candidate["mean_remaining_f_groups_f1_F"] - baseline["mean_remaining_f_groups_f1_F"]
    )
    material_remaining_gain = remaining_gain >= config.material_gain
    macro_gate = candidate["mean"]["f1_macro"] >= config.minimum_macro_f1
    supported = target_met and material_remaining_gain and macro_gate
    rejected_result: Literal[
        "H1_DIRECT_MORPHOLOGY_NOT_SUFFICIENT",
        "H4_SAMPLED_QRS_NOT_SUFFICIENT",
    ]
    if config.candidate_mode == "sampled_qrs_append":
        hypothesis = (
            "Sampled QRS shape materially improves patient-wise fusion-beat "
            "classification outside records 208/213."
        )
        rejected_result = "H4_SAMPLED_QRS_NOT_SUFFICIENT"
    else:
        hypothesis = (
            "Direct QRS morphology materially improves patient-wise fusion-beat "
            "classification outside records 208/213."
        )
        rejected_result = "H1_DIRECT_MORPHOLOGY_NOT_SUFFICIENT"
    experiment_result: Literal[
        "REPRESENTATION_HYPOTHESIS_SUPPORTED",
        "H1_DIRECT_MORPHOLOGY_NOT_SUFFICIENT",
        "H4_SAMPLED_QRS_NOT_SUFFICIENT",
    ] = (
        "REPRESENTATION_HYPOTHESIS_SUPPORTED" if supported else rejected_result
    )
    return ExperimentVerdict(
        hypothesis=hypothesis,
        result=experiment_result,
        checkpoint="PASS" if supported else "PASS_HYPOTHESIS_REJECTED",
        target_f1_f_met=target_met,
        material_gain_outside_208_213=material_remaining_gain,
        macro_gate_met=macro_gate,
    )


def run(config: E06RunConfig) -> dict[str, Any]:
    """Execute paired baseline/H1 folds and persist complete evidence."""
    config.output_dir.mkdir(parents=True, exist_ok=False)
    frame, signals, labels, base_features, dataset = _load_data(
        config.stage2_npz,
        config.stage2_parquet,
    )
    segment_center = signals.shape[1] // 2
    centered_r_peaks = np.full(labels.shape[0], segment_center, dtype=np.int64)
    candidate_base_names: tuple[str, ...]
    candidate_schema: E06FeatureSchema | SampledQRSSchema
    if config.candidate_mode == "sampled_qrs_append":
        engineered_features = extract_sampled_qrs_morphology(signals)
        candidate_schema = build_sampled_qrs_schema()
        engineered_feature_names = tuple(feature.name for feature in candidate_schema.features)
        candidate_name = "h4_sampled_qrs_morphology"
        candidate_base_names = BASE_FEATURE_NAMES
        candidate_features = np.column_stack([base_features, engineered_features])
        hypothesis_id = "H4_SAMPLED_QRS_MORPHOLOGY"
    else:
        engineered_features = extract_direct_morphology(signals, centered_r_peaks)
        candidate_schema = build_direct_morphology_schema()
        engineered_feature_names = DIRECT_MORPHOLOGY_FEATURE_NAMES
        hypothesis_id = "H1_DIRECT_QRS_MORPHOLOGY"
        if config.candidate_mode == "append":
            candidate_name = "h1_direct_morphology_append"
            candidate_base_names = BASE_FEATURE_NAMES
            candidate_features = np.column_stack([base_features, engineered_features])
        else:
            candidate_name = "h1_centered_morphology_replace"
            candidate_base_names = BASE_FEATURE_NAMES[:7]
            candidate_features = np.column_stack([base_features[:, :7], engineered_features])
    candidate_features = candidate_features.astype(np.float32, copy=False)
    feature_values = {
        "baseline16": base_features,
        candidate_name: candidate_features,
    }
    feature_sets = ("baseline16", candidate_name)
    (config.output_dir / "candidate_feature_schema.json").write_text(
        candidate_schema.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    groups = _column(frame, "record_id").astype(str).to_numpy()
    outer_splits = build_outer_splits(labels, groups, config.protocol)
    prediction_rows: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []

    for seed in config.seeds:
        for fold_index, (outer_train, outer_test) in enumerate(outer_splits):
            for feature_set in feature_sets:
                model_seed = seed + fold_index * 100
                values = feature_values[feature_set]
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
                        "train_groups": _to_int(
                            np.unique(groups[outer_train]).size,
                            "train group count",
                        ),
                        "test_groups": _to_int(
                            np.unique(groups[outer_test]).size,
                            "test group count",
                        ),
                    }
                )
                fold_metrics.append(metrics)

                rows = (
                    frame.iloc[outer_test]
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

    feature_set_values = _column(predictions_frame, "feature_set").astype(str).to_numpy()
    baseline_predictions = predictions_frame.loc[feature_set_values == "baseline16"]
    candidate_predictions = predictions_frame.loc[feature_set_values == candidate_name]
    baseline_summary = _aggregate_feature_set(baseline_predictions)
    candidate_summary = _aggregate_feature_set(candidate_predictions)
    verdict = _verdict(baseline_summary, candidate_summary, config)
    summary = {
        "baseline16": baseline_summary,
        candidate_name: candidate_summary,
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
        "hypothesis_id": hypothesis_id,
        "config": config.model_dump(mode="json"),
        "dataset": dataset.model_dump(mode="json"),
        "source_hashes": {
            "stage2_npz": _sha256_file(config.stage2_npz),
            "stage2_parquet": _sha256_file(config.stage2_parquet),
        },
        "feature_schema_sha256": candidate_schema.schema_sha256,
        "base_feature_names": list(BASE_FEATURE_NAMES),
        "candidate_feature_names": list(candidate_base_names) + list(engineered_feature_names),
        "base_future_context_features": [
            "rr_next",
            "rr_ratio",
            "rr_local_mean",
            "rr_local_std",
            "rmssd",
        ],
        "git": _git_identity(),
        "verdict": verdict.model_dump(mode="json"),
    }
    (config.output_dir / f"E06R_{hypothesis_id}_manifest.json").write_text(
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--candidate-mode",
        choices=["append", "replace_legacy_morphology", "sampled_qrs_append"],
        default="append",
    )
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
        candidate_mode=args.candidate_mode,
    )
    summary = run(config)
    LOGGER.info("E06R-H1 verdict: %s", summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
