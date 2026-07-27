"""Leakage-safe deterministic training cells and Stage 2 metrics."""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

from src.stage2_research.config import config_hash
from src.stage2_research.contracts import (
    REQUIRED_RUN_ARTIFACTS,
    ExitCode,
    InnerSplitManifest,
    MethodName,
    ProfileName,
    ResearchConfig,
    ResearchError,
    RunManifest,
    SamplerName,
    SplitManifest,
)
from src.stage2_research.data import INDEX_TO_LABEL, Stage2Dataset, frame_column
from src.stage2_research.features import FeatureBundle
from src.stage2_research.integrity import (
    artifact_hashes,
    atomic_write_json,
    atomic_write_text,
    configure_determinism,
    git_identity,
    hash_canonical,
    load_json,
    reset_incomplete_run,
    run_lock,
    sha256_array,
    utc_now,
    validate_descendant_path,
    validate_done_marker,
    validate_path_segment,
    write_done_marker,
)
from src.stage2_research.splits import split_indices
from src.stage2_research.tabular_io import (
    atomic_dataframe_csv,
    atomic_dataframe_parquet,
)
from src.stage2_research.validation import safe_float as _safe_float
from src.stage2_research.validation import safe_int as _safe_int


@dataclass(frozen=True)
class CellResult:
    """Outcome of one candidate/fold/seed cell."""

    run_dir: Path
    status: str
    metrics: dict[str, Any]
    config_hash: str
    resumed: bool


@dataclass(frozen=True)
class SamplingResult:
    """Train-only sampled matrix and provenance."""

    values: np.ndarray
    labels: np.ndarray
    source_indices: np.ndarray
    manifest: dict[str, Any]


@dataclass(frozen=True)
class MethodState:
    """Train-partition-derived loss and callback state."""

    loss: Any
    callbacks: tuple[Any, ...]
    manifest: dict[str, Any]


def _fit_preprocessor(train_values: np.ndarray) -> tuple[SimpleImputer, StandardScaler]:
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
    transformed = scaler.transform(imputer.transform(values))
    output = np.asarray(transformed, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ResearchError("preprocessed values contain NaN/Inf", ExitCode.INVALID_EXPERIMENT)
    return output


def _method_state(
    config: ResearchConfig,
    method: MethodName,
    train_labels: np.ndarray,
    *,
    total_epochs: int,
) -> MethodState:
    """Build a long-tail loss exclusively from one train partition."""
    import keras
    import tensorflow as tf

    counts = np.bincount(train_labels.astype(np.int64), minlength=3).astype(np.float64)
    if np.any(counts <= 0.0):
        raise ResearchError(
            f"{method} requires all S/V/F classes in train",
            ExitCode.INVALID_EXPERIMENT,
        )
    priors = counts / np.sum(counts)
    base_loss = keras.losses.SparseCategoricalCrossentropy()
    manifest: dict[str, Any] = {
        "method": method,
        "class_counts": counts.astype(np.int64).tolist(),
        "class_priors": priors.tolist(),
        "fit_scope": "train_partition_only",
    }
    callbacks: tuple[Any, ...] = ()
    if method in {"ce_control", "crt_patient_aware"}:
        return MethodState(loss=base_loss, callbacks=callbacks, manifest=manifest)

    def probability_logits(y_pred: Any) -> Any:
        values = tf.cast(y_pred, tf.float32)
        return tf.math.log(tf.clip_by_value(values, 1.0e-7, 1.0))

    if method == "logit_adjustment":
        adjustment = config.e08.logit_adjustment_tau * np.log(priors)
        adjustment_tensor = tf.constant(adjustment, dtype=tf.float32)
        sparse_logits_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        def logit_adjustment_loss(y_true: Any, y_pred: Any) -> Any:
            return sparse_logits_loss(y_true, probability_logits(y_pred) + adjustment_tensor)

        logit_adjustment_loss.__name__ = "logit_adjustment_loss"
        manifest.update(
            {
                "tau": config.e08.logit_adjustment_tau,
                "adjustment_vector": adjustment.tolist(),
            }
        )
        return MethodState(
            loss=logit_adjustment_loss,
            callbacks=callbacks,
            manifest=manifest,
        )
    if method == "balanced_softmax":
        adjustment = np.log(counts)
        count_tensor = tf.constant(adjustment, dtype=tf.float32)
        sparse_logits_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        def balanced_softmax_loss(y_true: Any, y_pred: Any) -> Any:
            return sparse_logits_loss(y_true, probability_logits(y_pred) + count_tensor)

        balanced_softmax_loss.__name__ = "balanced_softmax_loss"
        manifest["adjustment_vector"] = adjustment.tolist()
        return MethodState(
            loss=balanced_softmax_loss,
            callbacks=callbacks,
            manifest=manifest,
        )
    if method == "focal_legacy":
        alpha = np.asarray(config.e08.focal_alpha, dtype=np.float32)
        legacy_weight = np.asarray(config.e08.focal_class_weight, dtype=np.float32)
        effective_alpha = alpha * legacy_weight
        alpha_tensor = tf.constant(effective_alpha, dtype=tf.float32)
        gamma = config.e08.focal_gamma

        def focal_legacy_loss(y_true: Any, y_pred: Any) -> Any:
            labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            probabilities = tf.cast(y_pred, tf.float32)
            indices = tf.stack([tf.range(tf.shape(labels)[0]), labels], axis=1)
            true_probability = tf.gather_nd(probabilities, indices)
            alpha_value = tf.gather(alpha_tensor, labels)
            return (
                -alpha_value
                * tf.pow(1.0 - true_probability, gamma)
                * tf.math.log(tf.clip_by_value(true_probability, 1.0e-7, 1.0))
            )

        focal_legacy_loss.__name__ = "focal_legacy_loss"
        manifest.update(
            {
                "alpha": list(config.e08.focal_alpha),
                "gamma": gamma,
                "class_weight": list(config.e08.focal_class_weight),
                "effective_alpha": effective_alpha.tolist(),
                "legacy_parameters_frozen": True,
            }
        )
        return MethodState(
            loss=focal_legacy_loss,
            callbacks=callbacks,
            manifest=manifest,
        )
    if method == "ldam_drw":
        margins = config.e08.ldam_max_margin / np.power(counts, 0.25)
        margins *= config.e08.ldam_max_margin / np.max(margins)
        inverse = 1.0 / counts
        class_weights = inverse / np.mean(inverse)
        margins_tensor = tf.constant(margins, dtype=tf.float32)
        weights_tensor = tf.constant(class_weights, dtype=tf.float32)
        drw_active = tf.Variable(
            initial_value=tf.constant(False, dtype=tf.bool),
            trainable=False,
            dtype=tf.bool,
        )
        activation_epoch = max(
            1,
            _safe_int(
                np.ceil(total_epochs * config.e08.ldam_drw_epoch_fraction),
                "LDAM DRW activation epoch",
            ),
        )

        def ldam_drw_loss(y_true: Any, y_pred: Any) -> Any:
            labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            logits = probability_logits(y_pred)
            one_hot = tf.one_hot(labels, depth=3, dtype=tf.float32)
            adjusted = logits - one_hot * tf.gather(margins_tensor, labels)[:, None]
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=adjusted,
            )
            active_weights = tf.where(
                drw_active,
                tf.gather(weights_tensor, labels),
                tf.ones_like(losses),
            )
            return losses * active_weights

        ldam_drw_loss.__name__ = "ldam_drw_loss"

        class ActivateDRW(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
                del logs
                drw_active.assign(epoch + 1 >= activation_epoch)

        callbacks = (ActivateDRW(),)
        manifest.update(
            {
                "margins": margins.tolist(),
                "drw_activation_epoch": activation_epoch,
                "weights_before": [1.0, 1.0, 1.0],
                "weights_after": class_weights.tolist(),
            }
        )
        return MethodState(loss=ldam_drw_loss, callbacks=callbacks, manifest=manifest)
    raise ResearchError(f"unknown E08 method: {method}", ExitCode.ARGUMENT_ERROR)


def _build_softmax_model(input_dim: int, seed: int, *, loss: Any = None) -> Any:
    import keras

    keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu", name="encoder_dense"),
            keras.layers.Dropout(0.3, name="encoder_dropout"),
            keras.layers.Dense(3, activation="softmax", name="classifier"),
        ],
        name="stage2_mlp_128",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss or "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _encoder_hash(model: Any) -> str:
    """Hash the trainable encoder weights, excluding the classifier head."""
    weights = model.get_layer("encoder_dense").get_weights()
    return hash_canonical([sha256_array(np.asarray(value)) for value in weights])


def _prepare_crt_head(model: Any, seed: int) -> None:
    """Freeze encoder and deterministically reinitialize only the classifier."""
    import keras

    encoder = model.get_layer("encoder_dense")
    dropout = model.get_layer("encoder_dropout")
    classifier = model.get_layer("classifier")
    encoder.trainable = False
    dropout.trainable = False
    dropout.rate = 0.0
    classifier.trainable = True
    initializer = keras.initializers.GlorotUniform(seed=seed)
    initialized_kernel = initializer(
        shape=classifier.kernel.shape,
        dtype=classifier.kernel.dtype,
    )
    classifier.kernel.assign(initialized_kernel.numpy())
    classifier.bias.assign(np.zeros(tuple(classifier.bias.shape), dtype=np.float32))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(labels, return_counts=True)
    result = {"S": 0, "V": 0, "F": 0}
    for label, count in zip(unique, counts, strict=True):
        result[INDEX_TO_LABEL[_safe_int(label, "class label")]] = _safe_int(
            count,
            "class count",
        )
    return result


def _patient_sample_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    sqrt_weighted: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes, class_counts = np.unique(labels, return_counts=True)
    target_per_class = _safe_int(np.max(class_counts), "patient sampling target")
    sampled: list[np.ndarray] = []
    for class_value in sorted(classes.tolist()):
        class_indices = np.flatnonzero(labels == class_value)
        class_groups = groups[class_indices].astype(str)
        unique_groups, group_counts = np.unique(class_groups, return_counts=True)
        if unique_groups.size == 0:
            raise ResearchError("patient sampler found an empty class", ExitCode.INVALID_EXPERIMENT)
        if sqrt_weighted:
            probabilities = np.sqrt(group_counts.astype(np.float64))
            probabilities /= np.sum(probabilities)
        else:
            probabilities = np.full(unique_groups.size, 1.0 / unique_groups.size)
        chosen_groups = rng.choice(
            unique_groups,
            size=target_per_class,
            replace=True,
            p=probabilities,
        )
        class_samples = np.empty(target_per_class, dtype=np.int64)
        for position, group in enumerate(chosen_groups):
            group_candidates = class_indices[class_groups == group]
            class_samples[position] = rng.choice(group_candidates)
        sampled.append(class_samples)
    combined = np.concatenate(sampled)
    return combined[rng.permutation(combined.size)]


def _target_f_count(labels: np.ndarray, target_fraction: float = 0.125) -> int:
    if not 0.0 < target_fraction < 1.0:
        raise ResearchError("F target fraction is invalid", ExitCode.INVALID_EXPERIMENT)
    current_f = _safe_int(np.sum(labels == 2), "current F support")
    non_f = _safe_int(np.sum(labels != 2), "non-F support")
    target = _safe_int(
        np.ceil(target_fraction * non_f / (1.0 - target_fraction)),
        "target F support",
    )
    return max(current_f, target)


def _patient_targeted_f_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    sqrt_weighted: bool,
    target_fraction: float = 0.125,
    cap_multiplier: float = 2.0,
) -> tuple[np.ndarray, int, dict[str, int]]:
    rng = np.random.default_rng(seed)
    f_indices = np.flatnonzero(labels == 2)
    non_f_indices = np.flatnonzero(labels != 2)
    if f_indices.size == 0:
        raise ResearchError("patient F sampler found no F class", ExitCode.INVALID_EXPERIMENT)
    f_groups = groups[f_indices].astype(str)
    unique_groups, group_counts = np.unique(f_groups, return_counts=True)
    target_f = _target_f_count(labels, target_fraction)
    cap = _safe_int(
        np.ceil(cap_multiplier * target_f / unique_groups.size),
        "patient F sample cap",
    )
    used = np.zeros(unique_groups.size, dtype=np.int64)
    selected = np.empty(target_f, dtype=np.int64)
    base_weights = (
        np.sqrt(group_counts.astype(np.float64))
        if sqrt_weighted
        else np.ones(unique_groups.size, dtype=np.float64)
    )
    for position in range(target_f):
        eligible = used < cap
        if not np.any(eligible):
            raise ResearchError("patient F sample cap exhausted", ExitCode.INVALID_EXPERIMENT)
        probabilities = np.where(eligible, base_weights, 0.0)
        probabilities /= np.sum(probabilities)
        group_index = _safe_int(
            rng.choice(unique_groups.size, p=probabilities),
            "sampled patient index",
        )
        candidates = f_indices[f_groups == unique_groups[group_index]]
        selected[position] = rng.choice(candidates)
        used[group_index] += 1
    combined = np.concatenate((non_f_indices, selected))
    contributions = {
        str(group): _safe_int(count, "patient F contribution")
        for group, count in zip(unique_groups, used, strict=True)
    }
    return combined[rng.permutation(combined.size)], cap, contributions


def sample_training_values(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    source_indices: np.ndarray,
    *,
    sampler: SamplerName,
    seed: int,
    smote_k_neighbors: int,
) -> SamplingResult:
    """Apply one sampler only to an already isolated train partition."""
    source = np.asarray(source_indices, dtype=np.int64)
    before_counts = _class_counts(labels)
    synthetic_count = 0
    patient_cap: int | None = None
    patient_contributions: dict[str, int] = {}
    target_f_fraction: float | None = None
    if sampler in {"natural", "pd_s0_natural", "pd_s4_focal_gentle"}:
        sampled_values = values
        sampled_labels = labels
        sampled_source = source
    elif sampler == "random_oversampling":
        oversampler = RandomOverSampler(random_state=seed)
        oversample_untyped: Any = oversampler.fit_resample
        sampled_values_raw, sampled_labels_raw = cast(
            tuple[Any, Any],
            oversample_untyped(values, labels),
        )
        sampled_values = np.asarray(sampled_values_raw, dtype=np.float32)
        sampled_labels = np.asarray(sampled_labels_raw, dtype=np.int64)
        sample_indices = np.asarray(oversampler.sample_indices_, dtype=np.int64)
        sampled_source = source[sample_indices]
    elif sampler == "pd_s1_f_target":
        target_f_fraction = 0.125
        target_f = _target_f_count(labels, target_f_fraction)
        current_f = _safe_int(np.sum(labels == 2), "current F support")
        if target_f == current_f:
            sampled_values, sampled_labels, sampled_source = values, labels, source
        else:
            oversampler_type: Any = RandomOverSampler
            oversampler = oversampler_type(
                sampling_strategy={2: target_f},
                random_state=seed,
            )
            oversample_untyped = oversampler.fit_resample
            sampled_values_raw, sampled_labels_raw = cast(
                tuple[Any, Any],
                oversample_untyped(values, labels),
            )
            sampled_values = np.asarray(sampled_values_raw, dtype=np.float32)
            sampled_labels = np.asarray(sampled_labels_raw, dtype=np.int64)
            sample_indices = np.asarray(oversampler.sample_indices_, dtype=np.int64)
            sampled_source = source[sample_indices]
    elif sampler in {"patient_uniform", "patient_sqrt"}:
        local_indices = _patient_sample_indices(
            labels,
            groups,
            seed=seed,
            sqrt_weighted=sampler == "patient_sqrt",
        )
        sampled_values = values[local_indices]
        sampled_labels = labels[local_indices]
        sampled_source = source[local_indices]
    elif sampler in {
        "pd_s2_patient_uniform_capped",
        "pd_s3_patient_sqrt_capped",
    }:
        target_f_fraction = 0.125
        local_indices, patient_cap, patient_contributions = _patient_targeted_f_indices(
            labels,
            groups,
            seed=seed,
            sqrt_weighted=sampler == "pd_s3_patient_sqrt_capped",
            target_fraction=target_f_fraction,
            cap_multiplier=2.0,
        )
        sampled_values = values[local_indices]
        sampled_labels = labels[local_indices]
        sampled_source = source[local_indices]
    elif sampler in {"smote", "pd_s5_smote_feature"}:
        if sampler == "pd_s5_smote_feature":
            target_f_fraction = 0.125
            target_f = _target_f_count(labels, target_f_fraction)
            strategy: str | dict[int, int] = {2: target_f}
            minimum_support = _safe_int(np.sum(labels == 2), "SMOTE F support")
        else:
            strategy = "auto"
            minimum_support = _safe_int(
                np.min(np.unique(labels, return_counts=True)[1]),
                "SMOTE minimum support",
            )
        neighbors = min(smote_k_neighbors, minimum_support - 1)
        if neighbors < 1:
            raise ResearchError("SMOTE class support is insufficient", ExitCode.INVALID_EXPERIMENT)
        smote_type: Any = SMOTE
        smote = smote_type(
            sampling_strategy=strategy,
            random_state=seed,
            k_neighbors=neighbors,
        )
        smote_untyped: Any = smote.fit_resample
        sampled_values_raw, sampled_labels_raw = cast(
            tuple[Any, Any],
            smote_untyped(values, labels),
        )
        sampled_values = np.asarray(sampled_values_raw, dtype=np.float32)
        sampled_labels = np.asarray(sampled_labels_raw, dtype=np.int64)
        synthetic_count = sampled_labels.size - labels.size
        sampled_source = np.concatenate([source, np.full(synthetic_count, -1, dtype=np.int64)])
    else:
        raise ResearchError(f"unknown sampler: {sampler}", ExitCode.ARGUMENT_ERROR)
    if not np.isfinite(sampled_values).all():
        raise ResearchError("sampler produced NaN/Inf", ExitCode.INVALID_EXPERIMENT)
    observed_sources = set(sampled_source[sampled_source >= 0].tolist())
    allowed_sources = set(source.tolist())
    source_outside_partition_count = len(observed_sources - allowed_sources)
    if source_outside_partition_count:
        raise ResearchError("sampler crossed the train partition", ExitCode.LEAKAGE)
    f_fraction = _safe_float(
        np.mean(sampled_labels == 2),
        "sampled F fraction",
    )
    manifest = {
        "sampler": sampler,
        "random_state": seed,
        "before_counts": before_counts,
        "after_counts": _class_counts(sampled_labels),
        "input_count": _safe_int(labels.size, "sampler input count"),
        "output_count": _safe_int(sampled_labels.size, "sampler output count"),
        "synthetic_count": _safe_int(synthetic_count, "synthetic sample count"),
        "validation_or_test_sampled": False,
        "sampler_scope": "TRAIN_ONLY_FEATURE_SPACE",
        "target_f_fraction": target_f_fraction,
        "realized_f_fraction": f_fraction,
        "patient_cap": patient_cap,
        "patient_f_contributions": patient_contributions,
        "input_partition_index_hash": hash_canonical(source.tolist()),
        "source_outside_partition_count": source_outside_partition_count,
        "source_index_hash": hash_canonical(sampled_source.tolist()),
    }
    return SamplingResult(
        values=sampled_values,
        labels=sampled_labels,
        source_indices=sampled_source,
        manifest=manifest,
    )


def _history_rows(history: Any, phase: str) -> list[dict[str, Any]]:
    keys = sorted(history.history)
    n_epochs = max((len(history.history[key]) for key in keys), default=0)
    rows: list[dict[str, Any]] = []
    for epoch in range(n_epochs):
        row: dict[str, Any] = {"phase": phase, "epoch": epoch + 1}
        for key in keys:
            values = history.history[key]
            row[key] = _safe_float(values[epoch], f"history {key}") if epoch < len(values) else ""
        rows.append(row)
    return rows


def _history_csv(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "phase,epoch\n"
    fieldnames = sorted(
        {key for row in rows for key in row},
        key=lambda item: (item != "phase", item != "epoch", item),
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _per_group_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, group_frame in predictions.groupby("record_id", sort=True):
        y_true = frame_column(group_frame, "y_true").to_numpy(dtype=np.int64)
        y_pred = frame_column(group_frame, "y_pred").to_numpy(dtype=np.int64)
        f1_untyped: Any = f1_score
        f1_f = _safe_float(
            f1_untyped(y_true, y_pred, labels=[2], average="macro", zero_division=0.0),
            "group F1 F",
        )
        rows.append(
            {
                "record_id": str(group),
                "n_samples": _safe_int(y_true.size, "group sample count"),
                "F_support": _safe_int(np.sum(y_true == 2), "group F support"),
                "F_predicted": _safe_int(np.sum(y_pred == 2), "group F predicted"),
                "F1_F": f1_f,
            }
        )
    return pd.DataFrame(rows)


def _scope_f1(predictions: pd.DataFrame, scope: str) -> dict[str, Any]:
    records = frame_column(predictions, "record_id").astype(str).to_numpy()
    y_true = frame_column(predictions, "y_true").to_numpy(dtype=np.int64)
    y_pred = frame_column(predictions, "y_pred").to_numpy(dtype=np.int64)
    if scope == "208":
        mask = records == "208"
    elif scope == "213":
        mask = records == "213"
    elif scope == "outside_208_213":
        mask = ~np.isin(records, ["208", "213"])
    else:
        raise ResearchError(f"unknown metric scope: {scope}", ExitCode.EVALUATION_FAILURE)
    if not np.any(mask):
        return {"n_samples": 0, "F_support": 0, "F1_F": 0.0}
    f1_untyped: Any = f1_score
    value = f1_untyped(
        y_true[mask],
        y_pred[mask],
        labels=[2],
        average="macro",
        zero_division=0.0,
    )
    return {
        "n_samples": _safe_int(np.sum(mask), "scope sample count"),
        "F_support": _safe_int(np.sum(y_true[mask] == 2), "scope F support"),
        "F1_F": _safe_float(value, "scope F1 F"),
    }


def compute_metrics(predictions: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute fixed-order S/V/F, calibration, and group-scope metrics."""
    y_true = frame_column(predictions, "y_true").to_numpy(dtype=np.int64)
    y_pred = frame_column(predictions, "y_pred").to_numpy(dtype=np.int64)
    p_f = frame_column(predictions, "p_F").to_numpy(dtype=np.float64)
    f1_untyped: Any = f1_score
    per_class = np.asarray(
        f1_untyped(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0.0),
        dtype=np.float64,
    )
    macro = f1_untyped(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0.0,
    )
    precision_recall_untyped: Any = precision_recall_fscore_support
    precision_raw, recall_raw, _, support_raw = precision_recall_untyped(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    precision = np.asarray(precision_raw, dtype=np.float64)
    recall = np.asarray(recall_raw, dtype=np.float64)
    support = np.asarray(support_raw, dtype=np.int64)
    ap_f = (
        average_precision_score((y_true == 2).astype(np.int64), p_f) if np.any(y_true == 2) else 0.0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    group_frame = _per_group_metrics(predictions)
    metrics = {
        "F1_S": _safe_float(per_class[0], "F1 S"),
        "F1_V": _safe_float(per_class[1], "F1 V"),
        "F1_F": _safe_float(per_class[2], "F1 F"),
        "macro_F1": _safe_float(macro, "macro F1"),
        "precision_F": _safe_float(precision[2], "precision F"),
        "recall_F": _safe_float(recall[2], "recall F"),
        "AP_F": _safe_float(ap_f, "AP F"),
        "F_support": _safe_int(support[2], "F support"),
        "predicted_class_counts": _class_counts(y_pred),
        "confusion_matrix": matrix.tolist(),
        "scopes": {
            "record_208": _scope_f1(predictions, "208"),
            "record_213": _scope_f1(predictions, "213"),
            "outside_208_213": _scope_f1(predictions, "outside_208_213"),
        },
    }
    return metrics, group_frame


def _predictions_frame(
    dataset: Stage2Dataset,
    outer_test: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    feature_bundle: FeatureBundle,
) -> pd.DataFrame:
    columns = [
        "sample_id",
        "waveform_sha256",
        "dataset",
        "record_id",
        "patient_id",
        "beat_idx",
        "r_peak_sample",
    ]
    available = [name for name in columns if name in dataset.frame]
    frame = dataset.frame.iloc[outer_test].loc[:, available].copy()
    frame["y_true"] = dataset.labels[outer_test]
    frame["y_pred"] = predictions
    frame["p_S"] = probabilities[:, 0]
    frame["p_V"] = probabilities[:, 1]
    frame["p_F"] = probabilities[:, 2]
    log_probabilities = np.log(np.clip(probabilities.astype(np.float64), 1.0e-12, 1.0))
    frame["logit_S"] = log_probabilities[:, 0]
    frame["logit_V"] = log_probabilities[:, 1]
    frame["logit_F"] = log_probabilities[:, 2]
    frame["margin_F"] = log_probabilities[:, 2] - np.maximum(
        log_probabilities[:, 0],
        log_probabilities[:, 1],
    )
    for name in ("template_distance_F", "template_distance_V", "template_corr_F"):
        if name in feature_bundle.feature_names:
            index = feature_bundle.feature_names.index(name)
            frame[name] = feature_bundle.outer_values[outer_test, index]
    return frame


def _model_probabilities(model: Any, values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict(values, verbose=0), dtype=np.float32)
    if probabilities.shape != (values.shape[0], 3) or not np.isfinite(probabilities).all():
        raise ResearchError("invalid model probabilities", ExitCode.EVALUATION_FAILURE)
    return probabilities


def _cell_config_payload(
    config: ResearchConfig,
    *,
    stage: str,
    experiment_id: str,
    candidate: str,
    representation: str,
    fold: int,
    seed: int,
    profile_name: ProfileName,
    deterministic: bool,
    device: str,
    sampler: SamplerName,
    method: MethodName,
    preflight_hash: str,
    source_manifest_hash: str,
    runtime_identity_hash: str,
    feature_manifest_hash: str,
    split_manifest_hash: str,
) -> dict[str, Any]:
    profile = config.profiles[profile_name]
    return {
        "research_config_hash": config_hash(config),
        "stage": stage,
        "experiment_id": experiment_id,
        "candidate": candidate,
        "representation": representation,
        "fold": fold,
        "seed": seed,
        "profile": profile_name,
        "profile_config": profile.model_dump(mode="json"),
        "deterministic": deterministic,
        "device": device,
        "sampler": sampler,
        "method": method,
        "preflight_hash": preflight_hash,
        "source_manifest_hash": source_manifest_hash,
        "runtime_identity_hash": runtime_identity_hash,
        "feature_manifest_hash": feature_manifest_hash,
        "split_manifest_hash": split_manifest_hash,
        "protocol": config.split_contract.model_dump(mode="json"),
    }


def stage_run_dir(
    config: ResearchConfig,
    *,
    stage: str,
    experiment_id: str,
    candidate: str,
    fold: int,
    seed: int,
) -> Path:
    """Resolve one cell directory without touching production artifacts."""
    stage_dir = {
        "e06.5": "E06_5",
        "e06.5-pd": "E06_5_PD",
        "e07": "E07",
        "e07-pd": "E07_PD",
        "e08": "E08",
    }.get(stage)
    if stage_dir is None:
        raise ResearchError(f"unknown run stage: {stage}", ExitCode.ARGUMENT_ERROR)
    safe_experiment_id = validate_path_segment(experiment_id, label="experiment-id")
    safe_candidate = validate_path_segment(candidate, label="candidate")
    candidate_path = (
        config.output_root
        / stage_dir
        / safe_experiment_id
        / safe_candidate
        / f"fold_{fold}"
        / f"seed_{seed}"
    )
    return validate_descendant_path(
        config.output_root,
        candidate_path,
        label="stage run directory",
    )


def _existing_cell_result(run_dir: Path, cell_hash: str) -> CellResult:
    metrics = cast(dict[str, Any], load_json(run_dir / "metrics.json"))
    return CellResult(
        run_dir=run_dir,
        status="SKIPPED_DONE",
        metrics=metrics,
        config_hash=cell_hash,
        resumed=False,
    )


def train_e06_cell(
    config: ResearchConfig,
    dataset: Stage2Dataset,
    outer_manifest: SplitManifest,
    inner_manifest: InnerSplitManifest,
    feature_bundle: FeatureBundle,
    *,
    candidate: str,
    fold: int,
    seed: int,
    profile_name: ProfileName,
    experiment_id: str,
    deterministic: bool,
    device: str,
    sampler: SamplerName = "natural",
    method: MethodName = "ce_control",
    stage: str = "e06.5",
    run_name: str | None = None,
    representation_name: str | None = None,
    preflight_hash: str,
    source_manifest_hash: str,
    runtime_identity_hash: str,
    resume: bool = True,
    force: bool = False,
) -> CellResult:
    """Train, serialize, reload, evaluate, and complete one CE research cell."""
    cell_name = run_name or candidate
    representation = representation_name or candidate
    identity_hashes = {
        "preflight_hash": preflight_hash,
        "source_manifest_hash": source_manifest_hash,
        "runtime_identity_hash": runtime_identity_hash,
    }
    if any(len(value) != 64 for value in identity_hashes.values()):
        raise ResearchError(
            "training cell requires complete preflight/source/runtime identity",
            ExitCode.BLOCKED_PRECONDITION,
        )
    if stage in {"e06.5", "e06.5-pd"} and sampler != "natural":
        raise ResearchError("E06.5 sampling must remain natural", ExitCode.INVALID_EXPERIMENT)
    profile = config.profiles[profile_name]
    if profile_name == "audit" and (not deterministic or profile.max_parallel != 1):
        raise ResearchError(
            "audit execution requires deterministic serial execution", ExitCode.INVALID_EXPERIMENT
        )
    outer_train, outer_test, inner_train, inner_val = split_indices(
        outer_manifest,
        inner_manifest,
        fold,
    )
    cell_payload = _cell_config_payload(
        config,
        stage=stage,
        experiment_id=experiment_id,
        candidate=cell_name,
        representation=representation,
        fold=fold,
        seed=seed,
        profile_name=profile_name,
        deterministic=deterministic,
        device=device,
        sampler=sampler,
        method=method,
        preflight_hash=preflight_hash,
        source_manifest_hash=source_manifest_hash,
        runtime_identity_hash=runtime_identity_hash,
        feature_manifest_hash=feature_bundle.fold_manifest_hash,
        split_manifest_hash=outer_manifest.manifest_hash,
    )
    cell_hash = hash_canonical(cell_payload)
    run_dir = stage_run_dir(
        config,
        stage=stage,
        experiment_id=experiment_id,
        candidate=cell_name,
        fold=fold,
        seed=seed,
    )
    if force and (run_dir / "DONE").exists():
        raise ResearchError(
            "--force cannot overwrite a finalized run; use a new experiment-id",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    marker = (
        validate_done_marker(run_dir, expected_config_hash=cell_hash) if run_dir.exists() else None
    )
    if marker is not None:
        if resume:
            return _existing_cell_result(run_dir, cell_hash)
        raise ResearchError(
            "completed run exists; use --resume or a new experiment-id",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )

    with run_lock(run_dir, output_root=config.output_root):
        partial_entries = [path for path in run_dir.iterdir() if path.name != ".RUNNING.lock"]
        if partial_entries:
            config_path = run_dir / "config_resolved.json"
            if config_path.exists():
                stored = cast(dict[str, Any], load_json(config_path))
                if hash_canonical(stored) != cell_hash:
                    raise ResearchError(
                        "incomplete run config differs; use a new experiment-id",
                        ExitCode.INCOMPATIBLE_ARTIFACT,
                    )
            if not resume and not force:
                raise ResearchError(
                    "incomplete run exists; use --resume",
                    ExitCode.INTERRUPTED_RESUMABLE,
                )
            reset_incomplete_run(run_dir, output_root=config.output_root)
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(run_dir / "config_resolved.json", cell_payload)
        model_seed = seed + (fold - 1) * 100
        environment = configure_determinism(
            model_seed,
            deterministic=deterministic,
            device=device,
            split_random_state=config.split_contract.random_seed,
            sampler_random_state=seed,
        )
        atomic_write_json(run_dir / "environment.json", environment.model_dump(mode="json"))
        head, dirty = git_identity(config.project_root)
        started_at = utc_now()
        stage_label = {
            "e06.5": "E06.5",
            "e06.5-pd": "E06.5-PD",
            "e07": "E07",
            "e07-pd": "E07-PD",
            "e08": "E08",
        }.get(stage, stage)
        manifest = RunManifest(
            experiment_stage=stage_label,
            experiment_id=experiment_id,
            candidate=cell_name,
            fold=fold,
            seed=seed,
            model_seed=model_seed,
            git_head=head,
            git_dirty=dirty,
            dataset_manifest_hash=dataset.manifest_hash,
            split_manifest_hash=outer_manifest.manifest_hash,
            feature_manifest_hash=feature_bundle.fold_manifest_hash,
            config_hash=cell_hash,
            preflight_hash=preflight_hash,
            source_manifest_hash=source_manifest_hash,
            runtime_identity_hash=runtime_identity_hash,
            uv_lock_hash=config.uv_lock_sha256,
            python_version=environment.python_version,
            tensorflow_version=environment.tensorflow_version,
            keras_version=environment.keras_version,
            device=environment.device,
            deterministic=deterministic,
            sampling=sampler,
            loss=(
                "sparse_categorical_crossentropy"
                if method in {"ce_control", "crt_patient_aware"}
                else method
            ),
            architecture="minimal_mlp_128",
            started_at=started_at,
            finished_at="",
            status="RUNNING",
            profile=profile_name,
            publication_eligible=profile.publication_eligible,
            split_random_state=config.split_contract.random_seed,
            sampler_random_state=seed,
        )
        atomic_write_json(run_dir / "run_manifest.json", manifest.model_dump(mode="json"))
        try:
            inner_imputer, inner_scaler = _fit_preprocessor(
                feature_bundle.inner_values[inner_train]
            )
            inner_train_values = _transform(
                feature_bundle.inner_values[inner_train],
                inner_imputer,
                inner_scaler,
            )
            inner_val_values = _transform(
                feature_bundle.inner_values[inner_val],
                inner_imputer,
                inner_scaler,
            )
            inner_sampling = sample_training_values(
                inner_train_values,
                dataset.labels[inner_train],
                dataset.groups[inner_train],
                inner_train,
                sampler=sampler,
                seed=seed,
                smote_k_neighbors=config.e07.smote_k_neighbors,
            )
            inner_method = _method_state(
                config,
                method,
                dataset.labels[inner_train],
                total_epochs=profile.max_epochs,
            )
            model = _build_softmax_model(
                inner_sampling.values.shape[1],
                model_seed,
                loss=inner_method.loss,
            )
            import keras

            callback = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=profile.patience,
                restore_best_weights=True,
                verbose=0,
            )
            inner_history = model.fit(
                inner_sampling.values,
                inner_sampling.labels,
                validation_data=(inner_val_values, dataset.labels[inner_val]),
                epochs=profile.max_epochs,
                batch_size=profile.batch_size,
                callbacks=[callback, *inner_method.callbacks],
                verbose=0,
            )
            validation_losses = np.asarray(
                inner_history.history.get("val_loss", []),
                dtype=np.float64,
            )
            if validation_losses.size == 0 or not np.isfinite(validation_losses).all():
                raise ResearchError(
                    "inner validation produced invalid loss",
                    ExitCode.TRAINING_FAILURE,
                )
            best_epoch = _safe_int(np.argmin(validation_losses), "best epoch index") + 1
            history_rows = _history_rows(inner_history, "inner_selection")
            head_best_epoch = 1
            crt_inner_sampling_manifest: dict[str, Any] | None = None
            if method == "crt_patient_aware":
                encoder_before = _encoder_hash(model)
                _prepare_crt_head(model, model_seed + 10_000)
                head_sampling = sample_training_values(
                    inner_train_values,
                    dataset.labels[inner_train],
                    dataset.groups[inner_train],
                    inner_train,
                    sampler="patient_uniform",
                    seed=seed,
                    smote_k_neighbors=config.e07.smote_k_neighbors,
                )
                head_callback = keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=profile.patience,
                    restore_best_weights=True,
                    verbose=0,
                )
                head_history = model.fit(
                    head_sampling.values,
                    head_sampling.labels,
                    validation_data=(inner_val_values, dataset.labels[inner_val]),
                    epochs=profile.max_epochs,
                    batch_size=profile.batch_size,
                    callbacks=[head_callback],
                    verbose=0,
                )
                head_losses = np.asarray(
                    head_history.history.get("val_loss", []),
                    dtype=np.float64,
                )
                if head_losses.size == 0 or not np.isfinite(head_losses).all():
                    raise ResearchError(
                        "cRT head validation produced invalid loss",
                        ExitCode.TRAINING_FAILURE,
                    )
                head_best_epoch = (
                    _safe_int(
                        np.argmin(head_losses),
                        "cRT head best epoch index",
                    )
                    + 1
                )
                history_rows.extend(_history_rows(head_history, "inner_crt_head"))
                encoder_after = _encoder_hash(model)
                if encoder_before != encoder_after:
                    raise ResearchError(
                        "cRT encoder changed while frozen",
                        ExitCode.REGRESSION,
                    )
                crt_inner_sampling_manifest = head_sampling.manifest
                inner_method.manifest.update(
                    {
                        "encoder_hash_before_head": encoder_before,
                        "encoder_hash_after_head": encoder_after,
                        "encoder_immutable": True,
                        "head_best_epoch": head_best_epoch,
                        "head_sampler": "patient_uniform",
                        "dropout_rate_during_head": 0.0,
                        "fixed_representation": True,
                    }
                )
            keras.backend.clear_session()

            outer_imputer, outer_scaler = _fit_preprocessor(
                feature_bundle.outer_values[outer_train]
            )
            outer_train_values = _transform(
                feature_bundle.outer_values[outer_train],
                outer_imputer,
                outer_scaler,
            )
            outer_test_values = _transform(
                feature_bundle.outer_values[outer_test],
                outer_imputer,
                outer_scaler,
            )
            outer_sampling = sample_training_values(
                outer_train_values,
                dataset.labels[outer_train],
                dataset.groups[outer_train],
                outer_train,
                sampler=sampler,
                seed=seed,
                smote_k_neighbors=config.e07.smote_k_neighbors,
            )
            outer_method = _method_state(
                config,
                method,
                dataset.labels[outer_train],
                total_epochs=best_epoch,
            )
            final_model = _build_softmax_model(
                outer_sampling.values.shape[1],
                model_seed,
                loss=outer_method.loss,
            )
            outer_history = final_model.fit(
                outer_sampling.values,
                outer_sampling.labels,
                epochs=best_epoch,
                batch_size=profile.batch_size,
                callbacks=list(outer_method.callbacks),
                verbose=0,
            )
            history_rows.extend(_history_rows(outer_history, "outer_refit"))
            crt_outer_sampling_manifest: dict[str, Any] | None = None
            if method == "crt_patient_aware":
                encoder_before = _encoder_hash(final_model)
                _prepare_crt_head(final_model, model_seed + 10_000)
                head_sampling = sample_training_values(
                    outer_train_values,
                    dataset.labels[outer_train],
                    dataset.groups[outer_train],
                    outer_train,
                    sampler="patient_uniform",
                    seed=seed,
                    smote_k_neighbors=config.e07.smote_k_neighbors,
                )
                head_history = final_model.fit(
                    head_sampling.values,
                    head_sampling.labels,
                    epochs=head_best_epoch,
                    batch_size=profile.batch_size,
                    verbose=0,
                )
                history_rows.extend(_history_rows(head_history, "outer_crt_head"))
                encoder_after = _encoder_hash(final_model)
                if encoder_before != encoder_after:
                    raise ResearchError(
                        "cRT encoder changed during outer head retraining",
                        ExitCode.REGRESSION,
                    )
                crt_outer_sampling_manifest = head_sampling.manifest
                outer_method.manifest.update(
                    {
                        "encoder_hash_before_head": encoder_before,
                        "encoder_hash_after_head": encoder_after,
                        "encoder_immutable": True,
                        "head_epochs": head_best_epoch,
                        "head_sampler": "patient_uniform",
                        "dropout_rate_during_head": 0.0,
                        "fixed_representation": True,
                    }
                )
            probabilities = _model_probabilities(final_model, outer_test_values)
            predictions = np.argmax(probabilities, axis=1).astype(np.int64)
            checkpoint_path = run_dir / "checkpoint.keras"
            final_model.save(checkpoint_path)
            reloaded = keras.saving.load_model(
                checkpoint_path,
                compile=False,
                safe_mode=True,
            )
            reloaded_probabilities = _model_probabilities(reloaded, outer_test_values)
            reload_delta = _safe_float(
                np.max(np.abs(probabilities - reloaded_probabilities)),
                "save/reload prediction delta",
            )
            if reload_delta > 1.0e-7 or not np.array_equal(
                np.argmax(probabilities, axis=1),
                np.argmax(reloaded_probabilities, axis=1),
            ):
                raise ResearchError(
                    "save/reload prediction equivalence failed",
                    ExitCode.REGRESSION,
                    details={"max_abs_delta": reload_delta},
                )
            predictions_frame = _predictions_frame(
                dataset,
                outer_test,
                probabilities,
                predictions,
                feature_bundle,
            )
            metrics, group_metrics = compute_metrics(predictions_frame)
            metrics.update(
                {
                    "fold": fold,
                    "seed": seed,
                    "candidate": cell_name,
                    "representation": representation,
                    "method": method,
                    "profile": profile_name,
                    "deterministic": deterministic,
                    "preflight_hash": preflight_hash,
                    "source_manifest_hash": source_manifest_hash,
                    "runtime_identity_hash": runtime_identity_hash,
                    "best_epoch": best_epoch,
                    "n_outer_train": _safe_int(outer_train.size, "outer train size"),
                    "n_outer_test": _safe_int(outer_test.size, "outer test size"),
                    "n_inner_train": _safe_int(inner_train.size, "inner train size"),
                    "n_inner_validation": _safe_int(inner_val.size, "inner validation size"),
                    "save_reload_max_abs_delta": reload_delta,
                    "prediction_equivalence": True,
                    "publication_eligible": profile.publication_eligible,
                }
            )
            preprocessing_manifest = {
                "imputer_fit_scope": "inner_train_then_outer_train",
                "scaler_fit_scope": "inner_train_then_outer_train",
                "inner_train_indices_hash": hash_canonical(inner_train.tolist()),
                "outer_train_indices_hash": hash_canonical(outer_train.tolist()),
                "outer_test_indices_hash": hash_canonical(outer_test.tolist()),
                "outer_test_used_for_fit": False,
                "outer_test_used_for_selection": False,
            }
            sampling_manifest = {
                "inner": inner_sampling.manifest,
                "outer": outer_sampling.manifest,
                "crt_inner_head": crt_inner_sampling_manifest,
                "crt_outer_head": crt_outer_sampling_manifest,
            }
            method_manifest = {
                "method": method,
                "inner": inner_method.manifest,
                "outer": outer_method.manifest,
                "inner_fit_partition_index_hash": hash_canonical(inner_train.tolist()),
                "outer_fit_partition_index_hash": hash_canonical(outer_train.tolist()),
                "outer_test_used_for_method_fit": False,
            }
            atomic_write_json(run_dir / "metrics.json", metrics)
            atomic_write_json(
                run_dir / "confusion_matrix.json",
                {"labels": ["S", "V", "F"], "matrix": metrics["confusion_matrix"]},
            )
            atomic_dataframe_parquet(run_dir / "predictions.parquet", predictions_frame)
            atomic_dataframe_csv(run_dir / "group_metrics.csv", group_metrics)
            atomic_write_text(run_dir / "training_history.csv", _history_csv(history_rows))
            atomic_write_json(run_dir / "preprocessing_manifest.json", preprocessing_manifest)
            atomic_write_json(run_dir / "sampling_manifest.json", sampling_manifest)
            atomic_write_json(run_dir / "method_manifest.json", method_manifest)
            elapsed = max(0.0, environment.started_monotonic_seconds)
            stdout = (
                f"PASS candidate={cell_name} fold={fold} seed={seed}\n"
                f"best_epoch={best_epoch}\n"
                f"save_reload_max_abs_delta={reload_delta:.12g}\n"
                f"deterministic={deterministic}\n"
                f"deterministic_started_monotonic_seconds={elapsed:.6f}\n"
            )
            atomic_write_text(run_dir / "stdout.log", stdout)
            atomic_write_text(run_dir / "stderr.log", "")
            checkpoint_text = (
                "# Stage 2 research cell checkpoint\n\n"
                f"- status: PASS\n- candidate: {cell_name}\n- fold: {fold}\n"
                f"- seed: {seed}\n- F1(F): {metrics['F1_F']:.6f}\n"
                f"- macro-F1: {metrics['macro_F1']:.6f}\n"
                "- outer test used for selection: false\n"
                f"- save/reload prediction delta: {reload_delta:.12g}\n"
            )
            atomic_write_text(run_dir / "checkpoint.md", checkpoint_text)
            hashes = artifact_hashes(run_dir, REQUIRED_RUN_ARTIFACTS)
            manifest = manifest.model_copy(
                update={
                    "finished_at": utc_now(),
                    "status": "PASS",
                    "artifact_hashes": hashes,
                }
            )
            atomic_write_json(run_dir / "run_manifest.json", manifest.model_dump(mode="json"))
            write_done_marker(run_dir, manifest, REQUIRED_RUN_ARTIFACTS)
            keras.backend.clear_session()
            return CellResult(
                run_dir=run_dir,
                status="PASS",
                metrics=metrics,
                config_hash=cell_hash,
                resumed=False,
            )
        except (KeyboardInterrupt, Exception) as error:
            if isinstance(error, KeyboardInterrupt):
                interrupted = manifest.model_copy(
                    update={"finished_at": utc_now(), "status": "INTERRUPTED"}
                )
                atomic_write_json(
                    run_dir / "run_manifest.json",
                    interrupted.model_dump(mode="json"),
                )
                raise ResearchError(
                    "run interrupted and is resumable",
                    ExitCode.INTERRUPTED_RESUMABLE,
                ) from error
            failed = manifest.model_copy(update={"finished_at": utc_now(), "status": "FAILED"})
            atomic_write_json(run_dir / "run_manifest.json", failed.model_dump(mode="json"))
            if isinstance(error, ResearchError):
                raise
            raise ResearchError(
                f"training cell failed: {type(error).__name__}: {error}",
                ExitCode.TRAINING_FAILURE,
            ) from error
