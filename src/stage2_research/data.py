"""Immutable dataset loading and integrity validation for Stage 2 research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from src.stage2_research.contracts import ExitCode, ResearchConfig, ResearchError
from src.stage2_research.integrity import hash_canonical, sha256_file

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
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}
KEY_COLUMNS = ("dataset", "record_id", "beat_idx")


def _safe_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResearchError(
            f"{name} is not an integer",
            ExitCode.DATA_INTEGRITY,
        ) from error


def frame_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Read exactly one DataFrame column."""
    value = frame.loc[:, name]
    if isinstance(value, pd.DataFrame):
        raise ResearchError(
            f"duplicate DataFrame column: {name}",
            ExitCode.DATA_INTEGRITY,
        )
    return cast(pd.Series, value)


@dataclass(frozen=True)
class Stage2Dataset:
    """Validated S/V/F Stage 2 data."""

    frame: pd.DataFrame
    signals: np.ndarray
    labels: np.ndarray
    base_features: np.ndarray
    groups: np.ndarray
    manifest: dict[str, Any]
    manifest_hash: str


@dataclass(frozen=True)
class FullTemplateDataset:
    """Validated full N/S/V/F/Q beat source used for train-only templates."""

    frame: pd.DataFrame
    signals: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    manifest: dict[str, Any]


def _load_npz_array(path: Path, key: str, dtype: np.dtype[Any]) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise ResearchError(
                    f"NPZ key {key!r} missing: {path}",
                    ExitCode.DATA_INTEGRITY,
                )
            return np.asarray(archive[key], dtype=dtype)
    except (OSError, ValueError) as error:
        raise ResearchError(
            f"cannot load NPZ: {path}",
            ExitCode.DATA_INTEGRITY,
        ) from error


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path).reset_index(drop=True)
    except (OSError, ValueError, ImportError) as error:
        raise ResearchError(
            f"cannot load parquet: {path}",
            ExitCode.DATA_INTEGRITY,
        ) from error


def _verify_expected_hash(path: Path, expected: str) -> str:
    if not path.is_file():
        raise ResearchError(f"dataset not found: {path}", ExitCode.DATA_INTEGRITY)
    actual = sha256_file(path)
    if actual != expected:
        raise ResearchError(
            f"dataset hash mismatch: {path}",
            ExitCode.DATA_INTEGRITY,
            details={"expected": expected, "actual": actual},
        )
    return actual


def _validate_group_identity(frame: pd.DataFrame) -> None:
    pairs = frame.loc[:, ["dataset", "record_id"]].drop_duplicates()
    datasets_per_record = pairs.groupby("record_id")["dataset"].nunique()
    ambiguous = datasets_per_record[datasets_per_record > 1]
    if not ambiguous.empty:
        raise ResearchError(
            "record_id is ambiguous across datasets; patient-safe group key unavailable",
            ExitCode.DATA_INTEGRITY,
            details={"record_ids": ambiguous.index.astype(str).tolist()},
        )


def load_stage2_dataset(config: ResearchConfig) -> Stage2Dataset:
    """Load and prove the exact S/V/F dataset contract."""
    paths = config.datasets
    stage2_npz_hash = _verify_expected_hash(paths.stage2_npz.path, paths.stage2_npz.sha256)
    stage2_parquet_hash = _verify_expected_hash(
        paths.stage2_parquet.path,
        paths.stage2_parquet.sha256,
    )
    frame = _read_parquet(paths.stage2_parquet.path)
    signals = _load_npz_array(paths.stage2_npz.path, "X", np.dtype(np.float32))
    labels = _load_npz_array(paths.stage2_npz.path, "y", np.dtype(np.int64))
    if not (signals.ndim == 2 or (signals.ndim == 3 and signals.shape[-1] == 1)):
        raise ResearchError(
            f"unexpected Stage 2 signal shape: {signals.shape}",
            ExitCode.DATA_INTEGRITY,
        )
    if labels.ndim != 1 or len(frame) != signals.shape[0] or labels.size != signals.shape[0]:
        raise ResearchError(
            "Stage 2 parquet/signals/labels length mismatch",
            ExitCode.DATA_INTEGRITY,
        )
    missing = [
        name for name in (*KEY_COLUMNS, "label_aami", *BASE_FEATURE_NAMES) if name not in frame
    ]
    if missing:
        raise ResearchError(
            f"Stage 2 parquet columns missing: {missing}",
            ExitCode.DATA_INTEGRITY,
        )
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ResearchError("Stage 2 beat keys are not unique", ExitCode.DATA_INTEGRITY)
    _validate_group_identity(frame)
    label_names = frame_column(frame, "label_aami").astype(str).to_numpy()
    try:
        expected_labels = np.asarray([LABEL_TO_INDEX[item] for item in label_names], dtype=np.int64)
    except KeyError as error:
        raise ResearchError(
            "Stage 2 labels must be exactly S/V/F",
            ExitCode.DATA_INTEGRITY,
        ) from error
    if not np.array_equal(expected_labels, labels):
        raise ResearchError(
            "Stage 2 NPZ/parquet label alignment failed",
            ExitCode.DATA_INTEGRITY,
        )
    if not np.isfinite(signals).all():
        raise ResearchError("Stage 2 signals contain NaN/Inf", ExitCode.DATA_INTEGRITY)
    base_features = frame.loc[:, list(BASE_FEATURE_NAMES)].to_numpy(dtype=np.float32)
    if np.isinf(base_features).any():
        raise ResearchError("base features contain Inf", ExitCode.DATA_INTEGRITY)
    groups = frame_column(frame, "record_id").astype(str).to_numpy()
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_counts = {
        INDEX_TO_LABEL[_safe_int(label, "class label")]: _safe_int(
            count,
            "class count",
        )
        for label, count in zip(unique_labels, counts, strict=True)
    }
    if set(class_counts) != {"S", "V", "F"}:
        raise ResearchError("one or more S/V/F classes are absent", ExitCode.DATA_INTEGRITY)
    center_index = signals.shape[1] // 2
    mismatch_count = 0
    if "r_peak_in_segment" in frame:
        peaks = frame_column(frame, "r_peak_in_segment").to_numpy(dtype=np.int64)
        mismatch_count = _safe_int(
            np.sum(peaks != center_index),
            "R-peak center mismatch count",
        )
    manifest = {
        "schema_version": "stage2-dataset-v2.4",
        "stage2_npz_sha256": stage2_npz_hash,
        "stage2_parquet_sha256": stage2_parquet_hash,
        "n_samples": _safe_int(labels.size, "Stage 2 sample count"),
        "signal_shape": list(signals.shape),
        "signal_dtype": str(signals.dtype),
        "class_counts": class_counts,
        "n_groups": _safe_int(np.unique(groups).size, "Stage 2 group count"),
        "group_key": "record_id",
        "composite_beat_key": list(KEY_COLUMNS),
        "base_feature_names": list(BASE_FEATURE_NAMES),
        "base_feature_context": "offline_rr_next_disclosed",
        "r_peak_metadata_center_mismatch_count": mismatch_count,
    }
    manifest_hash = hash_canonical(manifest)
    manifest["manifest_hash"] = manifest_hash
    return Stage2Dataset(
        frame=frame,
        signals=signals,
        labels=labels,
        base_features=base_features,
        groups=groups,
        manifest=manifest,
        manifest_hash=manifest_hash,
    )


def load_full_template_dataset(config: ResearchConfig) -> FullTemplateDataset:
    """Load the immutable train-only template source."""
    paths = config.datasets
    full_npz_hash = _verify_expected_hash(paths.full_npz.path, paths.full_npz.sha256)
    full_parquet_hash = _verify_expected_hash(
        paths.full_parquet.path,
        paths.full_parquet.sha256,
    )
    frame = _read_parquet(paths.full_parquet.path)
    signals = _load_npz_array(paths.full_npz.path, "X", np.dtype(np.float32))
    if signals.ndim == 3 and signals.shape[-1] == 1:
        accepted_layout = signals.shape[1] == 500
    else:
        accepted_layout = signals.ndim == 2 and signals.shape[1] == 500
    if not accepted_layout or len(frame) != signals.shape[0]:
        raise ResearchError(
            f"full template source shape/length mismatch: {signals.shape}",
            ExitCode.DATA_INTEGRITY,
        )
    required = (*KEY_COLUMNS, "label_aami", "r_peak_sample")
    missing = [name for name in required if name not in frame]
    if missing:
        raise ResearchError(
            f"full template parquet columns missing: {missing}",
            ExitCode.DATA_INTEGRITY,
        )
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ResearchError("full template beat keys are not unique", ExitCode.DATA_INTEGRITY)
    _validate_group_identity(frame)
    labels = frame_column(frame, "label_aami").astype(str).to_numpy()
    allowed = {"N", "S", "V", "F", "Q"}
    if not set(np.unique(labels)).issubset(allowed) or "F" not in labels:
        raise ResearchError("invalid full template label set", ExitCode.DATA_INTEGRITY)
    if not np.isfinite(signals).all():
        raise ResearchError("full template signals contain NaN/Inf", ExitCode.DATA_INTEGRITY)
    groups = frame_column(frame, "record_id").astype(str).to_numpy()
    unique_labels, counts = np.unique(labels, return_counts=True)
    manifest = {
        "full_npz_sha256": full_npz_hash,
        "full_parquet_sha256": full_parquet_hash,
        "n_samples": _safe_int(signals.shape[0], "full template sample count"),
        "signal_shape": list(signals.shape),
        "class_counts": {
            str(label): _safe_int(count, "full template class count")
            for label, count in zip(unique_labels, counts, strict=True)
        },
        "n_groups": _safe_int(np.unique(groups).size, "full template group count"),
        "group_key": "record_id",
    }
    return FullTemplateDataset(
        frame=frame,
        signals=signals,
        labels=labels,
        groups=groups,
        manifest=manifest,
    )
