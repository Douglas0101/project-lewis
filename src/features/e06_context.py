"""Causal multi-beat RR context for the reopened E06 H3 ablation."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class CausalRRFeatureDefinition(BaseModel):
    """One causal temporal feature contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    family: Literal["causal_rr_context"] = "causal_rr_context"
    units: str
    requires_previous_context: bool = True
    requires_future_context: bool = False


class CausalRRFeatureSchema(BaseModel):
    """Content-addressed E06R-H3 schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    mode: Literal["causal"]
    fs_hz: float = Field(gt=0.0)
    features: tuple[CausalRRFeatureDefinition, ...]
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


_DEFINITIONS = (
    CausalRRFeatureDefinition(name="rr_prev_1", units="ms"),
    CausalRRFeatureDefinition(name="rr_prev_2", units="ms"),
    CausalRRFeatureDefinition(name="rr_prev_3", units="ms"),
    CausalRRFeatureDefinition(name="rr_prev_4", units="ms"),
    CausalRRFeatureDefinition(name="rr_prev_mean_4", units="ms"),
    CausalRRFeatureDefinition(name="rr_prev_std_4", units="ms"),
    CausalRRFeatureDefinition(name="rr_prev_cv_4", units="ratio"),
    CausalRRFeatureDefinition(name="rr_prev_median_4", units="ms"),
    CausalRRFeatureDefinition(name="rr_prev_mad_4", units="ms"),
    CausalRRFeatureDefinition(name="rr_current_over_median_4", units="ratio"),
    CausalRRFeatureDefinition(name="rr_delta_1", units="ms"),
    CausalRRFeatureDefinition(name="rr_delta_2", units="ms"),
    CausalRRFeatureDefinition(name="rr_acceleration", units="ms"),
    CausalRRFeatureDefinition(name="rr_rmssd_4", units="ms"),
    CausalRRFeatureDefinition(name="rr_pnn50_4", units="ratio"),
    CausalRRFeatureDefinition(name="tachycardia_flag", units="bool"),
    CausalRRFeatureDefinition(name="bradycardia_flag", units="bool"),
)

CAUSAL_RR_FEATURE_NAMES = tuple(feature.name for feature in _DEFINITIONS)
_KEY_COLUMNS = ["dataset", "record_id", "beat_idx"]
_REQUIRED_COLUMNS = _KEY_COLUMNS + ["r_peak_sample"]


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    value = frame.loc[:, name]
    if isinstance(value, pd.DataFrame):
        raise ValueError(f"duplicate DataFrame column: {name}")
    return cast(pd.Series, value)


def build_causal_rr_schema(fs_hz: float = 500.0) -> CausalRRFeatureSchema:
    """Build the immutable H3 feature schema."""
    if fs_hz <= 0.0:
        raise ValueError("fs_hz must be positive")
    payload = {
        "version": "e06r-h3-v1",
        "mode": "causal",
        "fs_hz": fs_hz,
        "features": [feature.model_dump(mode="json") for feature in _DEFINITIONS],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return CausalRRFeatureSchema(
        version="e06r-h3-v1",
        mode="causal",
        fs_hz=fs_hz,
        features=_DEFINITIONS,
        schema_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _validate_frames(full: pd.DataFrame, targets: pd.DataFrame) -> None:
    missing_full = [column for column in _REQUIRED_COLUMNS if column not in full.columns]
    missing_target = [column for column in _KEY_COLUMNS if column not in targets.columns]
    if missing_full or missing_target:
        raise ValueError(f"missing context columns: full={missing_full}, targets={missing_target}")
    if full.duplicated(_KEY_COLUMNS).any():
        raise ValueError("full beat keys must be unique")
    if targets.duplicated(_KEY_COLUMNS).any():
        raise ValueError("target beat keys must be unique")


def _shift(values: np.ndarray, count: int) -> np.ndarray:
    shifted = np.full(values.shape, np.nan, dtype=np.float64)
    if count < values.size:
        shifted[count:] = values[:-count]
    return shifted


def _record_context(peaks: np.ndarray, fs_hz: float) -> np.ndarray:
    n_beats = peaks.shape[0]
    rr_prev_1 = np.full(n_beats, np.nan, dtype=np.float64)
    if n_beats > 1:
        rr_prev_1[1:] = np.diff(peaks) / fs_hz * 1000.0
    rr_prev_2 = _shift(rr_prev_1, 1)
    rr_prev_3 = _shift(rr_prev_1, 2)
    rr_prev_4 = _shift(rr_prev_1, 3)
    history = np.column_stack([rr_prev_1, rr_prev_2, rr_prev_3, rr_prev_4])
    complete = np.isfinite(history).all(axis=1)

    mean_4 = np.full(n_beats, np.nan, dtype=np.float64)
    std_4 = np.full(n_beats, np.nan, dtype=np.float64)
    median_4 = np.full(n_beats, np.nan, dtype=np.float64)
    mad_4 = np.full(n_beats, np.nan, dtype=np.float64)
    rmssd_4 = np.full(n_beats, np.nan, dtype=np.float64)
    pnn50_4 = np.full(n_beats, np.nan, dtype=np.float64)
    if np.any(complete):
        complete_history = history[complete]
        mean_4[complete] = np.mean(complete_history, axis=1)
        std_4[complete] = np.std(complete_history, axis=1)
        median_values = np.median(complete_history, axis=1)
        median_4[complete] = median_values
        mad_4[complete] = np.median(
            np.abs(complete_history - median_values[:, np.newaxis]),
            axis=1,
        )
        differences = np.diff(complete_history, axis=1)
        rmssd_4[complete] = np.sqrt(np.mean(np.square(differences), axis=1))
        pnn50_4[complete] = np.mean(np.abs(differences) > 50.0, axis=1)

    epsilon = 1.0e-8
    cv_4 = std_4 / np.maximum(mean_4, epsilon)
    current_over_median = rr_prev_1 / np.maximum(median_4, epsilon)
    delta_1 = rr_prev_1 - rr_prev_2
    delta_2 = rr_prev_2 - rr_prev_3
    acceleration = rr_prev_1 - 2.0 * rr_prev_2 + rr_prev_3
    tachycardia = np.where(np.isfinite(rr_prev_1), rr_prev_1 < 600.0, np.nan)
    bradycardia = np.where(np.isfinite(rr_prev_1), rr_prev_1 > 1000.0, np.nan)

    return np.column_stack(
        [
            rr_prev_1,
            rr_prev_2,
            rr_prev_3,
            rr_prev_4,
            mean_4,
            std_4,
            cv_4,
            median_4,
            mad_4,
            current_over_median,
            delta_1,
            delta_2,
            acceleration,
            rmssd_4,
            pnn50_4,
            tachycardia,
            bradycardia,
        ]
    ).astype(np.float32, copy=False)


def extract_causal_rr_context(
    full_beats: pd.DataFrame,
    target_beats: pd.DataFrame,
    fs_hz: float = 500.0,
) -> np.ndarray:
    """Extract prior-only RR context and restore the exact target row order."""
    if fs_hz <= 0.0:
        raise ValueError("fs_hz must be positive")
    full = full_beats.reset_index(drop=True)
    targets = target_beats.reset_index(drop=True)
    _validate_frames(full, targets)

    full_context = np.full(
        (len(full), len(CAUSAL_RR_FEATURE_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    for _, group_frame in full.groupby(["dataset", "record_id"], sort=False):
        row_indices = group_frame.index.to_numpy(dtype=np.int64)
        peaks = _column(group_frame, "r_peak_sample").to_numpy(dtype=np.int64)
        order = np.argsort(peaks, kind="stable")
        sorted_rows = row_indices[order]
        sorted_peaks = peaks[order]
        full_context[sorted_rows] = _record_context(sorted_peaks, fs_hz)

    full_index = pd.MultiIndex.from_frame(full.loc[:, _KEY_COLUMNS])
    target_index = pd.MultiIndex.from_frame(targets.loc[:, _KEY_COLUMNS])
    locations = full_index.get_indexer(target_index)
    if np.any(locations < 0):
        raise ValueError("a target beat key is absent from the full sequence")
    return full_context[locations]
