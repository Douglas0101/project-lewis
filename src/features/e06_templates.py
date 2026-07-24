"""Fold-local template features for AAMI fusion beats (E06R-H2).

Fusion beats combine normal and ventricular activation.  The transformer learns
small N/V QRS template banks exclusively from the outer training groups and
measures whether a beat is better explained by an interior N↔V mixture than by
either endpoint.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.cluster import KMeans


class FusionTemplateConfig(BaseModel):
    """Immutable configuration for fold-local template banks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fs_hz: float = Field(default=500.0, gt=0.0)
    qrs_pre_ms: float = Field(default=120.0, gt=0.0)
    qrs_post_ms: float = Field(default=180.0, gt=0.0)
    normal_template_count: int = Field(default=8, ge=1, le=32)
    ventricular_template_count: int = Field(default=12, ge=1, le=32)
    max_beats_per_group: int = Field(default=256, ge=1)
    random_seed: int = Field(default=42, ge=0)
    epsilon: float = Field(default=1.0e-8, gt=0.0)

    @model_validator(mode="after")
    def validate_window(self) -> FusionTemplateConfig:
        if self.qrs_pre_ms + self.qrs_post_ms > 500.0:
            raise ValueError("template QRS window must not exceed 500 ms")
        return self


class TemplateFeatureDefinition(BaseModel):
    """One fold-local feature contract entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    family: Literal["fold_local_nv_templates"] = "fold_local_nv_templates"
    units: str
    fit_scope: Literal["outer_train_only"] = "outer_train_only"
    requires_previous_context: bool = False
    requires_future_context: bool = False


class TemplateFeatureSchema(BaseModel):
    """Content-addressed schema for E06R-H2."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    mode: Literal["causal"]
    features: tuple[TemplateFeatureDefinition, ...]
    extraction_config: FusionTemplateConfig
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


_TEMPLATE_DEFINITIONS = (
    TemplateFeatureDefinition(name="template_corr_N", units="correlation"),
    TemplateFeatureDefinition(name="template_corr_V", units="correlation"),
    TemplateFeatureDefinition(name="template_corr_delta", units="correlation"),
    TemplateFeatureDefinition(name="template_distance_N", units="normalized_l2"),
    TemplateFeatureDefinition(name="template_distance_V", units="normalized_l2"),
    TemplateFeatureDefinition(name="template_distance_ratio", units="ratio"),
    TemplateFeatureDefinition(name="derivative_corr_N", units="correlation"),
    TemplateFeatureDefinition(name="derivative_corr_V", units="correlation"),
    TemplateFeatureDefinition(name="derivative_corr_delta", units="correlation"),
    TemplateFeatureDefinition(name="hybrid_alpha", units="ratio"),
    TemplateFeatureDefinition(name="hybrid_interiority", units="ratio"),
    TemplateFeatureDefinition(name="hybrid_residual", units="normalized_l2"),
    TemplateFeatureDefinition(name="hybrid_residual_gain", units="normalized_l2"),
    TemplateFeatureDefinition(name="early_distance_delta", units="normalized_l2"),
    TemplateFeatureDefinition(name="mid_distance_delta", units="normalized_l2"),
    TemplateFeatureDefinition(name="late_distance_delta", units="normalized_l2"),
)

TEMPLATE_FEATURE_NAMES = tuple(feature.name for feature in _TEMPLATE_DEFINITIONS)


def _to_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} is not an integer") from error


def _milliseconds_to_samples(milliseconds: float, fs_hz: float) -> int:
    try:
        return int(round(milliseconds * fs_hz / 1000.0))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("invalid template time-to-sample conversion") from error


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_template_feature_schema(
    config: FusionTemplateConfig | None = None,
) -> TemplateFeatureSchema:
    """Build the immutable E06R-H2 feature schema."""
    resolved = config or FusionTemplateConfig()
    payload = {
        "version": "e06r-h2-v1",
        "mode": "causal",
        "features": [feature.model_dump(mode="json") for feature in _TEMPLATE_DEFINITIONS],
        "extraction_config": resolved.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return TemplateFeatureSchema(
        version="e06r-h2-v1",
        mode="causal",
        features=_TEMPLATE_DEFINITIONS,
        extraction_config=resolved,
        schema_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _signal_matrix(signals: np.ndarray) -> np.ndarray:
    values = np.asarray(signals, dtype=np.float32)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"signals must be 2-D or single-channel 3-D; got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("template signals contain NaN or Inf")
    return values


def _normalized_qrs_windows(
    signals: np.ndarray,
    config: FusionTemplateConfig,
) -> np.ndarray:
    values = _signal_matrix(signals)
    center = values.shape[1] // 2
    pre = _milliseconds_to_samples(config.qrs_pre_ms, config.fs_hz)
    post = _milliseconds_to_samples(config.qrs_post_ms, config.fs_hz)
    start = center - pre
    end = center + post
    if start < 0 or end > values.shape[1]:
        raise ValueError("template window would require prohibited padding")
    windows = values[:, start:end].astype(np.float64, copy=True)
    baseline_count = min(20, windows.shape[1])
    windows -= np.median(windows[:, :baseline_count], axis=1, keepdims=True)
    norms = np.linalg.norm(windows, axis=1, keepdims=True)
    valid = norms[:, 0] > config.epsilon
    if not np.all(valid):
        windows[~valid] = 0.0
        norms[~valid] = 1.0
    return (windows / norms).astype(np.float32, copy=False)


def _normalize_rows(values: np.ndarray, epsilon: float) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, epsilon)


class FusionTemplateTransformer:
    """Fit N/V template banks on outer-train groups and transform beats."""

    def __init__(self, config: FusionTemplateConfig | None = None) -> None:
        self.config = config or FusionTemplateConfig()
        self._normal_templates: np.ndarray | None = None
        self._ventricular_templates: np.ndarray | None = None
        self._source_groups: tuple[str, ...] = ()
        self._normal_group_count = 0
        self._ventricular_group_count = 0
        self._normal_beat_count = 0
        self._ventricular_beat_count = 0

    @property
    def normal_templates(self) -> np.ndarray:
        if self._normal_templates is None:
            raise RuntimeError("FusionTemplateTransformer must be fit first")
        return self._normal_templates

    @property
    def ventricular_templates(self) -> np.ndarray:
        if self._ventricular_templates is None:
            raise RuntimeError("FusionTemplateTransformer must be fit first")
        return self._ventricular_templates

    def _group_prototypes(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        target_label: str,
        allowed_groups: set[str],
    ) -> tuple[np.ndarray, int, int]:
        prototypes: list[np.ndarray] = []
        beat_count = 0
        target_groups = sorted(
            set(groups[(labels == target_label) & np.isin(groups, sorted(allowed_groups))])
        )
        for group in target_groups:
            indices = np.flatnonzero((groups == group) & (labels == target_label))
            if indices.size > self.config.max_beats_per_group:
                sample_positions = np.linspace(
                    0,
                    indices.size - 1,
                    self.config.max_beats_per_group,
                )
                sampled = indices[np.rint(sample_positions).astype(np.int64)]
            else:
                sampled = indices
            if sampled.size == 0:
                continue
            sampled_windows = _normalized_qrs_windows(signals[sampled], self.config)
            prototype = np.mean(sampled_windows, axis=0)
            prototypes.append(prototype)
            beat_count += _to_int(sampled.size, "template beat count")
        if not prototypes:
            raise ValueError(f"no {target_label} templates available in outer train groups")
        stacked = np.stack(prototypes).astype(np.float64)
        stacked = _normalize_rows(stacked, self.config.epsilon)
        return (
            stacked.astype(np.float32, copy=False),
            len(prototypes),
            beat_count,
        )

    def _cluster_templates(
        self,
        group_prototypes: np.ndarray,
        requested_count: int,
        seed_offset: int,
    ) -> np.ndarray:
        unique_prototypes = np.unique(
            np.round(group_prototypes, decimals=7),
            axis=0,
        )
        cluster_count = min(requested_count, unique_prototypes.shape[0])
        if cluster_count == 1:
            centers = np.mean(unique_prototypes, axis=0, keepdims=True)
        else:
            model = KMeans(
                n_clusters=cluster_count,
                random_state=self.config.random_seed + seed_offset,
                n_init=10,  # pyright: ignore[reportArgumentType]
            )
            model.fit(unique_prototypes)
            centers = np.asarray(model.cluster_centers_, dtype=np.float64)
        normalized = _normalize_rows(centers, self.config.epsilon)
        return normalized.astype(np.float32, copy=False)

    def fit(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        *,
        allowed_groups: set[str],
        forbidden_groups: set[str],
    ) -> FusionTemplateTransformer:
        """Fit template banks without consuming any outer-test group."""
        if allowed_groups & forbidden_groups:
            raise ValueError("allowed and forbidden template groups overlap")
        values = _signal_matrix(signals)
        label_values = np.asarray(labels).astype(str)
        group_values = np.asarray(groups).astype(str)
        if not (values.shape[0] == label_values.shape[0] == group_values.shape[0]):
            raise ValueError("template signals, labels and groups must have equal length")
        observed_source_groups = set(group_values[np.isin(group_values, sorted(allowed_groups))])
        if observed_source_groups & forbidden_groups:
            raise ValueError("forbidden outer-test group reached template fitting")
        if not observed_source_groups:
            raise ValueError("no allowed groups are present in template data")

        normal_groups, normal_group_count, normal_beat_count = self._group_prototypes(
            values,
            label_values,
            group_values,
            "N",
            allowed_groups,
        )
        ventricular_groups, ventricular_group_count, ventricular_beat_count = (
            self._group_prototypes(
                values,
                label_values,
                group_values,
                "V",
                allowed_groups,
            )
        )
        self._normal_templates = self._cluster_templates(
            normal_groups,
            self.config.normal_template_count,
            seed_offset=0,
        )
        self._ventricular_templates = self._cluster_templates(
            ventricular_groups,
            self.config.ventricular_template_count,
            seed_offset=1,
        )
        self._source_groups = tuple(sorted(observed_source_groups))
        self._normal_group_count = normal_group_count
        self._ventricular_group_count = ventricular_group_count
        self._normal_beat_count = normal_beat_count
        self._ventricular_beat_count = ventricular_beat_count
        return self

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """Compute endpoint correlations and N↔V hybrid-projection features."""
        windows = _normalized_qrs_windows(signals, self.config).astype(np.float64)
        normal_templates = self.normal_templates.astype(np.float64)
        ventricular_templates = self.ventricular_templates.astype(np.float64)
        corr_n_all = np.clip(windows @ normal_templates.T, -1.0, 1.0)
        corr_v_all = np.clip(windows @ ventricular_templates.T, -1.0, 1.0)
        best_n_index = np.argmax(corr_n_all, axis=1)
        best_v_index = np.argmax(corr_v_all, axis=1)
        rows = np.arange(windows.shape[0], dtype=np.int64)
        corr_n = corr_n_all[rows, best_n_index]
        corr_v = corr_v_all[rows, best_v_index]
        best_n = normal_templates[best_n_index]
        best_v = ventricular_templates[best_v_index]
        distance_n = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * corr_n))
        distance_v = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * corr_v))

        derivative_windows = _normalize_rows(np.diff(windows, axis=1), self.config.epsilon)
        derivative_n = _normalize_rows(np.diff(best_n, axis=1), self.config.epsilon)
        derivative_v = _normalize_rows(np.diff(best_v, axis=1), self.config.epsilon)
        derivative_corr_n = np.sum(derivative_windows * derivative_n, axis=1)
        derivative_corr_v = np.sum(derivative_windows * derivative_v, axis=1)

        nv_axis = best_v - best_n
        nv_norm_squared = np.sum(np.square(nv_axis), axis=1)
        alpha = np.sum((windows - best_n) * nv_axis, axis=1) / np.maximum(
            nv_norm_squared,
            self.config.epsilon,
        )
        alpha = np.clip(alpha, 0.0, 1.0)
        hybrid = best_n + alpha[:, np.newaxis] * nv_axis
        hybrid_residual = np.linalg.norm(windows - hybrid, axis=1)
        endpoint_min = np.minimum(distance_n, distance_v)

        segment_deltas: list[np.ndarray] = []
        for segment in np.array_split(np.arange(windows.shape[1]), 3):
            residual_n = np.linalg.norm(windows[:, segment] - best_n[:, segment], axis=1)
            residual_v = np.linalg.norm(windows[:, segment] - best_v[:, segment], axis=1)
            segment_deltas.append(residual_n - residual_v)

        output = np.column_stack(
            [
                corr_n,
                corr_v,
                corr_n - corr_v,
                distance_n,
                distance_v,
                distance_n / np.maximum(distance_v, self.config.epsilon),
                derivative_corr_n,
                derivative_corr_v,
                derivative_corr_n - derivative_corr_v,
                alpha,
                1.0 - np.abs(2.0 * alpha - 1.0),
                hybrid_residual,
                endpoint_min - hybrid_residual,
                segment_deltas[0],
                segment_deltas[1],
                segment_deltas[2],
            ]
        )
        if not np.isfinite(output).all():
            raise RuntimeError("template features contain NaN or Inf")
        return output.astype(np.float32, copy=False)

    def state_manifest(self) -> dict[str, Any]:
        """Return checksummed evidence of the fitted outer-train state."""
        return {
            "config": self.config.model_dump(mode="json"),
            "source_groups": list(self._source_groups),
            "source_group_count": len(self._source_groups),
            "normal_group_count": self._normal_group_count,
            "ventricular_group_count": self._ventricular_group_count,
            "normal_beat_count": self._normal_beat_count,
            "ventricular_beat_count": self._ventricular_beat_count,
            "normal_template_shape": list(self.normal_templates.shape),
            "ventricular_template_shape": list(self.ventricular_templates.shape),
            "normal_template_sha256": _sha256_array(self.normal_templates),
            "ventricular_template_sha256": _sha256_array(self.ventricular_templates),
        }


class FusionClassTemplateConfig(FusionTemplateConfig):
    """Immutable configuration for fold-local N/V/F template banks."""

    fusion_template_count: int = Field(default=8, ge=1, le=32)


class ClassTemplateFeatureDefinition(BaseModel):
    """One fold-local N/V/F feature contract entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    family: Literal["fold_local_nvf_templates"] = "fold_local_nvf_templates"
    units: str
    fit_scope: Literal["outer_train_only"] = "outer_train_only"
    requires_previous_context: bool = False
    requires_future_context: bool = False


class ClassTemplateFeatureSchema(BaseModel):
    """Content-addressed schema for E06R-H5."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    mode: Literal["causal"]
    features: tuple[ClassTemplateFeatureDefinition, ...]
    extraction_config: FusionClassTemplateConfig
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


_CLASS_TEMPLATE_DEFINITIONS = (
    ClassTemplateFeatureDefinition(name="template_corr_N", units="correlation"),
    ClassTemplateFeatureDefinition(name="template_corr_V", units="correlation"),
    ClassTemplateFeatureDefinition(name="template_corr_F", units="correlation"),
    ClassTemplateFeatureDefinition(name="template_corr_delta_NV", units="correlation"),
    ClassTemplateFeatureDefinition(name="template_corr_delta_NF", units="correlation"),
    ClassTemplateFeatureDefinition(name="template_corr_delta_VF", units="correlation"),
    ClassTemplateFeatureDefinition(name="template_distance_N", units="normalized_l2"),
    ClassTemplateFeatureDefinition(name="template_distance_V", units="normalized_l2"),
    ClassTemplateFeatureDefinition(name="template_distance_F", units="normalized_l2"),
    ClassTemplateFeatureDefinition(name="template_distance_ratio_NV", units="ratio"),
    ClassTemplateFeatureDefinition(name="template_distance_ratio_NF", units="ratio"),
    ClassTemplateFeatureDefinition(name="template_distance_ratio_VF", units="ratio"),
    ClassTemplateFeatureDefinition(name="derivative_corr_N", units="correlation"),
    ClassTemplateFeatureDefinition(name="derivative_corr_V", units="correlation"),
    ClassTemplateFeatureDefinition(name="derivative_corr_F", units="correlation"),
    ClassTemplateFeatureDefinition(name="derivative_corr_delta_NV", units="correlation"),
    ClassTemplateFeatureDefinition(name="derivative_corr_delta_NF", units="correlation"),
    ClassTemplateFeatureDefinition(name="derivative_corr_delta_VF", units="correlation"),
    ClassTemplateFeatureDefinition(name="template_F_margin_over_NV", units="correlation"),
    ClassTemplateFeatureDefinition(name="template_F_margin_over_V", units="correlation"),
    ClassTemplateFeatureDefinition(name="template_F_margin_over_N", units="correlation"),
)

CLASS_TEMPLATE_FEATURE_NAMES = tuple(feature.name for feature in _CLASS_TEMPLATE_DEFINITIONS)


def build_class_template_feature_schema(
    config: FusionClassTemplateConfig | None = None,
) -> ClassTemplateFeatureSchema:
    """Build the immutable E06R-H5 feature schema."""
    resolved = config or FusionClassTemplateConfig()
    payload = {
        "version": "e06r-h5-v1",
        "mode": "causal",
        "features": [feature.model_dump(mode="json") for feature in _CLASS_TEMPLATE_DEFINITIONS],
        "extraction_config": resolved.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return ClassTemplateFeatureSchema(
        version="e06r-h5-v1",
        mode="causal",
        features=_CLASS_TEMPLATE_DEFINITIONS,
        extraction_config=resolved,
        schema_sha256=hashlib.sha256(encoded).hexdigest(),
    )


class FusionClassTemplateTransformer(FusionTemplateTransformer):
    """Fit N/V/F template banks on outer-train groups and transform beats."""

    config: FusionClassTemplateConfig

    def __init__(self, config: FusionClassTemplateConfig | None = None) -> None:
        super().__init__(config)
        self.config = cast(FusionClassTemplateConfig, self.config)
        self._fusion_templates: np.ndarray | None = None
        self._fusion_group_count = 0
        self._fusion_beat_count = 0

    @property
    def fusion_templates(self) -> np.ndarray:
        if self._fusion_templates is None:
            raise RuntimeError("FusionClassTemplateTransformer must be fit first")
        return self._fusion_templates

    def fit(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        *,
        allowed_groups: set[str],
        forbidden_groups: set[str],
    ) -> FusionClassTemplateTransformer:
        """Fit N/V/F template banks without consuming any outer-test group."""
        super().fit(
            signals,
            labels,
            groups,
            allowed_groups=allowed_groups,
            forbidden_groups=forbidden_groups,
        )
        values = _signal_matrix(signals)
        label_values = np.asarray(labels).astype(str)
        group_values = np.asarray(groups).astype(str)
        fusion_groups, fusion_group_count, fusion_beat_count = self._group_prototypes(
            values,
            label_values,
            group_values,
            "F",
            allowed_groups,
        )
        self._fusion_templates = self._cluster_templates(
            fusion_groups,
            self.config.fusion_template_count,
            seed_offset=2,
        )
        self._fusion_group_count = fusion_group_count
        self._fusion_beat_count = fusion_beat_count
        return self

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """Compute N/V/F template correlations, distances and margins."""
        windows = _normalized_qrs_windows(signals, self.config).astype(np.float64)
        normal_templates = self.normal_templates.astype(np.float64)
        ventricular_templates = self.ventricular_templates.astype(np.float64)
        fusion_templates = self.fusion_templates.astype(np.float64)

        corr_n_all = np.clip(windows @ normal_templates.T, -1.0, 1.0)
        corr_v_all = np.clip(windows @ ventricular_templates.T, -1.0, 1.0)
        corr_f_all = np.clip(windows @ fusion_templates.T, -1.0, 1.0)

        best_n_index = np.argmax(corr_n_all, axis=1)
        best_v_index = np.argmax(corr_v_all, axis=1)
        best_f_index = np.argmax(corr_f_all, axis=1)
        rows = np.arange(windows.shape[0], dtype=np.int64)
        corr_n = corr_n_all[rows, best_n_index]
        corr_v = corr_v_all[rows, best_v_index]
        corr_f = corr_f_all[rows, best_f_index]

        best_n = normal_templates[best_n_index]
        best_v = ventricular_templates[best_v_index]
        best_f = fusion_templates[best_f_index]

        distance_n = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * corr_n))
        distance_v = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * corr_v))
        distance_f = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * corr_f))

        derivative_windows = _normalize_rows(np.diff(windows, axis=1), self.config.epsilon)
        derivative_n = _normalize_rows(np.diff(best_n, axis=1), self.config.epsilon)
        derivative_v = _normalize_rows(np.diff(best_v, axis=1), self.config.epsilon)
        derivative_f = _normalize_rows(np.diff(best_f, axis=1), self.config.epsilon)
        derivative_corr_n = np.sum(derivative_windows * derivative_n, axis=1)
        derivative_corr_v = np.sum(derivative_windows * derivative_v, axis=1)
        derivative_corr_f = np.sum(derivative_windows * derivative_f, axis=1)

        margin_over_n = corr_f - corr_n
        margin_over_v = corr_f - corr_v
        margin_over_nv = np.minimum(margin_over_n, margin_over_v)

        output = np.column_stack(
            [
                corr_n,
                corr_v,
                corr_f,
                corr_n - corr_v,
                corr_n - corr_f,
                corr_v - corr_f,
                distance_n,
                distance_v,
                distance_f,
                distance_n / np.maximum(distance_v, self.config.epsilon),
                distance_n / np.maximum(distance_f, self.config.epsilon),
                distance_v / np.maximum(distance_f, self.config.epsilon),
                derivative_corr_n,
                derivative_corr_v,
                derivative_corr_f,
                derivative_corr_n - derivative_corr_v,
                derivative_corr_n - derivative_corr_f,
                derivative_corr_v - derivative_corr_f,
                margin_over_nv,
                margin_over_v,
                margin_over_n,
            ]
        )
        if not np.isfinite(output).all():
            raise RuntimeError("class template features contain NaN or Inf")
        return output.astype(np.float32, copy=False)

    def state_manifest(self) -> dict[str, Any]:
        """Return checksummed evidence of the fitted outer-train state."""
        manifest = super().state_manifest()
        manifest["fusion_group_count"] = self._fusion_group_count
        manifest["fusion_beat_count"] = self._fusion_beat_count
        manifest["fusion_template_shape"] = list(self.fusion_templates.shape)
        manifest["fusion_template_sha256"] = _sha256_array(self.fusion_templates)
        return manifest
