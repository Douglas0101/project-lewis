"""Frozen Stage 2 representation manifests and fold-local feature caches."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.features.e06_context import (
    CAUSAL_RR_FEATURE_NAMES,
    build_causal_rr_schema,
    extract_causal_rr_context,
)
from src.features.e06_templates import (
    CLASS_TEMPLATE_FEATURE_NAMES,
    FusionClassTemplateConfig,
    FusionClassTemplateTransformer,
    build_class_template_feature_schema,
)
from src.stage2_research.contracts import (
    CandidateConfig,
    CandidateName,
    ExitCode,
    InnerSplitManifest,
    ResearchConfig,
    ResearchError,
    SplitManifest,
)
from src.stage2_research.data import (
    BASE_FEATURE_NAMES,
    FullTemplateDataset,
    Stage2Dataset,
)
from src.stage2_research.integrity import (
    atomic_write_json,
    hash_canonical,
    load_json,
    sha256_array,
    sha256_file,
)
from src.stage2_research.splits import split_indices


@dataclass(frozen=True)
class FeatureBundle:
    """Inner-selection and outer-refit feature views for one fold."""

    inner_values: np.ndarray
    outer_values: np.ndarray
    feature_names: tuple[str, ...]
    static_manifest_hash: str
    fold_manifest_hash: str
    template_state: dict[str, Any]
    cache_dir: Path | None


def candidate_static_manifest(candidate: CandidateConfig) -> dict[str, Any]:
    """Build a content-addressed representation definition without fitting state."""
    if candidate.name == "baseline":
        payload: dict[str, Any] = {
            "schema_version": "stage2-feature-manifest-v1",
            "candidate": candidate.name,
            "feature_families": list(candidate.feature_families),
            "feature_names": list(BASE_FEATURE_NAMES),
            "base_context": "offline_rr_next_disclosed",
            "fusion_template_count": 0,
        }
    else:
        template_config = FusionClassTemplateConfig(
            fusion_template_count=candidate.fusion_template_count,
        )
        rr_schema = build_causal_rr_schema()
        template_schema = build_class_template_feature_schema(template_config)
        payload = {
            "schema_version": "stage2-feature-manifest-v1",
            "candidate": candidate.name,
            "feature_families": list(candidate.feature_families),
            "feature_names": list(BASE_FEATURE_NAMES)
            + list(CAUSAL_RR_FEATURE_NAMES)
            + list(CLASS_TEMPLATE_FEATURE_NAMES),
            "base_context": "offline_rr_next_disclosed",
            "rr_context_source": candidate.rr_context_source,
            "rr_schema_hash": rr_schema.schema_sha256,
            "template_schema_hash": template_schema.schema_sha256,
            "fusion_template_count": candidate.fusion_template_count,
            "template_fit_scope": "inner_train_then_outer_train",
        }
    payload["manifest_hash"] = hash_canonical(payload)
    return payload


def freeze_static_feature_manifests(
    config: ResearchConfig,
) -> dict[CandidateName, dict[str, Any]]:
    """Create or validate candidate feature definitions."""
    destination = config.output_root / "manifests" / "feature_manifests.json"
    manifests: dict[CandidateName, dict[str, Any]] = {}
    for candidate in config.candidates.values():
        manifests[candidate.name] = candidate_static_manifest(candidate)
    payload = {
        "schema_version": "stage2-feature-manifest-collection-v1",
        "candidates": manifests,
    }
    payload["collection_hash"] = hash_canonical(payload)
    if destination.exists():
        stored = load_json(destination)
        if stored != payload:
            raise ResearchError(
                "frozen feature manifest collection drift",
                ExitCode.REGRESSION,
            )
    else:
        atomic_write_json(destination, payload)
    return manifests


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(temporary, values, allow_pickle=False)
        os.replace(temporary, path)
    except (OSError, ValueError) as error:
        raise ResearchError(
            f"cannot publish feature cache: {path}",
            ExitCode.INTERRUPTED_RESUMABLE,
        ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_npz(
    path: Path,
    *,
    normal: np.ndarray,
    ventricular: np.ndarray,
    fusion: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez(
            temporary,
            normal=normal,
            ventricular=ventricular,
            fusion=fusion,
        )
        os.replace(temporary, path)
    except (OSError, ValueError) as error:
        raise ResearchError(
            f"cannot publish template cache: {path}",
            ExitCode.INTERRUPTED_RESUMABLE,
        ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_array(path: Path) -> np.ndarray:
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ResearchError(
            f"cannot load feature cache: {path}",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        ) from error
    return np.asarray(values, dtype=np.float32)


def _cache_bundle(
    cache_dir: Path,
    expected_identity: dict[str, Any],
    feature_names: tuple[str, ...],
) -> FeatureBundle | None:
    manifest_path = cache_dir / "feature_cache_manifest.json"
    if not manifest_path.exists():
        return None
    stored = load_json(manifest_path)
    identity = stored.get("identity")
    if identity != expected_identity:
        raise ResearchError(
            f"feature cache identity mismatch: {cache_dir}",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    inner_path = cache_dir / "inner_values.npy"
    outer_path = cache_dir / "outer_values.npy"
    if not inner_path.is_file() or not outer_path.is_file():
        return None
    if sha256_file(inner_path) != stored.get("inner_file_sha256") or sha256_file(
        outer_path
    ) != stored.get("outer_file_sha256"):
        raise ResearchError(
            f"feature cache file hash mismatch: {cache_dir}",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    inner_values = _load_array(inner_path)
    outer_values = _load_array(outer_path)
    if sha256_array(inner_values) != stored.get("inner_array_sha256") or sha256_array(
        outer_values
    ) != stored.get("outer_array_sha256"):
        raise ResearchError(
            f"feature cache array hash mismatch: {cache_dir}",
            ExitCode.INCOMPATIBLE_ARTIFACT,
        )
    return FeatureBundle(
        inner_values=inner_values,
        outer_values=outer_values,
        feature_names=feature_names,
        static_manifest_hash=str(expected_identity["static_manifest_hash"]),
        fold_manifest_hash=str(stored["fold_manifest_hash"]),
        template_state=dict(stored.get("template_state", {})),
        cache_dir=cache_dir,
    )


def _fit_template_transformer(
    candidate: CandidateConfig,
    full: FullTemplateDataset,
    *,
    allowed_groups: set[str],
    forbidden_groups: set[str],
) -> FusionClassTemplateTransformer:
    transformer = FusionClassTemplateTransformer(
        FusionClassTemplateConfig(
            fusion_template_count=candidate.fusion_template_count,
        )
    )
    try:
        return transformer.fit(
            full.signals,
            full.labels,
            full.groups,
            allowed_groups=allowed_groups,
            forbidden_groups=forbidden_groups,
        )
    except ValueError as error:
        raise ResearchError(
            f"template construction failed: {error}",
            ExitCode.INVALID_EXPERIMENT,
        ) from error


def build_feature_bundle(
    config: ResearchConfig,
    dataset: Stage2Dataset,
    full: FullTemplateDataset,
    outer_manifest: SplitManifest,
    inner_manifest: InnerSplitManifest,
    *,
    candidate_name: str,
    fold: int,
) -> FeatureBundle:
    """Build or reuse one candidate/fold feature state with leakage guards."""
    if candidate_name not in config.candidates:
        raise ResearchError(
            f"unknown representation candidate: {candidate_name}",
            ExitCode.ARGUMENT_ERROR,
        )
    candidate = config.candidates[candidate_name]  # type: ignore[index]
    static = candidate_static_manifest(candidate)
    static_hash = str(static["manifest_hash"])
    feature_names = tuple(str(item) for item in static["feature_names"])
    if candidate.name == "baseline":
        fold_hash = hash_canonical(
            {
                "static_manifest_hash": static_hash,
                "split_manifest_hash": outer_manifest.manifest_hash,
                "fold": fold,
                "array_sha256": sha256_array(dataset.base_features),
            }
        )
        return FeatureBundle(
            inner_values=dataset.base_features,
            outer_values=dataset.base_features,
            feature_names=feature_names,
            static_manifest_hash=static_hash,
            fold_manifest_hash=fold_hash,
            template_state={},
            cache_dir=None,
        )

    outer_train, outer_test, inner_train, inner_val = split_indices(
        outer_manifest,
        inner_manifest,
        fold,
    )
    cache_root = (
        config.output_root / "cache_pd" / "features"
        if dataset.manifest.get("generation_namespace") == "E07R_PD"
        else config.output_root / "manifests" / "features"
    )
    cache_dir = cache_root / candidate.name / f"fold_{fold}"
    expected_identity = {
        "dataset_manifest_hash": dataset.manifest_hash,
        "outer_split_manifest_hash": outer_manifest.manifest_hash,
        "inner_split_manifest_hash": inner_manifest.manifest_hash,
        "static_manifest_hash": static_hash,
        "candidate": candidate.name,
        "fold": fold,
    }
    cached = _cache_bundle(cache_dir, expected_identity, feature_names)
    if cached is not None:
        return cached

    train_groups = set(dataset.groups[outer_train].astype(str).tolist())
    test_groups = set(dataset.groups[outer_test].astype(str).tolist())
    inner_train_groups = set(dataset.groups[inner_train].astype(str).tolist())
    inner_val_groups = set(dataset.groups[inner_val].astype(str).tolist())
    if train_groups & test_groups or inner_train_groups & (inner_val_groups | test_groups):
        raise ResearchError("template split groups overlap", ExitCode.LEAKAGE)

    # Preserve the exact E06 H6/H11/H12 definition used by the confirmed manifests.
    rr_context = extract_causal_rr_context(dataset.frame, dataset.frame)
    inner_transformer = _fit_template_transformer(
        candidate,
        full,
        allowed_groups=inner_train_groups,
        forbidden_groups=inner_val_groups | test_groups,
    )
    outer_transformer = _fit_template_transformer(
        candidate,
        full,
        allowed_groups=train_groups,
        forbidden_groups=test_groups,
    )
    inner_templates = inner_transformer.transform(dataset.signals)
    outer_templates = outer_transformer.transform(dataset.signals)
    inner_values = np.column_stack([dataset.base_features, rr_context, inner_templates]).astype(
        np.float32, copy=False
    )
    outer_values = np.column_stack([dataset.base_features, rr_context, outer_templates]).astype(
        np.float32, copy=False
    )
    if inner_values.shape[1] != len(feature_names) or outer_values.shape != inner_values.shape:
        raise ResearchError("candidate feature shape contract failed", ExitCode.INVALID_EXPERIMENT)
    if np.isinf(inner_values).any() or np.isinf(outer_values).any():
        raise ResearchError("candidate features contain Inf", ExitCode.INVALID_EXPERIMENT)

    inner_state = inner_transformer.state_manifest()
    outer_state = outer_transformer.state_manifest()
    inner_sources = {str(item) for item in inner_state["source_groups"]}
    outer_sources = {str(item) for item in outer_state["source_groups"]}
    if inner_sources & (inner_val_groups | test_groups) or outer_sources & test_groups:
        raise ResearchError("template source leakage detected", ExitCode.LEAKAGE)

    cache_dir.mkdir(parents=True, exist_ok=True)
    inner_path = cache_dir / "inner_values.npy"
    outer_path = cache_dir / "outer_values.npy"
    _atomic_save_npy(inner_path, inner_values)
    _atomic_save_npy(outer_path, outer_values)
    _atomic_save_npz(
        cache_dir / "inner_templates.npz",
        normal=inner_transformer.normal_templates,
        ventricular=inner_transformer.ventricular_templates,
        fusion=inner_transformer.fusion_templates,
    )
    _atomic_save_npz(
        cache_dir / "outer_templates.npz",
        normal=outer_transformer.normal_templates,
        ventricular=outer_transformer.ventricular_templates,
        fusion=outer_transformer.fusion_templates,
    )
    fold_payload = {
        "identity": expected_identity,
        "feature_names": list(feature_names),
        "inner_array_sha256": sha256_array(inner_values),
        "outer_array_sha256": sha256_array(outer_values),
        "inner_file_sha256": sha256_file(inner_path),
        "outer_file_sha256": sha256_file(outer_path),
        "template_state": {"inner": inner_state, "outer": outer_state},
        "template_leakage_count": 0,
        "rr_context_source": candidate.rr_context_source,
    }
    fold_hash = hash_canonical(fold_payload)
    fold_payload["fold_manifest_hash"] = fold_hash
    atomic_write_json(cache_dir / "feature_cache_manifest.json", fold_payload)
    return FeatureBundle(
        inner_values=inner_values,
        outer_values=outer_values,
        feature_names=feature_names,
        static_manifest_hash=static_hash,
        fold_manifest_hash=fold_hash,
        template_state={"inner": inner_state, "outer": outer_state},
        cache_dir=cache_dir,
    )
