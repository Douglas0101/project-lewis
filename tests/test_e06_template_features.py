"""E06 H2: fold-local N/V template and hybrid-projection contracts."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.e06_templates import (
    TEMPLATE_FEATURE_NAMES,
    FusionTemplateConfig,
    FusionTemplateTransformer,
    build_template_feature_schema,
)


def _waveforms() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, 500, dtype=np.float32)
    narrow = np.exp(-((x / 0.035) ** 2)).astype(np.float32)
    wide = (
        0.8 * np.exp(-(((x + 0.025) / 0.080) ** 2)) - 0.7 * np.exp(-(((x - 0.070) / 0.070) ** 2))
    ).astype(np.float32)
    fusion = 0.5 * narrow + 0.5 * wide
    return narrow, wide, fusion


def _training_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    narrow, wide, _ = _waveforms()
    signals: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    for group_index in range(6):
        group = f"train-{group_index}"
        for scale in (0.9, 1.0, 1.1, 1.2):
            signals.append(narrow * scale)
            labels.append("N")
            groups.append(group)
            signals.append(wide * scale)
            labels.append("V")
            groups.append(group)
    return (
        np.stack(signals).astype(np.float32),
        np.asarray(labels),
        np.asarray(groups),
    )


def test_template_schema_is_train_fold_local_and_versioned() -> None:
    schema = build_template_feature_schema(FusionTemplateConfig())

    assert schema.version == "e06r-h2-v1"
    assert [feature.name for feature in schema.features] == list(TEMPLATE_FEATURE_NAMES)
    assert all(feature.fit_scope == "outer_train_only" for feature in schema.features)
    assert all(not feature.requires_future_context for feature in schema.features)
    assert len(schema.schema_sha256) == 64


def test_template_transformer_rejects_outer_test_group_overlap() -> None:
    signals, labels, groups = _training_fixture()
    transformer = FusionTemplateTransformer(
        FusionTemplateConfig(normal_template_count=1, ventricular_template_count=1)
    )

    with pytest.raises(ValueError, match="forbidden"):
        transformer.fit(
            signals,
            labels,
            groups,
            allowed_groups=set(groups),
            forbidden_groups={groups[0]},
        )


def test_template_transform_requires_fit() -> None:
    narrow, _, _ = _waveforms()
    transformer = FusionTemplateTransformer()

    with pytest.raises(RuntimeError, match="fit"):
        transformer.transform(narrow[np.newaxis, :])


def test_hybrid_projection_distinguishes_parent_and_fusion_shapes() -> None:
    signals, labels, groups = _training_fixture()
    narrow, wide, fusion = _waveforms()
    transformer = FusionTemplateTransformer(
        FusionTemplateConfig(normal_template_count=1, ventricular_template_count=1)
    )
    transformer.fit(
        signals,
        labels,
        groups,
        allowed_groups=set(groups),
        forbidden_groups={"outer-test"},
    )

    features = transformer.transform(np.stack([narrow, wide, fusion]))
    by_name = {name: features[:, index] for index, name in enumerate(TEMPLATE_FEATURE_NAMES)}

    expected_shape = (3, len(TEMPLATE_FEATURE_NAMES))
    assert features.shape == expected_shape
    assert np.isfinite(features).all()
    assert by_name["template_corr_N"][0] > by_name["template_corr_V"][0]
    assert by_name["template_corr_V"][1] > by_name["template_corr_N"][1]
    assert 0.2 < by_name["hybrid_alpha"][2] < 0.8
    assert by_name["hybrid_residual_gain"][2] > 0.0
    assert by_name["hybrid_residual"][2] <= min(
        by_name["template_distance_N"][2],
        by_name["template_distance_V"][2],
    )


def test_template_transformer_is_deterministic() -> None:
    signals, labels, groups = _training_fixture()
    config = FusionTemplateConfig(
        normal_template_count=2,
        ventricular_template_count=2,
        random_seed=42,
    )
    first = FusionTemplateTransformer(config)
    second = FusionTemplateTransformer(config)
    kwargs = {
        "allowed_groups": set(groups),
        "forbidden_groups": {"outer-test"},
    }
    first.fit(signals, labels, groups, **kwargs)
    second.fit(signals, labels, groups, **kwargs)

    assert np.array_equal(first.normal_templates, second.normal_templates)
    assert np.array_equal(first.ventricular_templates, second.ventricular_templates)
    assert first.state_manifest() == second.state_manifest()
