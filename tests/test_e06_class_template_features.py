"""E06 H5: fold-local N/V/F class-template features."""

from __future__ import annotations

import numpy as np

from src.features.e06_templates import (
    CLASS_TEMPLATE_FEATURE_NAMES,
    FusionClassTemplateConfig,
    FusionClassTemplateTransformer,
    build_class_template_feature_schema,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, 500, dtype=np.float32)
    normal = np.exp(-((x / 0.035) ** 2)).astype(np.float32)
    ventricular = (
        0.8 * np.exp(-(((x + 0.025) / 0.080) ** 2)) - 0.7 * np.exp(-(((x - 0.070) / 0.070) ** 2))
    ).astype(np.float32)
    fusion = 0.5 * normal + 0.5 * ventricular
    signals: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    for index in range(6):
        for label, beat in (("N", normal), ("V", ventricular), ("F", fusion)):
            signals.append(beat * (0.9 + 0.04 * index))
            labels.append(label)
            groups.append(f"g{index}")
    return (
        np.stack(signals),
        np.asarray(labels),
        np.asarray(groups),
        np.stack([normal, ventricular, fusion]),
    )


def test_class_template_schema_is_fold_local() -> None:
    schema = build_class_template_feature_schema()

    assert schema.version == "e06r-h5-v1"
    assert [feature.name for feature in schema.features] == list(CLASS_TEMPLATE_FEATURE_NAMES)
    assert all(feature.fit_scope == "outer_train_only" for feature in schema.features)


def test_fusion_template_correlation_is_highest_for_fusion_shape() -> None:
    signals, labels, groups, probes = _fixture()
    transformer = FusionClassTemplateTransformer(
        FusionClassTemplateConfig(
            normal_template_count=1,
            ventricular_template_count=1,
            fusion_template_count=1,
        )
    )
    transformer.fit(
        signals,
        labels,
        groups,
        allowed_groups=set(groups),
        forbidden_groups={"outer-test"},
    )

    features = transformer.transform(probes)
    by_name = {name: features[:, index] for index, name in enumerate(CLASS_TEMPLATE_FEATURE_NAMES)}

    expected_shape = (3, len(CLASS_TEMPLATE_FEATURE_NAMES))
    assert features.shape == expected_shape
    assert np.isfinite(features).all()
    assert by_name["template_corr_F"][2] > by_name["template_corr_N"][2]
    assert by_name["template_corr_F"][2] > by_name["template_corr_V"][2]
    assert by_name["template_F_margin_over_NV"][2] > 0.0


def test_class_template_manifest_includes_fusion_state() -> None:
    signals, labels, groups, _ = _fixture()
    transformer = FusionClassTemplateTransformer(
        FusionClassTemplateConfig(
            normal_template_count=1,
            ventricular_template_count=1,
            fusion_template_count=1,
        )
    ).fit(
        signals,
        labels,
        groups,
        allowed_groups=set(groups),
        forbidden_groups={"outer-test"},
    )

    manifest = transformer.state_manifest()

    assert manifest["fusion_group_count"] == 6
    assert manifest["fusion_beat_count"] == 6
    assert len(manifest["fusion_template_sha256"]) == 64
