"""E06 H3: causal multi-beat RR context contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.e06_context import (
    CAUSAL_RR_FEATURE_NAMES,
    build_causal_rr_schema,
    extract_causal_rr_context,
)


def _full_sequence() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["MITDB"] * 7 + ["SVDB"] * 5,
            "record_id": ["100"] * 7 + ["800"] * 5,
            "beat_idx": list(range(7)) + list(range(5)),
            "r_peak_sample": [0, 400, 850, 1350, 1900, 2500, 3150] + [0, 500, 1000, 1500, 2000],
        }
    )


def test_causal_rr_schema_has_no_future_context() -> None:
    schema = build_causal_rr_schema()

    assert schema.version == "e06r-h3-v1"
    assert schema.mode == "causal"
    assert [feature.name for feature in schema.features] == list(CAUSAL_RR_FEATURE_NAMES)
    assert all(feature.requires_previous_context for feature in schema.features)
    assert all(not feature.requires_future_context for feature in schema.features)
    assert len(schema.schema_sha256) == 64


def test_causal_rr_context_values_and_target_order() -> None:
    full = _full_sequence()
    targets = full.iloc[[5, 3, 11]].copy()

    features = extract_causal_rr_context(full, targets, fs_hz=500.0)
    by_name = {name: features[:, index] for index, name in enumerate(CAUSAL_RR_FEATURE_NAMES)}

    expected_shape = (3, len(CAUSAL_RR_FEATURE_NAMES))
    assert features.shape == expected_shape
    assert by_name["rr_prev_1"][0] == 1200.0
    assert by_name["rr_prev_2"][0] == 1100.0
    assert by_name["rr_prev_3"][0] == 1000.0
    assert by_name["rr_prev_4"][0] == 900.0
    assert by_name["rr_prev_1"][1] == 1000.0
    assert by_name["rr_prev_1"][2] == 1000.0


def test_causal_rr_context_does_not_change_when_future_peak_changes() -> None:
    full = _full_sequence()
    target = full.iloc[[4]].copy()
    before = extract_causal_rr_context(full, target)

    changed = full.copy()
    changed.loc[6, "r_peak_sample"] = 5000
    after = extract_causal_rr_context(changed, target)

    assert np.array_equal(before, after, equal_nan=True)


def test_causal_rr_context_uses_nan_for_insufficient_history() -> None:
    full = _full_sequence()
    target = full.iloc[[0]].copy()

    features = extract_causal_rr_context(full, target)

    assert np.isnan(features[0, 0])
    assert np.isnan(features[0, 4])


def test_causal_rr_context_requires_unique_keys() -> None:
    full = pd.concat([_full_sequence(), _full_sequence().iloc[[0]]], ignore_index=True)
    target = _full_sequence().iloc[[0]].copy()

    try:
        extract_causal_rr_context(full, target)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate beat key was accepted")
