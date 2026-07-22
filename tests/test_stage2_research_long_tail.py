"""E08 train-partition long-tail and cRT contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.stage2_research import advanced_workflows
from src.stage2_research.config import load_research_config
from src.stage2_research.contracts import MethodName
from src.stage2_research.training import (
    _build_softmax_model,
    _encoder_hash,
    _method_state,
    _prepare_crt_head,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "stage2_research.yaml"
METHODS: tuple[MethodName, ...] = (
    "ce_control",
    "crt_patient_aware",
    "logit_adjustment",
    "balanced_softmax",
    "ldam_drw",
    "focal_legacy",
)


@pytest.mark.parametrize("method", METHODS)
def test_method_parameters_are_derived_from_train_labels(method: MethodName) -> None:
    config = load_research_config(CONFIG_PATH)
    train_labels = np.asarray([0] * 20 + [1] * 10 + [2] * 5, dtype=np.int64)

    state = _method_state(config, method, train_labels, total_epochs=10)

    assert state.manifest["class_counts"] == [20, 10, 5]
    assert state.manifest["fit_scope"] == "train_partition_only"
    assert len(state.manifest["class_priors"]) == 3
    if method == "logit_adjustment":
        assert state.manifest["tau"] == config.e08.logit_adjustment_tau
        assert len(state.manifest["adjustment_vector"]) == 3
    elif method == "balanced_softmax":
        assert len(state.manifest["adjustment_vector"]) == 3
    elif method == "ldam_drw":
        assert len(state.manifest["margins"]) == 3
        assert state.manifest["drw_activation_epoch"] == 5
        assert len(state.callbacks) == 1
    elif method == "focal_legacy":
        assert state.manifest["alpha"] == [0.2, 0.15, 3.0]
        assert state.manifest["gamma"] == 2.0
        assert state.manifest["class_weight"] == [1.0, 1.0, 8.0]
        assert state.manifest["legacy_parameters_frozen"]


@pytest.mark.parametrize(
    "method",
    ("logit_adjustment", "balanced_softmax", "ldam_drw", "focal_legacy"),
)
def test_long_tail_losses_are_finite(method: MethodName) -> None:
    import tensorflow as tf

    config = load_research_config(CONFIG_PATH)
    labels = np.asarray([0, 1, 2, 2], dtype=np.int64)
    probabilities = tf.constant(
        [[0.70, 0.20, 0.10], [0.20, 0.70, 0.10], [0.15, 0.25, 0.60], [0.2, 0.3, 0.5]],
        dtype=tf.float32,
    )
    state = _method_state(config, method, labels, total_epochs=10)
    losses = state.loss(tf.constant(labels), probabilities).numpy()

    assert losses.size >= 1
    assert np.isfinite(losses).all()
    assert np.all(losses >= 0.0)


def test_e08_ranking_maximizes_worst_fold_then_uses_complexity() -> None:
    scope = {"outside_208_213": {"F1_F": 0.1}}
    metrics = pd.DataFrame(
        [
            {"candidate": "ce_control", "fold": 1, "seed": 17, "F1_F": 0.2, "scopes": scope},
            {
                "candidate": "logit_adjustment",
                "fold": 1,
                "seed": 17,
                "F1_F": 0.3,
                "scopes": scope,
            },
            {
                "candidate": "balanced_softmax",
                "fold": 1,
                "seed": 17,
                "F1_F": 0.3,
                "scopes": scope,
            },
        ]
    )

    def summary(minimum: float) -> dict[str, object]:
        return {
            "F1_F": {"mean": 0.3, "std": 0.1, "min": minimum},
            "zero_F1_fold_count": 0,
            "macro_F1": {"mean": 0.5},
            "precision_F": {"mean": 0.4},
            "recall_F": {"mean": 0.4},
            "AP_F": {"mean": 0.4},
        }

    summaries = {
        "ce_control": summary(0.1),
        "logit_adjustment": summary(0.2),
        "balanced_softmax": summary(0.1),
    }
    ranking, _ = advanced_workflows._derive_e08_ranking(
        metrics,
        ("ce_control", "logit_adjustment", "balanced_softmax"),
        summaries,
    )
    assert ranking.index("logit_adjustment") < ranking.index("balanced_softmax")

    summaries["balanced_softmax"] = summary(0.2)
    ranking, _ = advanced_workflows._derive_e08_ranking(
        metrics,
        ("ce_control", "logit_adjustment", "balanced_softmax"),
        summaries,
    )
    assert ranking.index("logit_adjustment") < ranking.index("balanced_softmax")


def test_e08_control_exit_uses_control_relative_comparison() -> None:
    config = load_research_config(CONFIG_PATH)
    summary = {
        "F1_F": {"mean": 0.2},
        "zero_F1_fold_count": 0,
    }
    fields = advanced_workflows._derive_e08_exit_fields(
        config,
        summary,
        advanced_workflows._control_comparison(),
    )

    assert fields["variability_dominates_gain"]
    assert fields["NEXT_STAGE"] == "HYBRID_CONV1D"


def test_crt_head_training_cannot_mutate_encoder(tmp_path: Path) -> None:
    import keras
    import tensorflow as tf

    rng = np.random.default_rng(42)
    values = rng.normal(size=(24, 8)).astype(np.float32)
    labels = np.asarray([0, 1, 2] * 8, dtype=np.int64)
    model = _build_softmax_model(values.shape[1], seed=17)
    model.fit(values, labels, epochs=1, batch_size=8, verbose=0)
    encoder_before = _encoder_hash(model)

    _prepare_crt_head(model, seed=10_017)
    model.fit(values, labels, epochs=1, batch_size=8, verbose=0)
    encoder_after = _encoder_hash(model)

    assert encoder_before == encoder_after
    assert not model.get_layer("encoder_dense").trainable
    assert model.get_layer("encoder_dropout").rate == 0.0
    assert model.get_layer("classifier").trainable
    representation = keras.Model(
        inputs=model.inputs[0],
        outputs=model.get_layer("encoder_dropout").output,
    )
    values_tensor = tf.constant(values)
    first = representation(values_tensor, training=True).numpy()
    second = representation(values_tensor, training=True).numpy()
    np.testing.assert_array_equal(first, second)
    probabilities = np.asarray(model.predict(values, verbose=0))
    checkpoint = tmp_path / "crt.keras"
    model.save(checkpoint)
    reloaded = cast(
        Any,
        keras.saving.load_model(checkpoint, compile=False, safe_mode=True),
    )
    assert reloaded.get_layer("encoder_dropout").rate == 0.0
    reloaded_probabilities = np.asarray(reloaded.predict(values, verbose=0))
    np.testing.assert_array_equal(probabilities, reloaded_probabilities)
    keras.backend.clear_session()
