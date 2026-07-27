"""E07 train-only sampler contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.stage2_research.advanced_workflows import _require_stage_execution_contract
from src.stage2_research.config import load_research_config
from src.stage2_research.contracts import ResearchError, SamplerName
from src.stage2_research.training import sample_training_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "stage2_research.yaml"

SAMPLERS: tuple[SamplerName, ...] = (
    "natural",
    "random_oversampling",
    "patient_uniform",
    "patient_sqrt",
    "smote",
)
PD_SAMPLERS: tuple[SamplerName, ...] = (
    "pd_s0_natural",
    "pd_s1_f_target",
    "pd_s2_patient_uniform_capped",
    "pd_s3_patient_sqrt_capped",
    "pd_s4_focal_gentle",
    "pd_s5_smote_feature",
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    labels = np.asarray([0] * 24 + [1] * 16 + [2] * 8, dtype=np.int64)
    values = rng.normal(size=(labels.size, 6)).astype(np.float32)
    groups = np.asarray(
        [f"g{index % 6}" for index in range(24)]
        + [f"g{index % 4 + 6}" for index in range(16)]
        + [f"g{index % 3 + 10}" for index in range(8)]
    )
    indices = np.arange(labels.size, dtype=np.int64)
    return values, labels, groups, indices


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_sampler_is_deterministic_and_train_scoped(sampler: SamplerName) -> None:
    values, labels, groups, indices = _fixture()

    first = sample_training_values(
        values,
        labels,
        groups,
        indices,
        sampler=sampler,
        seed=17,
        smote_k_neighbors=3,
    )
    second = sample_training_values(
        values,
        labels,
        groups,
        indices,
        sampler=sampler,
        seed=17,
        smote_k_neighbors=3,
    )

    assert np.array_equal(first.values, second.values)
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.source_indices, second.source_indices)
    assert first.manifest == second.manifest
    assert not first.manifest["validation_or_test_sampled"]
    assert first.manifest["source_outside_partition_count"] == 0
    assert np.isfinite(first.values).all()
    assert set(first.source_indices[first.source_indices >= 0]) <= set(indices)


def test_patient_samplers_balance_classes_without_cross_partition_rows() -> None:
    values, labels, groups, indices = _fixture()

    for sampler in ("patient_uniform", "patient_sqrt"):
        result = sample_training_values(
            values,
            labels,
            groups,
            indices,
            sampler=sampler,
            seed=29,
            smote_k_neighbors=3,
        )
        counts = np.unique(result.labels, return_counts=True)[1]
        assert np.unique(counts).size == 1
        assert np.all(result.source_indices >= 0)


@pytest.mark.parametrize("sampler", PD_SAMPLERS)
def test_pd_sampler_is_deterministic_and_train_only(sampler: SamplerName) -> None:
    rng = np.random.default_rng(7)
    labels = np.asarray([0] * 40 + [1] * 30 + [2] * 3, dtype=np.int64)
    values = rng.normal(size=(labels.size, 5)).astype(np.float32)
    groups = np.asarray(
        [f"p{index % 8}" for index in range(40)]
        + [f"p{index % 6 + 8}" for index in range(30)]
        + ["f1", "f2", "f3"]
    )
    indices = np.arange(1000, 1000 + labels.size, dtype=np.int64)

    first = sample_training_values(
        values,
        labels,
        groups,
        indices,
        sampler=sampler,
        seed=17,
        smote_k_neighbors=2,
    )
    second = sample_training_values(
        values,
        labels,
        groups,
        indices,
        sampler=sampler,
        seed=17,
        smote_k_neighbors=2,
    )

    assert np.array_equal(first.values, second.values)
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.source_indices, second.source_indices)
    assert first.manifest == second.manifest
    assert first.manifest["sampler_scope"] == "TRAIN_ONLY_FEATURE_SPACE"
    assert first.manifest["validation_or_test_sampled"] is False
    assert first.manifest["source_outside_partition_count"] == 0
    assert set(first.source_indices[first.source_indices >= 0]) <= set(indices)
    if sampler in {
        "pd_s1_f_target",
        "pd_s2_patient_uniform_capped",
        "pd_s3_patient_sqrt_capped",
        "pd_s5_smote_feature",
    }:
        assert first.manifest["realized_f_fraction"] == pytest.approx(0.125)
    if sampler in {
        "pd_s2_patient_uniform_capped",
        "pd_s3_patient_sqrt_capped",
    }:
        cap = first.manifest["patient_cap"]
        assert cap is not None
        assert max(first.manifest["patient_f_contributions"].values()) <= cap
    if sampler == "pd_s5_smote_feature":
        assert first.manifest["synthetic_count"] == 7
        assert np.sum(first.source_indices < 0) == 7


def test_e07_execution_contract_rejects_wrong_profile_or_seeds() -> None:
    config = load_research_config(CONFIG_PATH)
    _require_stage_execution_contract(
        config,
        stage="E07",
        names=config.e07.samplers,
        expected_names=config.e07.samplers,
        folds=config.folds,
        seeds=config.e07.screening_seeds,
        expected_seeds=config.e07.screening_seeds,
        profile="screening",
        deterministic=False,
        device="cpu",
        max_parallel=1,
    )

    with pytest.raises(ResearchError):
        _require_stage_execution_contract(
            config,
            stage="E07",
            names=config.e07.samplers,
            expected_names=config.e07.samplers,
            folds=config.folds,
            seeds=config.e07.final_seeds,
            expected_seeds=config.e07.screening_seeds,
            profile="screening",
            deterministic=False,
            device="cpu",
            max_parallel=1,
        )


def test_smote_marks_only_synthetic_rows_with_negative_provenance() -> None:
    values, labels, groups, indices = _fixture()

    result = sample_training_values(
        values,
        labels,
        groups,
        indices,
        sampler="smote",
        seed=43,
        smote_k_neighbors=3,
    )

    synthetic = result.source_indices < 0
    assert np.sum(synthetic) == result.manifest["synthetic_count"]
    assert np.all(result.source_indices[~synthetic] >= 0)
