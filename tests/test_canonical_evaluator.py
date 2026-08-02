"""Testes do avaliador canônico (ML Protocol v2, evaluator v2.0).

Dados 100% sintéticos — nenhum teste carrega TensorFlow, datasets ou runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation.calibration_metrics import (
    apply_temperature,
    fit_temperature_multilabel,
)
from src.evaluation.canonical_evaluator import (
    _extract_legacy_calibration,
    evaluate,
    write_artifacts,
)
from src.evaluation.schema import (
    ComparabilityContract,
    MetricsJson,
    check_comparable,
)
from src.evaluation.thresholding import (
    ThresholdPolicy,
    apply_thresholds,
    fit_thresholds,
)

CLASSES = ("NORM", "CD", "MI", "HYP", "STTC")
RNG = np.random.RandomState(13)


def _synthetic_multilabel(n: int = 4000, k: int = 5, seed: int = 13):
    """Multi-label sintético com separação moderada (auroc ~0.85)."""
    rng = np.random.RandomState(seed)
    logits_true = rng.normal(0.0, 1.5, size=(n, k))
    probs_true = 1.0 / (1.0 + np.exp(-logits_true))
    y_true = (rng.uniform(size=(n, k)) < probs_true).astype(int)
    # garante pelo menos um positivo/negativo por classe
    for j in range(k):
        y_true[0, j] = 1
        y_true[1, j] = 0
    # modelo: logits verdadeiros + ruído (bom ranking)
    y_score = 1.0 / (1.0 + np.exp(-(logits_true + rng.normal(0, 0.5, size=(n, k)))))
    return y_true, y_score


def _underconfident_pair(n: int = 6000, seed: int = 7):
    """(y_true, probs) de um modelo sub-confiante: logits comprimidos por 0.3."""
    rng = np.random.RandomState(seed)
    z = rng.normal(0.0, 2.0, size=(n, 3))
    p_true = 1.0 / (1.0 + np.exp(-z))
    y_true = (rng.uniform(size=(n, 3)) < p_true).astype(int)
    y_model = 1.0 / (1.0 + np.exp(-0.3 * z))  # compressão → underconfidence
    return y_true, y_model


# 1 -------------------------------------------------------------------------
def test_known_metrics_perfect_and_random():
    y_true, _ = _synthetic_multilabel()
    perfect = y_true.astype(float)
    res = evaluate(
        y_true,
        perfect,
        task_profile="pretrain_scp_ecg_multilabel",
        split_id="synthetic",
        ontology_version="v3",
        n_bootstrap=0,
    )
    assert res.metrics["metrics"]["macro_auroc"] == pytest.approx(1.0)
    assert res.metrics["metrics"]["macro_pr_auc"] == pytest.approx(1.0)

    rng = np.random.RandomState(0)
    random_scores = rng.uniform(size=y_true.shape)
    res_rnd = evaluate(
        y_true,
        random_scores,
        task_profile="pretrain_scp_ecg_multilabel",
        split_id="synthetic",
        ontology_version="v3",
        n_bootstrap=0,
    )
    assert res_rnd.metrics["metrics"]["macro_auroc"] == pytest.approx(0.5, abs=0.05)


# 2 -------------------------------------------------------------------------
def test_auroc_invariant_to_temperature():
    y_true, y_score = _synthetic_multilabel()
    for temp in (0.3741, 0.5, 1.0, 2.5):
        y_cal = apply_temperature(y_score, temp)
        for j, name in enumerate(CLASSES):
            before = roc_auc_score(y_true[:, j], y_score[:, j])
            after = roc_auc_score(y_true[:, j], y_cal[:, j])
            assert after == pytest.approx(before, abs=1e-6), name


# 3 -------------------------------------------------------------------------
def test_ece_decreases_after_temperature_when_calibration_improves():
    y_true, y_model = _underconfident_pair()
    temp = fit_temperature_multilabel(y_true, y_model)
    assert 0.2 < temp < 0.45  # recupera a compressão 0.3 conhecida

    res_pre = evaluate(
        y_true, y_model,
        task_profile="pretrain_scp_ecg_multilabel", split_id="synthetic",
        ontology_version="v3", class_names=["c0", "c1", "c2"], n_bootstrap=0,
    )
    res_post = evaluate(
        y_true, y_model,
        task_profile="pretrain_scp_ecg_multilabel", split_id="synthetic",
        ontology_version="v3", class_names=["c0", "c1", "c2"],
        temperature=temp, n_bootstrap=0,
    )
    ece_pre = res_pre.metrics["metrics"]["ece_pre_calibration"]
    ece_post = res_post.metrics["metrics"]["ece_post_calibration"]
    assert ece_post < ece_pre


# 4 -------------------------------------------------------------------------
def test_temperature_below_one_sharpens_probabilities():
    _, y_model = _underconfident_pair()
    y_sharp = apply_temperature(y_model, 0.4)
    spread_before = np.mean(np.abs(y_model - 0.5))
    spread_after = np.mean(np.abs(y_sharp - 0.5))
    assert spread_after > spread_before


# 5 -------------------------------------------------------------------------
def test_macro_not_dominated_by_majority_class():
    n = 4000
    rng = np.random.RandomState(3)
    y_true = np.zeros((n, 2), dtype=int)
    y_true[: int(0.9 * n), 0] = 1  # classe 0: 90% prevalência
    y_true[int(0.9 * n):, 1] = 1  # classe 1: 10% prevalência (minoritária)
    y_score = np.zeros((n, 2))
    y_score[:, 0] = y_true[:, 0] + rng.normal(0, 0.01, n)  # perfeita
    y_score[:, 1] = rng.uniform(size=n)  # aleatória (auroc ~0.5)

    res = evaluate(
        y_true, y_score,
        task_profile="pretrain_scp_ecg_multilabel", split_id="synthetic",
        ontology_version="v3", class_names=["maj", "min"], n_bootstrap=0,
    )
    macro = res.metrics["metrics"]["macro_auroc"]
    weighted = roc_auc_score(y_true, y_score, average="weighted")
    assert 0.7 < macro < 0.8  # (1.0 + ~0.5) / 2 — não é arrastada pela majoritária
    assert weighted > 0.9  # weighted, sim, é dominada


# 6 -------------------------------------------------------------------------
def test_threshold_tuning_does_not_use_test_set():
    y_cal, p_cal = _synthetic_multilabel(n=3000, seed=21)
    policy = ThresholdPolicy(name="max_f1_per_class")
    thresholds_a = fit_thresholds(y_cal, p_cal, CLASSES, policy)

    # teste completamente diferente: thresholds congelados não podem mudar
    y_test_a, p_test_a = _synthetic_multilabel(n=3000, seed=99)
    y_test_b, p_test_b = _synthetic_multilabel(n=3000, seed=123)
    decisions_a = apply_thresholds(p_test_a, thresholds_a, CLASSES)
    decisions_b = apply_thresholds(p_test_b, thresholds_a, CLASSES)
    thresholds_b = fit_thresholds(y_cal, p_cal, CLASSES, policy)  # refit no MESMO cal
    assert thresholds_a == thresholds_b
    assert decisions_a.shape == y_test_a.shape == decisions_b.shape == y_test_b.shape


# 7 -------------------------------------------------------------------------
def test_metrics_json_schema_valid(tmp_path):
    y_true, y_score = _synthetic_multilabel()
    res = evaluate(
        y_true, y_score,
        task_profile="pretrain_scp_ecg_multilabel", split_id="synthetic",
        ontology_version="v3", temperature=0.3741, n_bootstrap=0,
    )
    parsed = MetricsJson.model_validate(res.metrics)
    assert parsed.schema_version == "2.0"
    assert parsed.evaluator_version == "v2.0"
    assert parsed.metrics.temperature == pytest.approx(0.3741)

    write_artifacts(res, tmp_path)
    on_disk = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    reparsed = MetricsJson.model_validate(on_disk)
    assert reparsed.schema_version == "2.0"
    for artifact in (
        "metrics_per_class.json", "calibration.json", "thresholds.json",
        "reliability.json", "confidence_intervals.json",
    ):
        assert (tmp_path / artifact).exists(), artifact


# 8 -------------------------------------------------------------------------
def test_graceful_failure_on_split_task_mismatch():
    y_true_ml, y_score_ml = _synthetic_multilabel()
    with pytest.raises(ValueError, match="beat_classification_aami espera"):
        evaluate(
            y_true_ml, y_score_ml,
            task_profile="beat_classification_aami", split_id="x",
            ontology_version="v3", n_bootstrap=0,
        )
    y_labels = np.random.RandomState(5).randint(0, 5, size=100)
    y_prob = np.random.RandomState(5).uniform(size=(100, 5))
    with pytest.raises(ValueError, match="multi-label espera"):
        evaluate(
            y_labels, y_prob,
            task_profile="pretrain_scp_ecg_multilabel", split_id="x",
            ontology_version="v3", n_bootstrap=0,
        )


# 9 -------------------------------------------------------------------------
def test_comparability_contract_marks_non_comparable():
    base = ComparabilityContract(
        task_profile="pretrain_scp_ecg_multilabel",
        split_id="chapman-record-disjoint-val0.1-seed13",
        ontology_version="v3",
    )
    same = ComparabilityContract(
        task_profile="pretrain_scp_ecg_multilabel",
        split_id="chapman-record-disjoint-val0.1-seed13",
        ontology_version="v3",
    )
    assert check_comparable(base, same).status == "COMPARABLE"

    other_version = base.model_copy(update={"evaluator_version": "v1.0"})
    result = check_comparable(base, other_version)
    assert result.status == "NON_COMPARABLE"
    assert any("evaluator_version" in r for r in result.reasons)

    other_split = base.model_copy(update={"split_id": "outro-split"})
    result = check_comparable(base, other_split)
    assert result.status == "NON_COMPARABLE"
    assert any("split_id" in r for r in result.reasons)


# 10 ------------------------------------------------------------------------
def test_ece_norm0_stratified_metric():
    """ECE do estrato NORM=0 deve expor descalibração escondida pela marginal."""
    n = 8000
    rng = np.random.RandomState(11)
    y_true = np.zeros((n, 2), dtype=int)
    y_true[: int(0.7 * n), 0] = 1  # NORM=1 em 70%
    y_true[int(0.7 * n):, 1] = 1  # CD=1 apenas no estrato NORM=0 (30%)
    y_score = np.zeros((n, 2))
    # estrato NORM=1: bem calibrado (probs altas p/ NORM=1, baixas p/ CD=0)
    y_score[: int(0.7 * n), 0] = rng.beta(20, 1, int(0.7 * n))
    y_score[: int(0.7 * n), 1] = rng.beta(1, 20, int(0.7 * n))
    # estrato NORM=0: mal calibrado (confiança alta sistemática em NORM)
    y_score[int(0.7 * n):, 0] = rng.beta(8, 2, n - int(0.7 * n))  # prediz NORM alto, y=0
    y_score[int(0.7 * n):, 1] = rng.beta(8, 2, n - int(0.7 * n))  # idem em CD, y=1

    res = evaluate(
        y_true, y_score,
        task_profile="pretrain_scp_ecg_multilabel", split_id="synthetic",
        ontology_version="v3", class_names=["NORM", "CD"],
        temperature=1.0, n_bootstrap=0,
    )
    m = res.metrics["metrics"]
    assert m["ece_post_calibration_norm0"] is not None
    assert m["ece_post_calibration_norm0"] > 2.0 * m["ece_post_calibration"]
    strat = res.calibration["stratified_norm0"]
    assert strat["n_samples"] == n - int(0.7 * n)
    assert set(strat["per_class"].keys()) == {"NORM", "CD"}


# 11 ------------------------------------------------------------------------
def test_per_class_confidence_intervals(tmp_path):
    y_true, y_score = _synthetic_multilabel()
    res = evaluate(
        y_true, y_score,
        task_profile="pretrain_scp_ecg_multilabel", split_id="synthetic",
        ontology_version="v3", n_bootstrap=50,
    )
    ci = res.confidence_intervals
    assert "per_class_pr_auc" in ci and "per_class_auroc" in ci
    for name in CLASSES:
        for target in ("per_class_pr_auc", "per_class_auroc"):
            entry = ci[target][name]
            assert entry is not None, (target, name)
            lo, hi = entry["ci_95"]
            assert 0.0 <= lo <= hi <= 1.0
    write_artifacts(res, tmp_path)
    on_disk = json.loads((tmp_path / "confidence_intervals.json").read_text(encoding="utf-8"))
    assert set(on_disk["per_class_pr_auc"].keys()) == set(CLASSES)


# 12 ------------------------------------------------------------------------
def test_schema_backward_compatible_without_norm0_field():
    """Artefatos schema 2.0 sem o campo novo (ex.: T9.3) continuam validando."""
    y_true, y_score = _synthetic_multilabel()
    res = evaluate(
        y_true, y_score,
        task_profile="pretrain_scp_ecg_multilabel", split_id="synthetic",
        ontology_version="v3", n_bootstrap=0,
    )
    legacy_like = res.metrics.copy()
    legacy_like["metrics"] = {
        k: v for k, v in legacy_like["metrics"].items()
        if k != "ece_post_calibration_norm0"
    }
    parsed = MetricsJson.model_validate(legacy_like)
    assert parsed.metrics.ece_post_calibration_norm0 is None


# 13 (G6) -------------------------------------------------------------------
def test_reconcile_legacy_nested_schema():
    """G6: parser lê schema aninhado do A0 novo (temperature_scaling + before/after)."""
    legacy = {
        "before": {"macro": {"ece": 0.025, "brier": 0.18, "nll": 0.39}},
        "after": {"macro": {"ece": 0.020, "brier": 0.17, "nll": 0.38}},
        "temperature_scaling": {
            "temperature": 0.97,
            "nll_before": 0.39,
            "nll_after": 0.38,
        },
    }
    result = _extract_legacy_calibration(legacy)
    assert result["temperature"] == pytest.approx(0.97)
    assert result["ece_before"] == pytest.approx(0.025)
    assert result["ece_after"] == pytest.approx(0.020)
    assert result["nll_before"] == pytest.approx(0.39)
    assert result["nll_after"] == pytest.approx(0.38)
    assert result["brier_before"] == pytest.approx(0.18)
    assert result["brier_after"] == pytest.approx(0.17)


# 14 (G6) -------------------------------------------------------------------
def test_reconcile_legacy_flat_schema_still_works():
    """Backward compat: schema plano (contrato T1) continua funcionando."""
    flat = {
        "temperature": 0.3741,
        "ece_before": 0.1508,
        "ece_after": 0.0152,
        "nll_before": 0.4317,
        "nll_after": 0.3417,
    }
    result = _extract_legacy_calibration(flat)
    assert result["temperature"] == pytest.approx(0.3741)
    assert result["ece_before"] == pytest.approx(0.1508)
    assert result["ece_after"] == pytest.approx(0.0152)
    assert result["nll_before"] == pytest.approx(0.4317)
    assert result["nll_after"] == pytest.approx(0.3417)


# 15 (G6) -------------------------------------------------------------------
A0_NOVO_CAL = Path("experiments/20260729_042301_pretrain_chapman/calibration.json")


@pytest.mark.skipif(not A0_NOVO_CAL.exists(), reason="arquivo legado não versionado")
def test_reconcile_legacy_real_a0_novo_file():
    """Integração: calibration.json real do A0 novo (aninhado, pós-T em
    temperature_scaling.calibration_after) é extraído sem None nos campos-chave."""
    import json as _json

    legacy = _json.loads(A0_NOVO_CAL.read_text(encoding="utf-8"))
    result = _extract_legacy_calibration(legacy)
    assert result["temperature"] == pytest.approx(0.9130257150830182)
    assert result["ece_before"] == pytest.approx(0.025059479457722123)
    assert result["ece_after"] == pytest.approx(0.02037379373237904)
    assert result["nll_before"] is not None
    assert result["nll_after"] is not None
