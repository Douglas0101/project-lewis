"""Validação estrutural dos configs ML Protocol v2 (configs/ml_protocol/v2/).

Os YAMLs são contratos normativos (T9.4) — este teste garante que as chaves
obrigatórias do protocolo existem e que as regras duras estão declaradas
(SMOTE só em treino, thresholds fora do teste, calibração obrigatória, ES por
métrica equalizada). Nenhum treino é executado aqui.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path("configs/ml_protocol/v2")

TASK_PROFILES = (
    "pretrain_scp_ecg_multilabel",
    "beat_classification_aami",
    "rhythm_afib_afl",
)

FORBIDDEN_ALWAYS = {"smote_on_validation", "smote_on_test", "test_threshold_tuning"}


def _load(name: str) -> dict:
    path = CONFIG_DIR / name
    assert path.exists(), f"config ausente: {path}"
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.mark.parametrize("filename", [f"{p}.yaml" for p in TASK_PROFILES])
def test_task_profile_structure(filename: str):
    cfg = _load(filename)
    assert cfg["schema_version"] == 2.0
    assert cfg["task_profile"] in TASK_PROFILES
    assert cfg["ontology_version"] == "v3"

    # métricas primárias declaradas e early stopping por métrica equalizada
    assert cfg["metrics"]["primary"], "métricas primárias ausentes"
    es_metric = cfg["training"]["early_stopping"]["metric"]
    assert es_metric in ("val_macro_pr_auc", "val_macro_f1"), es_metric
    assert cfg["training"]["early_stopping"]["mode"] == "max"

    # regras duras: SMOTE/threshold fora de val/test
    forbidden = set(cfg["training"].get("forbidden", []))
    assert FORBIDDEN_ALWAYS <= forbidden, forbidden

    # calibração obrigatória com split separado e n_bins=15
    cal = cfg["calibration"]
    assert cal["required"] is True
    assert cal["n_bins"] == 15
    assert cal["split"] == "calibration"

    # thresholds ajustados em calibration, aplicados congelados
    thr = cfg["thresholding"]
    assert thr["fit_split"] == "calibration"
    assert thr["apply_to_test"] == "frozen"


def test_pretrain_profile_specifics():
    cfg = _load("pretrain_scp_ecg_multilabel.yaml")
    assert cfg["labels"]["classes"] == ["NORM", "CD", "MI", "HYP", "STTC"]
    assert cfg["labels"]["rejection_class"] == "Q_OR_UNKNOWN"
    assert cfg["data"]["split_id"] == "chapman-record-disjoint-paired-v2"
    assert "ece_post_calibration_norm0" in cfg["metrics"]["secondary"]
    assert cfg["training"]["early_stopping"]["metric"] == "val_macro_pr_auc"


def test_teacher_configs_have_budget_and_no_tflm():
    for name, cells in (("teacher_resnet1d.yaml", None), ("teacher_inception1d.yaml", "D4")):
        cfg = _load(name)
        for arch in cfg["archetypes"].values():
            assert arch["params_target"] >= 500_000
            assert arch["normalization"] == "GroupNorm"
        assert cfg["budget_test"]["required"] is True
        assert cfg["evaluation"]["offline_only"] is True
        assert cfg["training"]["early_stopping"]["metric"] == "val_macro_pr_auc"


def test_distillation_kd_normative_form():
    cfg = _load("distillation_kd.yaml")
    kd = cfg["loss"]["kd_term"]
    assert "BCE_with_logits" in kd and "sigmoid" in kd, kd
    assert "softmax(z" not in kd, "softmax-KD não é a forma normativa (multi-label)"
    assert cfg["loss"]["alpha_sweep"] == [0.5, 0.7, 0.9]
    assert cfg["loss"]["tau_sweep"] == [2.0, 4.0, 6.0]
    assert cfg["training"]["teacher_frozen"] is True
    assert cfg["training"]["student_init"]["from_scratch"] == "forbidden"
    assert cfg["evaluation"]["protocol_status"] == "PROSPECTIVE"
    assert cfg["evaluation"]["calibration"]["split"] == "calibration"


def test_split_paired_v2_is_spec_only():
    cfg = _load("split_paired_v2.yaml")
    assert cfg["split_id"] == "chapman-record-disjoint-paired-v2"
    partitions = cfg["partitions"]
    total = sum(p["ratio"] for p in partitions.values())
    assert total == pytest.approx(1.0)
    assert "calibration" in partitions and "test" in partitions
    assert cfg["immutability"] is True
    assert cfg["generation"] == "pending_governance_T10.3"
    assert cfg["rules"]["seed"] == 13
