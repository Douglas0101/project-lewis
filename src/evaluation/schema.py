"""Schemas e contrato de comparabilidade do avaliador canônico (pydantic v2).

Espelha docs/ml_protocol_v2.md §7 (protocolo de comparação) e §10 (schema
mínimo do ``metrics.json``). ``schema_version`` é ``"2.0"`` — artefatos de
outro schema não validam aqui.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

EVALUATOR_VERSION = "v2.0"
SCHEMA_VERSION = "2.0"

TASK_PROFILES = ("pretrain_scp_ecg_multilabel", "beat_classification_aami")

#: Campos do contrato de comparabilidade (protocolo v2 §7) — todos devem
#: coincidir para que duas avaliações sejam comparáveis.
COMPARABILITY_FIELDS = (
    "evaluator_version",
    "task_profile",
    "split_id",
    "ontology_version",
    "n_bins",
    "calibration_method",
    "threshold_policy",
    "preprocessing_version",
)


class ComparabilityContract(BaseModel):
    """Identidade protocolar de uma avaliação."""

    evaluator_version: str = EVALUATOR_VERSION
    task_profile: str
    split_id: str
    ontology_version: str
    n_bins: int = 15
    calibration_method: str = "temperature_scaling"
    threshold_policy: str = "max_f1_per_class"
    preprocessing_version: str = "v1.0"


class ComparabilityResult(BaseModel):
    """Veredito de comparabilidade entre duas avaliações."""

    status: Literal["COMPARABLE", "NON_COMPARABLE"]
    reasons: list[str] = Field(default_factory=list)


def check_comparable(a: ComparabilityContract, b: ComparabilityContract) -> ComparabilityResult:
    """COMPARABLE somente se TODOS os campos do contrato coincidirem."""
    reasons = [
        f"{field}: {getattr(a, field)!r} != {getattr(b, field)!r}"
        for field in COMPARABILITY_FIELDS
        if getattr(a, field) != getattr(b, field)
    ]
    if reasons:
        return ComparabilityResult(status="NON_COMPARABLE", reasons=reasons)
    return ComparabilityResult(status="COMPARABLE", reasons=[])


class MetricsBlock(BaseModel):
    """Bloco de métricas agregadas do schema 2.0 (nulos explícitos)."""

    macro_pr_auc: Optional[float] = None
    macro_auroc: Optional[float] = None
    macro_f1_at_0_5: Optional[float] = None
    macro_f1_tuned: Optional[float] = None
    bce: Optional[float] = None
    bce_post_temperature: Optional[float] = None
    nll: Optional[float] = None
    nll_post_temperature: Optional[float] = None
    brier_mean: Optional[float] = None
    ece_pre_calibration: Optional[float] = None
    ece_post_calibration: Optional[float] = None
    mce_post_calibration: Optional[float] = None
    temperature: Optional[float] = None


class MetricsJson(BaseModel):
    """``metrics.json`` schema 2.0 (docs/ml_protocol_v2.md §10)."""

    schema_version: Literal["2.0"] = SCHEMA_VERSION  # type: ignore[assignment]
    run_id: str
    task_profile: str
    split_id: str
    ontology_version: str
    evaluator_version: str = EVALUATOR_VERSION
    n_samples: int
    protocol_status: Literal["PROSPECTIVE", "RETROSPECTIVE", "FROZEN_PARAMS"]
    metrics: MetricsBlock
    per_class: dict = Field(default_factory=dict)
    thresholds: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)
