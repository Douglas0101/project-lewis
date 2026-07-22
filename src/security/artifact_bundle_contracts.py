"""Contratos estritos para attestation de bundles de artefatos (v2, shadow).

Alinhado a ``docs/rebuild_spec/09_authenticated_attestation_schema.md`` e ao
idiom v1 (``authenticated_decision_contracts.py``): Pydantic v2 strict,
modelos congelados, ``extra="forbid"``, fail-closed. **Nenhuma verificação
criptográfica de backend** — o backend Sigstore é injetável e, ausente, o
resultado é sempre ``shadow=True, operational=False``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints

IN_TOTO_STATEMENT_V1 = "https://in-toto.io/Statement/v1"
ARTIFACT_BUNDLE_PREDICATE_V2 = "https://project-lewis.dev/attestations/artifact-bundle/v2"
DSSE_IN_TOTO_PAYLOAD_TYPE = "application/vnd.in-toto+json"
POLICY_ID_BUNDLE_V2 = "project-lewis/artifact-bundle/v2"

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyText = Annotated[str, StringConstraints(min_length=1, max_length=4096)]
UtcText = Annotated[
    str,
    StringConstraints(
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$",
    ),
]
SequenceNumber = Annotated[StrictInt, Field(ge=0)]


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        populate_by_name=False,
    )


class BundleOutcome(StrEnum):
    APPROVED_FOR_AUDIT = "APPROVED_FOR_AUDIT"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REJECTED_AUTHENTICATED = "REJECTED_AUTHENTICATED"


class BundlePredicate(StrictFrozenModel):
    """Predicate v2 do Project-Lewis para attestation de bundle de artefatos."""

    bundleDigest: Sha256Hex
    components: dict[str, Sha256Hex]
    trainingRunId: NonEmptyText
    sourceRevision: NonEmptyText
    environmentHash: Sha256Hex
    decisionId: NonEmptyText
    nonce: NonEmptyText
    sequence: SequenceNumber
    validFromUtc: UtcText
    validUntilUtc: UtcText
    policyId: str = POLICY_ID_BUNDLE_V2
    policyVersion: str = "2.0.0"
    shadow: bool = True
    operational: bool = False
    outcome: BundleOutcome
    waivers: list[NonEmptyText] = Field(default_factory=list)


class BundleSubject(StrictFrozenModel):
    name: NonEmptyText
    digest: dict[str, Sha256Hex]


class BundleStatement(StrictFrozenModel):
    """in-toto Statement v1 para bundles de artefatos."""

    type: str = Field(alias="_type", default=IN_TOTO_STATEMENT_V1)
    subject: list[BundleSubject]
    predicateType: str = ARTIFACT_BUNDLE_PREDICATE_V2
    predicate: BundlePredicate


class BundleDsseEnvelope(StrictFrozenModel):
    """Envelope DSSE (payloadType fixo in-toto; assinaturas verificadas pelo
    backend injetável — aqui apenas a forma do envelope)."""

    payloadType: str = DSSE_IN_TOTO_PAYLOAD_TYPE
    payload: NonEmptyText  # base64
    signatures: list[dict[str, str]]
