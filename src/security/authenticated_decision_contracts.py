"""Strict contracts for authenticated research decisions in shadow mode."""

from __future__ import annotations

import base64
import binascii
import re
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

IN_TOTO_STATEMENT_V1 = "https://in-toto.io/Statement/v1"
RESEARCH_DECISION_PREDICATE_V1 = "https://project-lewis.dev/attestations/research-decision/v1"
DSSE_IN_TOTO_PAYLOAD_TYPE = "application/vnd.in-toto+json"
SIGNED_DECISION_SET_V1 = "project-lewis.dev/signed-decision-set/v1"
SIGSTORE_BUNDLE_MEDIA_TYPE_V03 = "application/vnd.dev.sigstore.bundle.v0.3+json"
POLICY_ID_V1 = "project-lewis/research-decision/v1"
POLICY_VERSION_V1 = "1.0.0"

NonEmptyText = Annotated[str, StringConstraints(min_length=1, max_length=4096)]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=256)]
AuthorizationRecordId = Annotated[
    str,
    StringConstraints(pattern=r"^authz_[a-z0-9]{16,64}$"),
]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Base64Text = Annotated[str, StringConstraints(min_length=1, max_length=6_000_000)]
UtcText = Annotated[
    str,
    StringConstraints(
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$",
    ),
]
SequenceNumber = Annotated[StrictInt, Field(ge=0)]


class StrictFrozenModel(BaseModel):
    """Base contract: no coercion, no extension fields, and immutable values."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        populate_by_name=False,
    )


class DecisionOutcome(StrEnum):
    """Only states permitted by the v1 shadow policy."""

    REJECTED_AUTHENTICATED = "REJECTED_AUTHENTICATED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    APPROVED_FOR_AUDIT = "APPROVED_FOR_AUDIT"


class SignerRole(StrEnum):
    """Locally authorized roles; submitted bundles never assign these roles."""

    EVIDENCE_BOT = "EVIDENCE_BOT"
    SCIENTIFIC_APPROVER = "SCIENTIFIC_APPROVER"


class TrustedTimeSource(StrEnum):
    """Authenticated time sources accepted from a future Sigstore backend."""

    REKOR = "REKOR"
    RFC3161 = "RFC3161"
    OTHER_RATIFIED = "OTHER_RATIFIED"


class ReasonCode(StrEnum):
    """Controlled verifier reason vocabulary."""

    ALL_MANDATORY_GATES_SUCCESS = "ALL_MANDATORY_GATES_PASS"
    INPUT_INVALID = "INPUT_INVALID"
    DUPLICATE_JSON_KEY = "DUPLICATE_JSON_KEY"
    NON_FINITE_JSON = "NON_FINITE_JSON"
    AMBIGUOUS_JSON_TYPE = "AMBIGUOUS_JSON_TYPE"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    BACKEND_ERROR = "BACKEND_ERROR"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    BUNDLE_HASH_MISMATCH = "BUNDLE_HASH_MISMATCH"
    DSSE_INVALID = "DSSE_INVALID"
    PAYLOAD_MISMATCH = "PAYLOAD_MISMATCH"
    STATEMENT_INVALID = "STATEMENT_INVALID"
    IDENTITY_UNAUTHORIZED = "IDENTITY_UNAUTHORIZED"
    QUORUM_NOT_MET = "QUORUM_NOT_MET"
    SEPARATION_OF_DUTIES_FAILED = "SEPARATION_OF_DUTIES_FAILED"
    TRUSTED_TIME_INVALID = "TRUSTED_TIME_INVALID"
    VALIDITY_INVALID = "VALIDITY_INVALID"
    POLICY_HASH_MISMATCH = "POLICY_HASH_MISMATCH"
    SUBJECT_MISSING = "SUBJECT_MISSING"
    SUBJECT_HASH_MISMATCH = "SUBJECT_HASH_MISMATCH"
    EVIDENCE_MISSING = "EVIDENCE_MISSING"
    EVIDENCE_HASH_MISMATCH = "EVIDENCE_HASH_MISMATCH"
    EVIDENCE_EVALUATOR_UNAVAILABLE = "EVIDENCE_EVALUATOR_UNAVAILABLE"
    EVIDENCE_GATE_FAILED = "EVIDENCE_GATE_FAILED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    EVIDENCE_AMBIGUOUS = "EVIDENCE_AMBIGUOUS"
    WAIVER_PRESENT = "WAIVER_PRESENT"
    REPLAY_STORE_UNAVAILABLE = "REPLAY_STORE_UNAVAILABLE"
    REPLAY_DETECTED = "REPLAY_DETECTED"
    SEQUENCE_REPLAYED = "SEQUENCE_REPLAYED"


class MandatoryGateName(StrEnum):
    """Approved audit-entry gates; proposed research thresholds are excluded."""

    PREFLIGHT_SUCCEEDED = "PREFLIGHT_PASS"
    E06_5_SMOKE_SUCCEEDED = "E06_5_SMOKE_PASS"
    AUDIT_MATRIX_FROZEN = "AUDIT_MATRIX_FROZEN"
    AUDIT_NOT_STARTED = "AUDIT_NOT_STARTED"
    SOURCE_IDENTITY_BOUND = "SOURCE_IDENTITY_BOUND"
    ARTIFACT_INTEGRITY = "ARTIFACT_INTEGRITY"
    NO_LEAKAGE = "NO_LEAKAGE"
    OUTER_TEST_ISOLATED = "OUTER_TEST_ISOLATED"
    FOCUSED_TESTS_SUCCEEDED = "FOCUSED_TESTS_PASS"


class EvidenceGateStatus(StrEnum):
    """Derived status returned by a trusted evidence evaluator."""

    SUCCEEDED = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT = "INSUFFICIENT"
    AMBIGUOUS = "AMBIGUOUS"


class DsseSignature(StrictFrozenModel):
    """One DSSE signature entry in the Project-Lewis Sigstore profile."""

    keyid: Annotated[str, StringConstraints(max_length=256)] = ""
    sig: Base64Text

    @field_validator("sig")
    @classmethod
    def validate_signature_base64(cls, value: str) -> str:
        _decode_base64(value, urlsafe=False, field_name="DSSE signature")
        return value


class DsseEnvelope(StrictFrozenModel):
    """DSSE envelope whose exact bytes were authenticated by the backend."""

    payload_type: Literal["application/vnd.in-toto+json"] = Field(alias="payloadType")
    payload: Base64Text
    signatures: tuple[DsseSignature, ...] = Field(min_length=1, max_length=1)

    @field_validator("payload")
    @classmethod
    def validate_payload_base64(cls, value: str) -> str:
        _decode_base64(value, urlsafe=False, field_name="DSSE payload")
        return value


class Sha256Digest(StrictFrozenModel):
    """Project profile intentionally accepts SHA-256 only."""

    sha256: Sha256Hex


class InTotoSubject(StrictFrozenModel):
    """Digest-addressed subject in an in-toto Statement."""

    name: ShortText
    digest: Sha256Digest


class RequesterClaim(StrictFrozenModel):
    """Local authorization record for the requester, not a raw OIDC identity."""

    authorization_id: AuthorizationRecordId = Field(alias="authorizationId")


class QuorumClaim(StrictFrozenModel):
    """Authenticated quorum claim; local policy remains authoritative."""

    threshold: Annotated[StrictInt, Field(ge=2, le=8)]
    required_roles: tuple[SignerRole, ...] = Field(
        alias="requiredRoles",
        min_length=2,
        max_length=8,
    )

    @model_validator(mode="after")
    def validate_unique_roles(self) -> QuorumClaim:
        if len(set(self.required_roles)) != len(self.required_roles):
            raise ValueError("requiredRoles must be unique")
        return self


class PolicyClaim(StrictFrozenModel):
    """Signed binding to the locally frozen policy document."""

    policy_id: Literal["project-lewis/research-decision/v1"] = Field(alias="policyId")
    version: Literal["1.0.0"]
    sha256: Sha256Hex
    quorum: QuorumClaim


class EvidenceReference(StrictFrozenModel):
    """Digest-only evidence reference; its path comes from trusted context."""

    name: ShortText
    sha256: Sha256Hex
    media_type: ShortText = Field(alias="mediaType")


class WaiverClaim(StrictFrozenModel):
    """Any waiver forces human review in policy v1."""

    code: ShortText
    justification: NonEmptyText
    evidence_name: ShortText | None = Field(default=None, alias="evidenceName")


class ValidityClaim(StrictFrozenModel):
    """UTC validity window bound into the signed predicate."""

    issued_on: UtcText = Field(alias="issuedOn")
    not_before: UtcText = Field(alias="notBefore")
    expires_on: UtcText = Field(alias="expiresOn")


class ResearchDecisionPredicate(StrictFrozenModel):
    """Project-Lewis `research-decision/v1` predicate."""

    decision_id: ShortText = Field(alias="decisionId")
    nonce: ShortText
    sequence: SequenceNumber
    scope: Literal["E06_5_AUDIT"]
    requester: RequesterClaim
    policy: PolicyClaim
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1, max_length=32)
    waivers: tuple[WaiverClaim, ...] = Field(default=(), max_length=16)
    validity: ValidityClaim
    claimed_reason_codes: tuple[ReasonCode, ...] = Field(
        alias="claimedReasonCodes",
        min_length=1,
        max_length=32,
    )

    @field_validator("decision_id")
    @classmethod
    def validate_decision_id(cls, value: str) -> str:
        try:
            parsed = UUID(value)
        except ValueError as error:
            raise ValueError("decisionId must be a UUID") from error
        if parsed.version not in {4, 7} or str(parsed) != value:
            raise ValueError("decisionId must be canonical UUIDv4 or UUIDv7")
        return value

    @field_validator("nonce")
    @classmethod
    def validate_nonce(cls, value: str) -> str:
        decoded = _decode_base64(value, urlsafe=True, field_name="nonce")
        if len(decoded) != 32 or "=" in value:
            raise ValueError("nonce must be 256-bit base64url without padding")
        return value

    @model_validator(mode="after")
    def validate_unique_collections(self) -> ResearchDecisionPredicate:
        evidence_names = [item.name for item in self.evidence]
        if len(set(evidence_names)) != len(evidence_names):
            raise ValueError("evidence names must be unique")
        if len(set(self.claimed_reason_codes)) != len(self.claimed_reason_codes):
            raise ValueError("claimedReasonCodes must be unique")
        return self


class InTotoStatement(StrictFrozenModel):
    """Strict Project-Lewis profile of in-toto Statement v1."""

    statement_type: Literal["https://in-toto.io/Statement/v1"] = Field(alias="_type")
    subject: tuple[InTotoSubject, ...] = Field(min_length=1, max_length=16)
    predicate_type: Literal["https://project-lewis.dev/attestations/research-decision/v1"] = Field(
        alias="predicateType"
    )
    predicate: ResearchDecisionPredicate

    @model_validator(mode="after")
    def validate_unique_subjects(self) -> InTotoStatement:
        names = [item.name for item in self.subject]
        if len(set(names)) != len(names):
            raise ValueError("subject names must be unique")
        return self


class SigstoreBundleDescriptor(StrictFrozenModel):
    """Opaque Sigstore bundle bytes plus an independently checked digest."""

    bundle_media_type: Literal["application/vnd.dev.sigstore.bundle.v0.3+json"] = Field(
        alias="bundleMediaType"
    )
    bundle_base64: Base64Text = Field(alias="bundleBase64")
    bundle_sha256: Sha256Hex = Field(alias="bundleSha256")

    @field_validator("bundle_base64")
    @classmethod
    def validate_bundle_base64(cls, value: str) -> str:
        _decode_base64(value, urlsafe=False, field_name="Sigstore bundle")
        return value


class SignedDecisionSet(StrictFrozenModel):
    """Collection of independent Sigstore bundles over byte-identical payloads."""

    format: Literal["project-lewis.dev/signed-decision-set/v1"]
    bundles: tuple[SigstoreBundleDescriptor, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_unique_bundle_hashes(self) -> SignedDecisionSet:
        hashes = [item.bundle_sha256 for item in self.bundles]
        if len(set(hashes)) != len(hashes):
            raise ValueError("bundle hashes must be unique")
        return self


class AuthorizedPrincipal(StrictFrozenModel):
    """Locally ratified exact OIDC principal and its non-PII authorization record."""

    authorization_id: AuthorizationRecordId = Field(alias="authorizationId")
    role: SignerRole
    issuer: NonEmptyText
    identity: NonEmptyText


class ShadowVerificationPolicy(StrictFrozenModel):
    """Local authoritative settings; no production identity defaults exist."""

    policy_id: Literal["project-lewis/research-decision/v1"] = Field(alias="policyId")
    version: Literal["1.0.0"]
    sha256: Sha256Hex
    authorized_principals: tuple[AuthorizedPrincipal, ...] = Field(
        alias="authorizedPrincipals",
        min_length=2,
        max_length=8,
    )
    allowed_requester_ids: tuple[AuthorizationRecordId, ...] = Field(
        alias="allowedRequesterIds",
        min_length=1,
        max_length=16,
    )
    required_roles: tuple[SignerRole, ...] = Field(
        alias="requiredRoles",
        min_length=2,
        max_length=2,
    )
    quorum_threshold: Literal[2] = Field(alias="quorumThreshold")
    required_subject_names: tuple[ShortText, ...] = Field(
        alias="requiredSubjectNames",
        min_length=1,
        max_length=16,
    )
    required_evidence_names: tuple[ShortText, ...] = Field(
        alias="requiredEvidenceNames",
        min_length=1,
        max_length=32,
    )
    required_gate_names: tuple[MandatoryGateName, ...] = Field(
        alias="requiredGateNames",
        min_length=1,
        max_length=32,
    )
    allowed_trusted_time_sources: tuple[TrustedTimeSource, ...] = Field(
        alias="allowedTrustedTimeSources",
        min_length=1,
        max_length=3,
    )
    max_clock_skew_seconds: Annotated[StrictInt, Field(ge=0, le=300)] = Field(
        default=300,
        alias="maxClockSkewSeconds",
    )
    max_decision_lifetime_seconds: Annotated[
        StrictInt,
        Field(ge=60, le=86_400),
    ] = Field(default=86_400, alias="maxDecisionLifetimeSeconds")

    @model_validator(mode="after")
    def validate_authorization_policy(self) -> ShadowVerificationPolicy:
        expected_roles = {SignerRole.EVIDENCE_BOT, SignerRole.SCIENTIFIC_APPROVER}
        if set(self.required_roles) != expected_roles:
            raise ValueError("requiredRoles must be evidence bot + scientific approver")
        authorization_ids = [item.authorization_id for item in self.authorized_principals]
        principals = [(item.issuer, item.identity) for item in self.authorized_principals]
        if len(set(authorization_ids)) != len(authorization_ids):
            raise ValueError("authorizationId values must be unique")
        if len(set(principals)) != len(principals):
            raise ValueError("OIDC principals must be unique")
        for values, name in (
            (self.allowed_requester_ids, "allowedRequesterIds"),
            (self.required_subject_names, "requiredSubjectNames"),
            (self.required_evidence_names, "requiredEvidenceNames"),
            (self.required_gate_names, "requiredGateNames"),
            (self.allowed_trusted_time_sources, "allowedTrustedTimeSources"),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
        return self


class DerivedGateResult(StrictFrozenModel):
    """One gate rederived by a trusted evidence evaluator."""

    name: MandatoryGateName
    status: EvidenceGateStatus


class EvidenceAssessment(StrictFrozenModel):
    """Semantic evidence result returned by an injected evaluator."""

    gates: tuple[DerivedGateResult, ...] = Field(min_length=1, max_length=32)
    reason_codes: tuple[ReasonCode, ...] = Field(
        alias="reasonCodes",
        min_length=1,
        max_length=32,
    )

    @model_validator(mode="after")
    def validate_unique_gates(self) -> EvidenceAssessment:
        names = [item.name for item in self.gates]
        if len(set(names)) != len(names):
            raise ValueError("derived gate names must be unique")
        return self


class VerifiedAuthorizationRecord(StrictFrozenModel):
    """Non-PII signer reference safe for a shadow report."""

    authorization_id: AuthorizationRecordId = Field(alias="authorizationId")
    role: SignerRole


class ShadowVerificationReport(StrictFrozenModel):
    """Non-operational result of the v1 shadow verifier."""

    outcome: DecisionOutcome
    shadow: Literal[True] = True
    operational: Literal[False] = False
    decision_id: str | None = Field(default=None, alias="decisionId")
    reason_codes: tuple[ReasonCode, ...] = Field(alias="reasonCodes", min_length=1)
    authorization_records: tuple[VerifiedAuthorizationRecord, ...] = Field(
        default=(),
        alias="authorizationRecords",
    )
    cryptographic_backend: ShortText = Field(alias="cryptographicBackend")
    evidence_evaluator: ShortText = Field(alias="evidenceEvaluator")
    policy_sha256: Sha256Hex | None = Field(default=None, alias="policySha256")
    evaluated_at: UtcText = Field(alias="evaluatedAt")
    notes: tuple[NonEmptyText, ...] = ()


def decode_standard_base64(value: str, *, field_name: str) -> bytes:
    """Decode strict standard base64 for verifier use."""
    return _decode_base64(value, urlsafe=False, field_name=field_name)


def decode_urlsafe_base64(value: str, *, field_name: str) -> bytes:
    """Decode strict unpadded base64url for verifier use."""
    return _decode_base64(value, urlsafe=True, field_name=field_name)


def _decode_base64(value: str, *, urlsafe: bool, field_name: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    try:
        if urlsafe:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
                raise ValueError(f"{field_name} is not unpadded base64url")
            padding = "=" * ((4 - len(value) % 4) % 4)
            return base64.b64decode(value + padding, altchars=b"-_", validate=True)
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"{field_name} is not valid base64") from error
