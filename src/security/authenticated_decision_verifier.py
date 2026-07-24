"""Fail-closed, non-operational verifier for authenticated research decisions."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import NotRequired, Protocol, TypedDict

from pydantic import ValidationError

from src.security.authenticated_decision_contracts import (
    DecisionOutcome,
    DsseEnvelope,
    EvidenceAssessment,
    EvidenceGateStatus,
    InTotoStatement,
    ReasonCode,
    ShadowVerificationPolicy,
    ShadowVerificationReport,
    SignedDecisionSet,
    TrustedTimeSource,
    VerifiedAuthorizationRecord,
    decode_standard_base64,
)
from src.security.authenticated_decision_json import (
    MAX_DSSE_ENVELOPE_BYTES,
    MAX_STATEMENT_BYTES,
    DecisionJsonError,
    parse_strict_json,
)

MAX_SIGSTORE_BUNDLE_BYTES = 4 * 1024 * 1024
DEFAULT_TRUSTED_FILE_BYTES = 4 * 1024 * 1024
MAX_TRUSTED_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_EVIDENCE_BYTES = 32 * 1024 * 1024


class _ReportContext(TypedDict):
    backend: str
    evaluator: str
    evaluated_at: datetime
    decision_id: NotRequired[str]


class SigstoreBackendUnavailable(RuntimeError):
    """No ratified cryptographic backend is available."""


class SigstoreVerificationFailure(RuntimeError):
    """The backend conclusively rejected a Sigstore bundle."""


class EvidenceEvaluatorUnavailable(RuntimeError):
    """No trusted semantic evidence evaluator is available."""


class EvidenceEvaluationFailure(RuntimeError):
    """The trusted evidence evaluator could not safely derive gate results."""


@dataclass(frozen=True)
class VerifiedSigstoreBundle:
    """Exact DSSE bytes and certificate facts authenticated by a backend."""

    dsse_envelope_bytes: bytes
    identity: str
    issuer: str
    trusted_time: datetime
    trusted_time_source: TrustedTimeSource


class SigstoreVerificationBackend(Protocol):
    """Cryptographic boundary. Implementations must verify exact returned bytes."""

    kind: str

    def verify(self, bundle_bytes: bytes) -> VerifiedSigstoreBundle:
        """Verify one complete Sigstore bundle or raise a controlled exception."""
        raise RuntimeError("protocol method has no concrete backend")


class EvidenceEvaluator(Protocol):
    """Semantic boundary for rederiving approved gates from trusted evidence bytes."""

    kind: str

    def evaluate(
        self,
        evidence_bytes: Mapping[str, bytes],
        statement: InTotoStatement,
        policy: ShadowVerificationPolicy,
    ) -> EvidenceAssessment:
        """Rederive mandatory gates without trusting signed PASS claims."""
        raise RuntimeError("protocol method has no concrete evaluator")


class UnavailableSigstoreBackend:
    """Default backend: inability to verify can never become approval."""

    kind = "unavailable"

    def verify(self, bundle_bytes: bytes) -> VerifiedSigstoreBundle:
        del bundle_bytes
        raise SigstoreBackendUnavailable("no ratified Sigstore backend is configured")


class UnavailableEvidenceEvaluator:
    """Default evaluator: hash-valid evidence alone cannot approve an audit."""

    kind = "unavailable"

    def evaluate(
        self,
        evidence_bytes: Mapping[str, bytes],
        statement: InTotoStatement,
        policy: ShadowVerificationPolicy,
    ) -> EvidenceAssessment:
        del evidence_bytes, statement, policy
        raise EvidenceEvaluatorUnavailable("no trusted evidence evaluator is configured")


@dataclass(frozen=True)
class TrustedFileBinding:
    """Caller-supplied path constrained to an explicit trusted root."""

    path: Path
    allowed_root: Path
    max_bytes: int = DEFAULT_TRUSTED_FILE_BYTES


@dataclass(frozen=True)
class ShadowVerificationContext:
    """All filesystem paths come from trusted caller context, never from the payload."""

    policy_document: TrustedFileBinding
    subjects: Mapping[str, TrustedFileBinding]
    evidence: Mapping[str, TrustedFileBinding]
    requester_authorization_id: str


@dataclass
class InMemoryReplayStore:
    """Ephemeral replay state for tests and shadow sessions only."""

    decision_ids: set[str] = field(default_factory=set)
    nonces: set[str] = field(default_factory=set)
    sequences: dict[tuple[str, str], int] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def check_and_record(
        self,
        *,
        decision_id: str,
        nonce: str,
        subject_key: str,
        scope: str,
        sequence: int,
    ) -> ReasonCode | None:
        """Atomically consume one decision in ephemeral process-local state."""
        with self._lock:
            if decision_id in self.decision_ids or nonce in self.nonces:
                return ReasonCode.REPLAY_DETECTED
            previous = self.sequences.get((subject_key, scope))
            if previous is not None and sequence <= previous:
                return ReasonCode.SEQUENCE_REPLAYED
            self.decision_ids.add(decision_id)
            self.nonces.add(nonce)
            self.sequences[(subject_key, scope)] = sequence
        return None


@dataclass(frozen=True)
class _TrustedFileContent:
    content: bytes
    sha256: str


class _TrustedFileMissing(FileNotFoundError):
    pass


class _TrustedFileUnsafe(ValueError):
    pass


@dataclass(frozen=True)
class _PrincipalAuthorizationFailure:
    outcome: DecisionOutcome
    reasons: tuple[ReasonCode, ...]
    notes: tuple[str, ...] = ()


def verify_signed_decision_shadow(
    submission_bytes: bytes,
    *,
    context: ShadowVerificationContext,
    policy: ShadowVerificationPolicy,
    backend: SigstoreVerificationBackend | None = None,
    evidence_evaluator: EvidenceEvaluator | None = None,
    replay_store: InMemoryReplayStore | None = None,
    now: datetime | None = None,
) -> ShadowVerificationReport:
    """Calculate a shadow result without signing, executing, or persisting production state."""
    resolved_backend = backend or UnavailableSigstoreBackend()
    resolved_evaluator = evidence_evaluator or UnavailableEvidenceEvaluator()
    report_context: _ReportContext = {
        "backend": _safe_component_kind(getattr(resolved_backend, "kind", None)),
        "evaluator": _safe_component_kind(getattr(resolved_evaluator, "kind", None)),
        "evaluated_at": datetime.now(UTC),
    }
    try:
        evaluated_at = _require_utc_now(now)
    except ValueError:
        return _report(
            DecisionOutcome.REVIEW_REQUIRED,
            (ReasonCode.VALIDITY_INVALID,),
            report_context,
            notes=("trusted caller supplied a non-UTC evaluation time",),
        )
    report_context["evaluated_at"] = evaluated_at
    if not isinstance(policy, ShadowVerificationPolicy):
        return _report(
            DecisionOutcome.REVIEW_REQUIRED,
            (ReasonCode.POLICY_HASH_MISMATCH,),
            report_context,
            notes=("local policy contract is unavailable or invalid",),
        )

    try:
        signed_data = parse_strict_json(submission_bytes, profile="signed_set")
        signed_set = SignedDecisionSet.model_validate(signed_data, strict=True)
    except (DecisionJsonError, ValidationError) as error:
        return _report(
            DecisionOutcome.REJECTED_AUTHENTICATED,
            (_controlled_error_reason(error, ReasonCode.INPUT_INVALID),),
            report_context,
            notes=("signed decision set rejected by the strict boundary",),
        )

    verified_bundles: list[VerifiedSigstoreBundle] = []
    verified_envelopes: list[DsseEnvelope] = []
    payloads: list[bytes] = []
    for descriptor in signed_set.bundles:
        try:
            bundle_bytes = decode_standard_base64(
                descriptor.bundle_base64,
                field_name="Sigstore bundle",
            )
        except ValueError:
            return _report(
                DecisionOutcome.REJECTED_AUTHENTICATED,
                (ReasonCode.INPUT_INVALID,),
                report_context,
                notes=("bundle base64 rejected",),
            )
        if len(bundle_bytes) > MAX_SIGSTORE_BUNDLE_BYTES:
            return _report(
                DecisionOutcome.REJECTED_AUTHENTICATED,
                (ReasonCode.RESOURCE_LIMIT_EXCEEDED,),
                report_context,
                notes=("Sigstore bundle exceeds the shadow limit",),
            )
        if _sha256(bundle_bytes) != descriptor.bundle_sha256:
            return _report(
                DecisionOutcome.REJECTED_AUTHENTICATED,
                (ReasonCode.BUNDLE_HASH_MISMATCH,),
                report_context,
            )
        try:
            verified = resolved_backend.verify(bundle_bytes)
        except Exception as error:
            if isinstance(error, SigstoreBackendUnavailable):
                return _report(
                    DecisionOutcome.REVIEW_REQUIRED,
                    (ReasonCode.BACKEND_UNAVAILABLE,),
                    report_context,
                    notes=("cryptographic backend is unavailable",),
                )
            if isinstance(error, SigstoreVerificationFailure):
                return _report(
                    DecisionOutcome.REJECTED_AUTHENTICATED,
                    (ReasonCode.SIGNATURE_INVALID,),
                    report_context,
                )
            return _report(
                DecisionOutcome.REVIEW_REQUIRED,
                (ReasonCode.BACKEND_ERROR,),
                report_context,
                notes=("cryptographic backend raised an unclassified error",),
            )
        if not isinstance(verified, VerifiedSigstoreBundle):
            return _report(
                DecisionOutcome.REVIEW_REQUIRED,
                (ReasonCode.BACKEND_ERROR,),
                report_context,
                notes=("backend returned an unsupported result type",),
            )
        if (
            not isinstance(verified.dsse_envelope_bytes, bytes)
            or not verified.dsse_envelope_bytes
            or len(verified.dsse_envelope_bytes) > MAX_DSSE_ENVELOPE_BYTES
        ):
            return _report(
                DecisionOutcome.REVIEW_REQUIRED,
                (ReasonCode.BACKEND_ERROR,),
                report_context,
                notes=("backend returned invalid DSSE bytes",),
            )
        if (
            type(verified.identity) is not str
            or not verified.identity
            or type(verified.issuer) is not str
            or not verified.issuer
            or not isinstance(verified.trusted_time_source, TrustedTimeSource)
            or verified.trusted_time_source not in policy.allowed_trusted_time_sources
            or not _is_utc(verified.trusted_time)
        ):
            return _report(
                DecisionOutcome.REVIEW_REQUIRED,
                (ReasonCode.TRUSTED_TIME_INVALID,),
                report_context,
                notes=("backend did not provide exact identity and trusted UTC time",),
            )
        try:
            envelope_data = parse_strict_json(
                verified.dsse_envelope_bytes,
                profile="dsse",
            )
            envelope = DsseEnvelope.model_validate(envelope_data, strict=True)
            payload = decode_standard_base64(envelope.payload, field_name="DSSE payload")
        except (DecisionJsonError, ValidationError, ValueError):
            return _report(
                DecisionOutcome.REJECTED_AUTHENTICATED,
                (ReasonCode.DSSE_INVALID,),
                report_context,
            )
        if not payload or len(payload) > MAX_STATEMENT_BYTES:
            return _report(
                DecisionOutcome.REJECTED_AUTHENTICATED,
                (ReasonCode.RESOURCE_LIMIT_EXCEEDED,),
                report_context,
                notes=("in-toto payload exceeds the shadow limit",),
            )
        verified_bundles.append(verified)
        verified_envelopes.append(envelope)
        payloads.append(payload)

    del verified_envelopes
    canonical_payload = payloads[0]
    if any(payload != canonical_payload for payload in payloads[1:]):
        return _report(
            DecisionOutcome.REJECTED_AUTHENTICATED,
            (ReasonCode.PAYLOAD_MISMATCH,),
            report_context,
        )

    try:
        statement_data = parse_strict_json(canonical_payload, profile="statement")
        statement = InTotoStatement.model_validate(statement_data, strict=True)
    except (DecisionJsonError, ValidationError) as error:
        return _report(
            DecisionOutcome.REJECTED_AUTHENTICATED,
            (_controlled_error_reason(error, ReasonCode.STATEMENT_INVALID),),
            report_context,
            notes=("authenticated statement rejected by the strict boundary",),
        )

    decision_id = statement.predicate.decision_id
    report_context["decision_id"] = decision_id
    validity_error = _validate_validity(
        statement,
        verified_bundles,
        policy,
        evaluated_at,
    )
    if validity_error is not None:
        return _report(
            DecisionOutcome.REJECTED_AUTHENTICATED,
            (validity_error,),
            report_context,
        )

    policy_result = _validate_policy_binding(context, policy, statement)
    if isinstance(policy_result, ShadowVerificationReport):
        return _replace_report_context(policy_result, report_context)
    policy_content = policy_result

    principals_result = _authorize_principals(
        verified_bundles,
        policy,
        statement,
        trusted_requester_id=context.requester_authorization_id,
    )
    if isinstance(principals_result, _PrincipalAuthorizationFailure):
        return _report(
            principals_result.outcome,
            principals_result.reasons,
            report_context,
            notes=principals_result.notes,
        )
    authorization_records = principals_result

    if statement.predicate.waivers:
        return _report(
            DecisionOutcome.REVIEW_REQUIRED,
            (ReasonCode.WAIVER_PRESENT,),
            report_context,
            authorization_records=authorization_records,
            policy_sha256=policy_content.sha256,
            notes=("waiver claims require authenticated human adjudication",),
        )

    subject_result = _validate_subjects(context, policy, statement)
    if isinstance(subject_result, ShadowVerificationReport):
        return _replace_report_context(subject_result, report_context)
    subject_key = subject_result

    evidence_result = _validate_evidence(context, policy, statement)
    if isinstance(evidence_result, ShadowVerificationReport):
        return _replace_report_context(evidence_result, report_context)
    evidence_bytes = evidence_result

    if replay_store is None:
        return _report(
            DecisionOutcome.REVIEW_REQUIRED,
            (ReasonCode.REPLAY_STORE_UNAVAILABLE,),
            report_context,
            authorization_records=authorization_records,
            policy_sha256=policy_content.sha256,
        )
    try:
        assessment = resolved_evaluator.evaluate(evidence_bytes, statement, policy)
    except Exception as error:
        if isinstance(error, EvidenceEvaluatorUnavailable):
            return _report(
                DecisionOutcome.REVIEW_REQUIRED,
                (ReasonCode.EVIDENCE_EVALUATOR_UNAVAILABLE,),
                report_context,
                authorization_records=authorization_records,
                policy_sha256=policy_content.sha256,
            )
        if isinstance(error, EvidenceEvaluationFailure):
            return _report(
                DecisionOutcome.REVIEW_REQUIRED,
                (ReasonCode.EVIDENCE_AMBIGUOUS,),
                report_context,
                authorization_records=authorization_records,
                policy_sha256=policy_content.sha256,
            )
        return _report(
            DecisionOutcome.REVIEW_REQUIRED,
            (ReasonCode.EVIDENCE_AMBIGUOUS,),
            report_context,
            authorization_records=authorization_records,
            policy_sha256=policy_content.sha256,
            notes=("evidence evaluator raised an unclassified error",),
        )

    if not isinstance(assessment, EvidenceAssessment):
        return _report(
            DecisionOutcome.REVIEW_REQUIRED,
            (ReasonCode.EVIDENCE_AMBIGUOUS,),
            report_context,
            authorization_records=authorization_records,
            policy_sha256=policy_content.sha256,
            notes=("evidence evaluator returned an unsupported result type",),
        )
    assessment_result = _decision_from_assessment(assessment, policy)
    if assessment_result is not None:
        outcome, reasons = assessment_result
        return _report(
            outcome,
            reasons,
            report_context,
            authorization_records=authorization_records,
            policy_sha256=policy_content.sha256,
        )

    replay_reason = replay_store.check_and_record(
        decision_id=decision_id,
        nonce=statement.predicate.nonce,
        subject_key=subject_key,
        scope=statement.predicate.scope,
        sequence=statement.predicate.sequence,
    )
    if replay_reason is not None:
        return _report(
            DecisionOutcome.REJECTED_AUTHENTICATED,
            (replay_reason,),
            report_context,
            authorization_records=authorization_records,
            policy_sha256=policy_content.sha256,
        )
    return _report(
        DecisionOutcome.APPROVED_FOR_AUDIT,
        (ReasonCode.ALL_MANDATORY_GATES_SUCCESS,),
        report_context,
        authorization_records=authorization_records,
        policy_sha256=policy_content.sha256,
        notes=(
            "shadow-only approval; no action was executed",
            "ephemeral replay state only",
        ),
    )


def _validate_policy_binding(
    context: ShadowVerificationContext,
    policy: ShadowVerificationPolicy,
    statement: InTotoStatement,
) -> _TrustedFileContent | ShadowVerificationReport:
    try:
        policy_content = _read_trusted_file(context.policy_document)
    except (_TrustedFileMissing, _TrustedFileUnsafe) as error:
        if isinstance(error, _TrustedFileMissing):
            return _bare_report(
                DecisionOutcome.INSUFFICIENT_EVIDENCE,
                ReasonCode.EVIDENCE_MISSING,
                note="policy document is missing",
            )
        return _bare_report(
            DecisionOutcome.REJECTED_AUTHENTICATED,
            ReasonCode.POLICY_HASH_MISMATCH,
            note="policy document binding is unsafe",
        )
    claim = statement.predicate.policy
    if (
        policy_content.sha256 != policy.sha256
        or policy_content.sha256 != claim.sha256
        or claim.policy_id != policy.policy_id
        or claim.version != policy.version
        or claim.quorum.threshold != policy.quorum_threshold
        or set(claim.quorum.required_roles) != set(policy.required_roles)
    ):
        return _bare_report(
            DecisionOutcome.REJECTED_AUTHENTICATED,
            ReasonCode.POLICY_HASH_MISMATCH,
        )
    return policy_content


def _authorize_principals(
    verified_bundles: list[VerifiedSigstoreBundle],
    policy: ShadowVerificationPolicy,
    statement: InTotoStatement,
    *,
    trusted_requester_id: str,
) -> tuple[VerifiedAuthorizationRecord, ...] | _PrincipalAuthorizationFailure:
    allowed = {(item.issuer, item.identity): item for item in policy.authorized_principals}
    records: list[VerifiedAuthorizationRecord] = []
    seen_ids: set[str] = set()
    for bundle in verified_bundles:
        principal = allowed.get((bundle.issuer, bundle.identity))
        if principal is None:
            return _PrincipalAuthorizationFailure(
                DecisionOutcome.REJECTED_AUTHENTICATED,
                (ReasonCode.IDENTITY_UNAUTHORIZED,),
            )
        if principal.authorization_id in seen_ids:
            return _PrincipalAuthorizationFailure(
                DecisionOutcome.REJECTED_AUTHENTICATED,
                (ReasonCode.SEPARATION_OF_DUTIES_FAILED,),
                ("duplicate signer principal cannot satisfy quorum",),
            )
        seen_ids.add(principal.authorization_id)
        records.append(
            VerifiedAuthorizationRecord(
                authorizationId=principal.authorization_id,
                role=principal.role,
            )
        )
    roles = {item.role for item in records}
    if len(records) < policy.quorum_threshold or not set(policy.required_roles) <= roles:
        return _PrincipalAuthorizationFailure(
            DecisionOutcome.REVIEW_REQUIRED,
            (ReasonCode.QUORUM_NOT_MET,),
        )
    requester_id = statement.predicate.requester.authorization_id
    if requester_id != trusted_requester_id:
        return _PrincipalAuthorizationFailure(
            DecisionOutcome.REJECTED_AUTHENTICATED,
            (ReasonCode.IDENTITY_UNAUTHORIZED,),
            ("signed requester differs from trusted caller context",),
        )
    if requester_id not in set(policy.allowed_requester_ids):
        return _PrincipalAuthorizationFailure(
            DecisionOutcome.REJECTED_AUTHENTICATED,
            (ReasonCode.IDENTITY_UNAUTHORIZED,),
            ("requester authorization record is not allowed",),
        )
    if requester_id in seen_ids:
        return _PrincipalAuthorizationFailure(
            DecisionOutcome.REJECTED_AUTHENTICATED,
            (ReasonCode.SEPARATION_OF_DUTIES_FAILED,),
            ("requester cannot satisfy an approval role",),
        )
    return tuple(sorted(records, key=lambda item: (item.role.value, item.authorization_id)))


def _validate_subjects(
    context: ShadowVerificationContext,
    policy: ShadowVerificationPolicy,
    statement: InTotoStatement,
) -> str | ShadowVerificationReport:
    claimed = {item.name: item.digest.sha256 for item in statement.subject}
    required = set(policy.required_subject_names)
    if not required <= set(claimed) or not required <= set(context.subjects):
        return _bare_report(
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            ReasonCode.SUBJECT_MISSING,
        )
    if set(claimed) != required or set(context.subjects) != required:
        return _bare_report(
            DecisionOutcome.REVIEW_REQUIRED,
            ReasonCode.EVIDENCE_AMBIGUOUS,
            note="subject set differs from the locally ratified policy",
        )
    digests: list[tuple[str, str]] = []
    for name in sorted(required):
        try:
            content = _read_trusted_file(context.subjects[name])
        except (_TrustedFileMissing, _TrustedFileUnsafe) as error:
            if isinstance(error, _TrustedFileMissing):
                return _bare_report(
                    DecisionOutcome.INSUFFICIENT_EVIDENCE,
                    ReasonCode.SUBJECT_MISSING,
                )
            return _bare_report(
                DecisionOutcome.REJECTED_AUTHENTICATED,
                ReasonCode.SUBJECT_HASH_MISMATCH,
                note="subject binding is unsafe",
            )
        if content.sha256 != claimed[name]:
            return _bare_report(
                DecisionOutcome.REJECTED_AUTHENTICATED,
                ReasonCode.SUBJECT_HASH_MISMATCH,
            )
        digests.append((name, content.sha256))
    encoded = "\n".join(f"{name}:{digest}" for name, digest in digests).encode("utf-8")
    return _sha256(encoded)


def _validate_evidence(
    context: ShadowVerificationContext,
    policy: ShadowVerificationPolicy,
    statement: InTotoStatement,
) -> dict[str, bytes] | ShadowVerificationReport:
    claimed = {item.name: item.sha256 for item in statement.predicate.evidence}
    required = set(policy.required_evidence_names)
    if not required <= set(claimed) or not required <= set(context.evidence):
        return _bare_report(
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            ReasonCode.EVIDENCE_MISSING,
        )
    if set(claimed) != required or set(context.evidence) != required:
        return _bare_report(
            DecisionOutcome.REVIEW_REQUIRED,
            ReasonCode.EVIDENCE_AMBIGUOUS,
            note="evidence set differs from the locally ratified policy",
        )
    content_by_name: dict[str, bytes] = {}
    total_bytes = 0
    for name in sorted(required):
        try:
            content = _read_trusted_file(context.evidence[name])
        except (_TrustedFileMissing, _TrustedFileUnsafe) as error:
            if isinstance(error, _TrustedFileMissing):
                return _bare_report(
                    DecisionOutcome.INSUFFICIENT_EVIDENCE,
                    ReasonCode.EVIDENCE_MISSING,
                )
            return _bare_report(
                DecisionOutcome.REJECTED_AUTHENTICATED,
                ReasonCode.EVIDENCE_HASH_MISMATCH,
                note="evidence binding is unsafe",
            )
        if content.sha256 != claimed[name]:
            return _bare_report(
                DecisionOutcome.REJECTED_AUTHENTICATED,
                ReasonCode.EVIDENCE_HASH_MISMATCH,
            )
        total_bytes += len(content.content)
        if total_bytes > MAX_TOTAL_EVIDENCE_BYTES:
            return _bare_report(
                DecisionOutcome.REVIEW_REQUIRED,
                ReasonCode.RESOURCE_LIMIT_EXCEEDED,
                note="aggregate evidence exceeds the shadow memory limit",
            )
        content_by_name[name] = content.content
    return content_by_name


def _decision_from_assessment(
    assessment: EvidenceAssessment,
    policy: ShadowVerificationPolicy,
) -> tuple[DecisionOutcome, tuple[ReasonCode, ...]] | None:
    gate_by_name = {item.name: item.status for item in assessment.gates}
    required = set(policy.required_gate_names)
    if not required <= set(gate_by_name):
        return DecisionOutcome.INSUFFICIENT_EVIDENCE, (ReasonCode.INSUFFICIENT_EVIDENCE,)
    if set(gate_by_name) != required:
        return DecisionOutcome.REVIEW_REQUIRED, (ReasonCode.EVIDENCE_AMBIGUOUS,)
    statuses = [gate_by_name[name] for name in policy.required_gate_names]
    if EvidenceGateStatus.FAIL in statuses:
        return DecisionOutcome.REJECTED_AUTHENTICATED, (ReasonCode.EVIDENCE_GATE_FAILED,)
    if EvidenceGateStatus.INSUFFICIENT in statuses:
        return DecisionOutcome.INSUFFICIENT_EVIDENCE, (ReasonCode.INSUFFICIENT_EVIDENCE,)
    if EvidenceGateStatus.AMBIGUOUS in statuses:
        return DecisionOutcome.REVIEW_REQUIRED, (ReasonCode.EVIDENCE_AMBIGUOUS,)
    if any(status is not EvidenceGateStatus.SUCCEEDED for status in statuses):
        return DecisionOutcome.REVIEW_REQUIRED, (ReasonCode.EVIDENCE_AMBIGUOUS,)
    return None


def _validate_validity(
    statement: InTotoStatement,
    bundles: list[VerifiedSigstoreBundle],
    policy: ShadowVerificationPolicy,
    now: datetime,
) -> ReasonCode | None:
    try:
        issued = _parse_utc(statement.predicate.validity.issued_on)
        not_before = _parse_utc(statement.predicate.validity.not_before)
        expires = _parse_utc(statement.predicate.validity.expires_on)
    except ValueError:
        return ReasonCode.VALIDITY_INVALID
    if not issued <= not_before < expires:
        return ReasonCode.VALIDITY_INVALID
    skew = timedelta(seconds=policy.max_clock_skew_seconds)
    if not_before - issued > skew:
        return ReasonCode.VALIDITY_INVALID
    if (expires - issued).total_seconds() > policy.max_decision_lifetime_seconds:
        return ReasonCode.VALIDITY_INVALID
    if now + skew < not_before or now - skew > expires:
        return ReasonCode.VALIDITY_INVALID
    for bundle in bundles:
        if not issued - skew <= bundle.trusted_time <= issued + skew:
            return ReasonCode.TRUSTED_TIME_INVALID
    return None


def _read_trusted_file(binding: TrustedFileBinding) -> _TrustedFileContent:
    if (
        type(binding.max_bytes) is not int
        or binding.max_bytes <= 0
        or binding.max_bytes > MAX_TRUSTED_FILE_BYTES
    ):
        raise _TrustedFileUnsafe("invalid trusted-file size limit")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise _TrustedFileUnsafe("descriptor-relative no-follow access is unavailable")

    root = binding.allowed_root.absolute()
    path = binding.path.absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise _TrustedFileUnsafe("path escapes its allowed root") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise _TrustedFileUnsafe("path must be a strict descendant")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        if error.errno == errno.ENOENT:
            raise _TrustedFileMissing(str(root)) from error
        raise _TrustedFileUnsafe("trusted root could not be resolved") from error
    if resolved_root != root or not resolved_root.is_dir():
        raise _TrustedFileUnsafe("trusted root must be a real directory without symlinks")

    root_fd = -1
    directory_fds: list[int] = []
    file_fd = -1
    digest = hashlib.sha256()
    content = bytearray()
    try:
        root_fd = _open_safe_component(
            None,
            root,
            directory=True,
            no_follow=no_follow,
            directory_flag=directory_flag,
        )
        current_fd = root_fd
        for part in relative.parts[:-1]:
            next_fd = _open_safe_component(
                current_fd,
                part,
                directory=True,
                no_follow=no_follow,
                directory_flag=directory_flag,
            )
            directory_fds.append(next_fd)
            current_fd = next_fd
        file_fd = _open_safe_component(
            current_fd,
            relative.parts[-1],
            directory=False,
            no_follow=no_follow,
            directory_flag=directory_flag,
        )
        initial = os.fstat(file_fd)
        if not stat.S_ISREG(initial.st_mode):
            raise _TrustedFileUnsafe("trusted path must reference a regular file")
        if initial.st_size > binding.max_bytes:
            raise _TrustedFileUnsafe("trusted file exceeds its size limit")
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            if len(content) + len(chunk) > binding.max_bytes:
                raise _TrustedFileUnsafe("trusted file exceeds its size limit")
            content.extend(chunk)
            digest.update(chunk)
        final = os.fstat(file_fd)
    except OSError as error:
        if error.errno == errno.ENOENT:
            raise _TrustedFileMissing(str(path)) from error
        raise _TrustedFileUnsafe("trusted file could not be opened safely") from error
    finally:
        if file_fd >= 0:
            _close_fd(file_fd)
        for descriptor in reversed(directory_fds):
            _close_fd(descriptor)
        if root_fd >= 0:
            _close_fd(root_fd)
    if (
        initial.st_dev,
        initial.st_ino,
        initial.st_size,
        initial.st_mtime_ns,
    ) != (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
    ):
        raise _TrustedFileUnsafe("trusted file changed while hashing")
    return _TrustedFileContent(content=bytes(content), sha256=digest.hexdigest())


def _open_safe_component(
    parent_fd: int | None,
    component: Path | str,
    *,
    directory: bool,
    no_follow: int,
    directory_flag: int,
) -> int:
    """Open one pre-validated path component without following symlinks."""
    flags = os.O_RDONLY | no_follow
    safe_component = os.fspath(component)
    if directory:
        flags |= directory_flag
    else:
        flags |= os.O_NONBLOCK
    if parent_fd is None:
        return _os_open_for_contained_file(safe_component, flags)
    return _os_open_for_contained_file(safe_component, flags, dir_fd=parent_fd)


def _os_open_for_contained_file(component: str, flags: int, **kwargs: int) -> int:
    """Internal no-follow open; callers must perform containment checks first."""
    return os.open(component, flags, **kwargs)


def _close_fd(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        return


def _bare_report(
    outcome: DecisionOutcome,
    reason: ReasonCode,
    *,
    note: str | None = None,
) -> ShadowVerificationReport:
    return ShadowVerificationReport(
        outcome=outcome,
        reasonCodes=(reason,),
        cryptographicBackend="pending",
        evidenceEvaluator="pending",
        evaluatedAt="1970-01-01T00:00:00Z",
        notes=() if note is None else (note,),
    )


def _replace_report_context(
    report: ShadowVerificationReport,
    report_context: _ReportContext,
) -> ShadowVerificationReport:
    return report.model_copy(
        update={
            "decision_id": report_context.get("decision_id"),
            "cryptographic_backend": report_context["backend"],
            "evidence_evaluator": report_context["evaluator"],
            "evaluated_at": _format_utc(report_context["evaluated_at"]),
        }
    )


def _report(
    outcome: DecisionOutcome,
    reasons: tuple[ReasonCode, ...],
    report_context: _ReportContext,
    *,
    authorization_records: tuple[VerifiedAuthorizationRecord, ...] = (),
    policy_sha256: str | None = None,
    notes: tuple[str, ...] = (),
) -> ShadowVerificationReport:
    unique_reasons = tuple(dict.fromkeys(reasons))
    return ShadowVerificationReport(
        outcome=outcome,
        decisionId=report_context.get("decision_id"),
        reasonCodes=unique_reasons,
        authorizationRecords=authorization_records,
        cryptographicBackend=report_context["backend"],
        evidenceEvaluator=report_context["evaluator"],
        policySha256=policy_sha256,
        evaluatedAt=_format_utc(report_context["evaluated_at"]),
        notes=notes,
    )


def _require_utc_now(value: datetime | None) -> datetime:
    resolved = value or datetime.now(UTC)
    if not _is_utc(resolved):
        raise ValueError("shadow verifier now must be timezone-aware UTC")
    return resolved


def _is_utc(value: datetime) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() == timedelta(0)
    )


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    if not _is_utc(parsed):
        raise ValueError("timestamp must be UTC")
    return parsed


def _format_utc(value: object) -> str:
    if not isinstance(value, datetime) or not _is_utc(value):
        raise ValueError("report timestamp must be UTC")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _safe_component_kind(value: object) -> str:
    if type(value) is not str or not value or len(value) > 256:
        return "invalid-component"
    return value


def _controlled_error_reason(error: Exception, fallback: ReasonCode) -> ReasonCode:
    reason = getattr(error, "reason_code", None)
    return reason if isinstance(reason, ReasonCode) else fallback


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
