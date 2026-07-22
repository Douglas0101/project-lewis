"""Fail-closed paths for the non-operational authenticated-decision verifier."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from src.security.authenticated_decision_contracts import (
    AuthorizedPrincipal,
    DecisionOutcome,
    DerivedGateResult,
    EvidenceAssessment,
    EvidenceGateStatus,
    MandatoryGateName,
    ReasonCode,
    ShadowVerificationPolicy,
    ShadowVerificationReport,
    SignerRole,
    TrustedTimeSource,
)
from src.security.authenticated_decision_verifier import (
    EvidenceEvaluatorUnavailable,
    InMemoryReplayStore,
    ShadowVerificationContext,
    SigstoreVerificationFailure,
    TrustedFileBinding,
    VerifiedSigstoreBundle,
    verify_signed_decision_shadow,
)

NOW = datetime(2026, 7, 17, 10, 30, tzinfo=UTC)
SIGNING_TIME = datetime(2026, 7, 17, 10, 0, 30, tzinfo=UTC)
DECISION_ID = str(UUID("12345678-1234-4234-9234-123456789abc"))
BOT_ISSUER = "https://token.actions.fixture.invalid"
BOT_IDENTITY = "fixture://project-lewis/evidence-bot"
HUMAN_ISSUER = "https://oidc.fixture.invalid"
HUMAN_IDENTITY = "fixture://project-lewis/scientific-approver"
REQUIRED_EVIDENCE = ("focused-tests", "preflight", "run-matrix", "smoke-gate")


class _FixtureBackend:
    kind = "fixture"

    def __init__(
        self,
        verified: Mapping[bytes, VerifiedSigstoreBundle],
        *,
        rejected: set[bytes] | None = None,
    ) -> None:
        self._verified = dict(verified)
        self._rejected = rejected or set()

    def verify(self, bundle_bytes: bytes) -> VerifiedSigstoreBundle:
        if bundle_bytes in self._rejected:
            raise SigstoreVerificationFailure("fixture rejection")
        return self._verified[bundle_bytes]


class _FixtureEvaluator:
    kind = "fixture"

    def __init__(self, status: EvidenceGateStatus = EvidenceGateStatus.SUCCEEDED) -> None:
        self._status = status

    def evaluate(
        self,
        evidence_bytes: Mapping[str, bytes],
        statement: Any,
        policy: ShadowVerificationPolicy,
    ) -> EvidenceAssessment:
        assert set(evidence_bytes) == set(policy.required_evidence_names)
        assert statement.predicate.scope == "E06_5_AUDIT"
        reason = {
            EvidenceGateStatus.SUCCEEDED: ReasonCode.ALL_MANDATORY_GATES_SUCCESS,
            EvidenceGateStatus.FAIL: ReasonCode.EVIDENCE_GATE_FAILED,
            EvidenceGateStatus.INSUFFICIENT: ReasonCode.INSUFFICIENT_EVIDENCE,
            EvidenceGateStatus.AMBIGUOUS: ReasonCode.EVIDENCE_AMBIGUOUS,
        }[self._status]
        return EvidenceAssessment(
            gates=tuple(
                DerivedGateResult(name=name, status=self._status)
                for name in policy.required_gate_names
            ),
            reasonCodes=(reason,),
        )


class _BrokenBackend:
    kind = "broken-fixture"

    def verify(self, bundle_bytes: bytes) -> VerifiedSigstoreBundle:
        del bundle_bytes
        raise RuntimeError("fixture infrastructure failure")


class _BadResultBackend:
    kind = "bad-result-fixture"

    def verify(self, bundle_bytes: bytes) -> Any:
        del bundle_bytes
        return {"not": "a verified bundle"}


class _BadResultEvaluator:
    kind = "bad-result-fixture"

    def evaluate(self, evidence_bytes: Mapping[str, bytes], statement: Any, policy: Any) -> Any:
        del evidence_bytes, statement, policy
        return {"not": "an evidence assessment"}


class _UnavailableFixtureEvaluator:
    kind = "unavailable-fixture"

    def evaluate(self, evidence_bytes: Mapping[str, bytes], statement: Any, policy: Any) -> Any:
        del evidence_bytes, statement, policy
        raise EvidenceEvaluatorUnavailable("fixture unavailable")


@dataclass
class _Case:
    signed_set: dict[str, Any]
    context: ShadowVerificationContext
    policy: ShadowVerificationPolicy
    backend: _FixtureBackend
    evaluator: _FixtureEvaluator
    replay_store: InMemoryReplayStore
    evidence_paths: dict[str, Path]
    policy_path: Path

    def submission(self) -> bytes:
        return _json_bytes(self.signed_set)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _nonce(byte: bytes = b"n") -> str:
    return base64.urlsafe_b64encode(byte * 32).decode("ascii").rstrip("=")


def _make_case(
    tmp_path: Path,
    *,
    statement_mutator: Callable[[dict[str, Any]], None] | None = None,
    signers: tuple[tuple[str, str], ...] = (
        (BOT_ISSUER, BOT_IDENTITY),
        (HUMAN_ISSUER, HUMAN_IDENTITY),
    ),
    payload_mismatch: bool = False,
    invalid_dsse: bool = False,
    trusted_time: datetime = SIGNING_TIME,
    trusted_time_source: TrustedTimeSource = TrustedTimeSource.REKOR,
    bundle_hash_mismatch: bool = False,
    rejected_bundle_index: int | None = None,
    evaluator_status: EvidenceGateStatus = EvidenceGateStatus.SUCCEEDED,
) -> _Case:
    policy_path = tmp_path / "policy.md"
    policy_path.write_text("fixture policy v1\n", encoding="utf-8")
    subject_path = tmp_path / "audit-plan.csv"
    subject_path.write_text("candidate,fold,seed\nH6,1,17\n", encoding="utf-8")
    evidence_paths: dict[str, Path] = {}
    for name in REQUIRED_EVIDENCE:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name, "status": "PASS"}), encoding="utf-8")
        evidence_paths[name] = path

    policy_sha = _sha256(policy_path.read_bytes())
    statement: dict[str, Any] = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {
                "name": "audit-plan",
                "digest": {"sha256": _sha256(subject_path.read_bytes())},
            }
        ],
        "predicateType": "https://project-lewis.dev/attestations/research-decision/v1",
        "predicate": {
            "decisionId": DECISION_ID,
            "nonce": _nonce(),
            "sequence": 1,
            "scope": "E06_5_AUDIT",
            "requester": {"authorizationId": "authz_requester00000001"},
            "policy": {
                "policyId": "project-lewis/research-decision/v1",
                "version": "1.0.0",
                "sha256": policy_sha,
                "quorum": {
                    "threshold": 2,
                    "requiredRoles": ["EVIDENCE_BOT", "SCIENTIFIC_APPROVER"],
                },
            },
            "evidence": [
                {
                    "name": name,
                    "sha256": _sha256(evidence_paths[name].read_bytes()),
                    "mediaType": "application/json",
                }
                for name in REQUIRED_EVIDENCE
            ],
            "waivers": [],
            "validity": {
                "issuedOn": "2026-07-17T10:00:00Z",
                "notBefore": "2026-07-17T10:00:00Z",
                "expiresOn": "2026-07-17T11:00:00Z",
            },
            "claimedReasonCodes": ["ALL_MANDATORY_GATES_PASS"],
        },
    }
    if statement_mutator is not None:
        statement_mutator(statement)

    payloads = [_json_bytes(statement) for _ in signers]
    if payload_mismatch and len(payloads) > 1:
        different = deepcopy(statement)
        different["predicate"]["sequence"] = 2
        payloads[-1] = _json_bytes(different)

    descriptors: list[dict[str, str]] = []
    verified: dict[bytes, VerifiedSigstoreBundle] = {}
    rejected: set[bytes] = set()
    for index, ((issuer, identity), payload) in enumerate(zip(signers, payloads, strict=True)):
        bundle_bytes = _json_bytes({"fixtureBundle": index, "identity": identity})
        if invalid_dsse:
            dsse_bytes = b'{"payloadType":"application/vnd.in-toto+json"}'
        else:
            dsse_bytes = _json_bytes(
                {
                    "payloadType": "application/vnd.in-toto+json",
                    "payload": base64.b64encode(payload).decode("ascii"),
                    "signatures": [
                        {
                            "keyid": f"fixture-{index}",
                            "sig": base64.b64encode(f"sig-{index}".encode()).decode("ascii"),
                        }
                    ],
                }
            )
        verified[bundle_bytes] = VerifiedSigstoreBundle(
            dsse_envelope_bytes=dsse_bytes,
            identity=identity,
            issuer=issuer,
            trusted_time=trusted_time,
            trusted_time_source=trusted_time_source,
        )
        if rejected_bundle_index == index:
            rejected.add(bundle_bytes)
        descriptor_hash = "0" * 64 if bundle_hash_mismatch and index == 0 else _sha256(bundle_bytes)
        descriptors.append(
            {
                "bundleMediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
                "bundleBase64": base64.b64encode(bundle_bytes).decode("ascii"),
                "bundleSha256": descriptor_hash,
            }
        )

    policy = ShadowVerificationPolicy(
        policyId="project-lewis/research-decision/v1",
        version="1.0.0",
        sha256=policy_sha,
        authorizedPrincipals=(
            AuthorizedPrincipal(
                authorizationId="authz_bot0000000000001",
                role=SignerRole.EVIDENCE_BOT,
                issuer=BOT_ISSUER,
                identity=BOT_IDENTITY,
            ),
            AuthorizedPrincipal(
                authorizationId="authz_scientist0000001",
                role=SignerRole.SCIENTIFIC_APPROVER,
                issuer=HUMAN_ISSUER,
                identity=HUMAN_IDENTITY,
            ),
        ),
        allowedRequesterIds=("authz_requester00000001",),
        requiredRoles=(SignerRole.EVIDENCE_BOT, SignerRole.SCIENTIFIC_APPROVER),
        quorumThreshold=2,
        requiredSubjectNames=("audit-plan",),
        requiredEvidenceNames=REQUIRED_EVIDENCE,
        requiredGateNames=tuple(MandatoryGateName),
        allowedTrustedTimeSources=(TrustedTimeSource.REKOR,),
    )
    context = ShadowVerificationContext(
        policy_document=TrustedFileBinding(policy_path, tmp_path),
        subjects={"audit-plan": TrustedFileBinding(subject_path, tmp_path)},
        evidence={
            name: TrustedFileBinding(path, tmp_path) for name, path in evidence_paths.items()
        },
        requester_authorization_id="authz_requester00000001",
    )
    return _Case(
        signed_set={
            "format": "project-lewis.dev/signed-decision-set/v1",
            "bundles": descriptors,
        },
        context=context,
        policy=policy,
        backend=_FixtureBackend(verified, rejected=rejected),
        evaluator=_FixtureEvaluator(evaluator_status),
        replay_store=InMemoryReplayStore(),
        evidence_paths=evidence_paths,
        policy_path=policy_path,
    )


def _assert_single_reason(
    report: ShadowVerificationReport,
    reason: ReasonCode,
) -> None:
    assert len(report.reason_codes) == 1
    assert report.reason_codes[0] == reason


def _verify(case: _Case, **kwargs: Any) -> ShadowVerificationReport:
    return verify_signed_decision_shadow(
        case.submission(),
        context=case.context,
        policy=case.policy,
        backend=kwargs.pop("backend", case.backend),
        evidence_evaluator=kwargs.pop("evidence_evaluator", case.evaluator),
        replay_store=kwargs.pop("replay_store", case.replay_store),
        now=kwargs.pop("now", NOW),
        **kwargs,
    )


def test_valid_fixture_can_only_compute_non_operational_audit_approval(tmp_path: Path) -> None:
    case = _make_case(tmp_path)

    report = _verify(case)

    assert report.outcome == DecisionOutcome.APPROVED_FOR_AUDIT
    assert report.shadow
    assert not report.operational
    assert report.cryptographic_backend == "fixture"
    assert report.evidence_evaluator == "fixture"
    assert {item.role for item in report.authorization_records} == {
        SignerRole.EVIDENCE_BOT,
        SignerRole.SCIENTIFIC_APPROVER,
    }
    serialized = json.dumps(report.model_dump(mode="json", by_alias=True))
    assert BOT_IDENTITY not in serialized
    assert HUMAN_IDENTITY not in serialized


def test_default_missing_crypto_backend_requires_review(tmp_path: Path) -> None:
    case = _make_case(tmp_path)

    report = _verify(case, backend=None)

    assert report.outcome == DecisionOutcome.REVIEW_REQUIRED
    _assert_single_reason(report, ReasonCode.BACKEND_UNAVAILABLE)


def test_unknown_submission_field_is_rejected_before_pydantic(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    case.signed_set["unexpected"] = True

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(report, ReasonCode.UNKNOWN_FIELD)


def test_unclassified_backend_error_requires_review(tmp_path: Path) -> None:
    case = _make_case(tmp_path)

    report = _verify(case, backend=_BrokenBackend())

    assert report.outcome == DecisionOutcome.REVIEW_REQUIRED
    _assert_single_reason(report, ReasonCode.BACKEND_ERROR)


def test_backend_wrong_result_type_requires_review(tmp_path: Path) -> None:
    case = _make_case(tmp_path)

    report = _verify(case, backend=_BadResultBackend())

    assert report.outcome == DecisionOutcome.REVIEW_REQUIRED
    _assert_single_reason(report, ReasonCode.BACKEND_ERROR)


def test_conclusive_signature_failure_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path, rejected_bundle_index=0)

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(report, ReasonCode.SIGNATURE_INVALID)


def test_bundle_hash_mismatch_is_rejected_before_backend(tmp_path: Path) -> None:
    case = _make_case(tmp_path, bundle_hash_mismatch=True)

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(report, ReasonCode.BUNDLE_HASH_MISMATCH)


@pytest.mark.parametrize(
    "signers",
    [
        ((BOT_ISSUER, "fixture://unauthorized"), (HUMAN_ISSUER, HUMAN_IDENTITY)),
        (("https://wrong-issuer.invalid", BOT_IDENTITY), (HUMAN_ISSUER, HUMAN_IDENTITY)),
    ],
)
def test_exact_identity_and_issuer_are_required(
    tmp_path: Path,
    signers: tuple[tuple[str, str], ...],
) -> None:
    case = _make_case(tmp_path, signers=signers)

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(report, ReasonCode.IDENTITY_UNAUTHORIZED)


def test_quorum_missing_human_approver_requires_review(tmp_path: Path) -> None:
    case = _make_case(tmp_path, signers=((BOT_ISSUER, BOT_IDENTITY),))

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REVIEW_REQUIRED
    _assert_single_reason(report, ReasonCode.QUORUM_NOT_MET)


def test_requester_cannot_self_approve(tmp_path: Path) -> None:
    def mutate(statement: dict[str, Any]) -> None:
        statement["predicate"]["requester"]["authorizationId"] = "authz_scientist0000001"

    case = _make_case(tmp_path, statement_mutator=mutate)
    case.policy = case.policy.model_copy(
        update={"allowed_requester_ids": ("authz_scientist0000001",)}
    )
    case.context = ShadowVerificationContext(
        policy_document=case.context.policy_document,
        subjects=case.context.subjects,
        evidence=case.context.evidence,
        requester_authorization_id="authz_scientist0000001",
    )

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(report, ReasonCode.SEPARATION_OF_DUTIES_FAILED)


def test_signed_requester_cannot_impersonate_trusted_caller(tmp_path: Path) -> None:
    def mutate(statement: dict[str, Any]) -> None:
        statement["predicate"]["requester"]["authorizationId"] = "authz_requester00000002"

    case = _make_case(tmp_path, statement_mutator=mutate)
    case.policy = case.policy.model_copy(
        update={
            "allowed_requester_ids": (
                "authz_requester00000001",
                "authz_requester00000002",
            )
        }
    )

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(report, ReasonCode.IDENTITY_UNAUTHORIZED)


def test_payloads_must_be_byte_identical_across_bundles(tmp_path: Path) -> None:
    case = _make_case(tmp_path, payload_mismatch=True)

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(report, ReasonCode.PAYLOAD_MISMATCH)


def test_invalid_dsse_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path, invalid_dsse=True)

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(report, ReasonCode.DSSE_INVALID)


def test_policy_hash_is_recomputed_from_trusted_path(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    case.policy_path.write_text("tampered policy\n", encoding="utf-8")

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(report, ReasonCode.POLICY_HASH_MISMATCH)


def test_subject_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    subject_path = case.context.subjects["audit-plan"].path
    subject_path.write_text("tampered\n", encoding="utf-8")

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(report, ReasonCode.SUBJECT_HASH_MISMATCH)


def test_evidence_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    case.evidence_paths["smoke-gate"].write_text("tampered\n", encoding="utf-8")

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(report, ReasonCode.EVIDENCE_HASH_MISMATCH)


def test_missing_evidence_is_insufficient_not_approved(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    case.evidence_paths["smoke-gate"].unlink()

    report = _verify(case)

    assert report.outcome == DecisionOutcome.INSUFFICIENT_EVIDENCE
    _assert_single_reason(report, ReasonCode.EVIDENCE_MISSING)


def test_non_utc_evaluation_time_requires_review(tmp_path: Path) -> None:
    case = _make_case(tmp_path)

    report = _verify(case, now=NOW.replace(tzinfo=None))

    assert report.outcome == DecisionOutcome.REVIEW_REQUIRED
    _assert_single_reason(report, ReasonCode.VALIDITY_INVALID)


def test_expired_decision_is_rejected(tmp_path: Path) -> None:
    def mutate(statement: dict[str, Any]) -> None:
        statement["predicate"]["validity"]["expiresOn"] = "2026-07-17T10:10:00Z"

    case = _make_case(tmp_path, statement_mutator=mutate)

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(report, ReasonCode.VALIDITY_INVALID)


def test_stale_preauthorization_window_is_rejected(tmp_path: Path) -> None:
    def mutate(statement: dict[str, Any]) -> None:
        statement["predicate"]["validity"] = {
            "issuedOn": "2020-01-01T00:00:00Z",
            "notBefore": "2026-07-17T10:00:00Z",
            "expiresOn": "2026-07-17T11:00:00Z",
        }

    case = _make_case(tmp_path, statement_mutator=mutate)

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(report, ReasonCode.VALIDITY_INVALID)


def test_unverified_trusted_time_requires_review(tmp_path: Path) -> None:
    case = _make_case(tmp_path, trusted_time=NOW.replace(tzinfo=None))

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REVIEW_REQUIRED
    _assert_single_reason(report, ReasonCode.TRUSTED_TIME_INVALID)


def test_unratified_trusted_time_source_requires_review(tmp_path: Path) -> None:
    case = _make_case(tmp_path, trusted_time_source=TrustedTimeSource.RFC3161)

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REVIEW_REQUIRED
    _assert_single_reason(report, ReasonCode.TRUSTED_TIME_INVALID)


def test_waiver_forces_review_even_when_all_gates_pass(tmp_path: Path) -> None:
    def mutate(statement: dict[str, Any]) -> None:
        statement["predicate"]["waivers"] = [
            {
                "code": "PREEXISTING_FULL_SUITE_FAILURE",
                "justification": "fixture waiver requiring human review",
                "evidenceName": "focused-tests",
            }
        ]

    case = _make_case(tmp_path, statement_mutator=mutate)

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REVIEW_REQUIRED
    _assert_single_reason(report, ReasonCode.WAIVER_PRESENT)


def test_waiver_takes_review_precedence_over_missing_evidence(tmp_path: Path) -> None:
    def mutate(statement: dict[str, Any]) -> None:
        statement["predicate"]["waivers"] = [
            {
                "code": "PREEXISTING_FULL_SUITE_FAILURE",
                "justification": "fixture waiver requiring human review",
            }
        ]

    case = _make_case(tmp_path, statement_mutator=mutate)
    case.evidence_paths["smoke-gate"].unlink()

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REVIEW_REQUIRED
    _assert_single_reason(report, ReasonCode.WAIVER_PRESENT)


def test_evidence_evaluator_wrong_result_type_requires_review(tmp_path: Path) -> None:
    case = _make_case(tmp_path)

    report = _verify(case, evidence_evaluator=_BadResultEvaluator())

    assert report.outcome == DecisionOutcome.REVIEW_REQUIRED
    _assert_single_reason(report, ReasonCode.EVIDENCE_AMBIGUOUS)


def test_default_evidence_evaluator_requires_review(tmp_path: Path) -> None:
    case = _make_case(tmp_path)

    report = _verify(case, evidence_evaluator=None)

    assert report.outcome == DecisionOutcome.REVIEW_REQUIRED
    _assert_single_reason(report, ReasonCode.EVIDENCE_EVALUATOR_UNAVAILABLE)


def test_missing_evidence_evaluator_requires_review(tmp_path: Path) -> None:
    case = _make_case(tmp_path)

    report = _verify(case, evidence_evaluator=_UnavailableFixtureEvaluator())

    assert report.outcome == DecisionOutcome.REVIEW_REQUIRED
    _assert_single_reason(report, ReasonCode.EVIDENCE_EVALUATOR_UNAVAILABLE)


@pytest.mark.parametrize(
    ("status", "outcome", "reason"),
    [
        (
            EvidenceGateStatus.FAIL,
            DecisionOutcome.REJECTED_AUTHENTICATED,
            ReasonCode.EVIDENCE_GATE_FAILED,
        ),
        (
            EvidenceGateStatus.INSUFFICIENT,
            DecisionOutcome.INSUFFICIENT_EVIDENCE,
            ReasonCode.INSUFFICIENT_EVIDENCE,
        ),
        (
            EvidenceGateStatus.AMBIGUOUS,
            DecisionOutcome.REVIEW_REQUIRED,
            ReasonCode.EVIDENCE_AMBIGUOUS,
        ),
    ],
)
def test_semantic_evidence_status_controls_fail_closed_outcome(
    tmp_path: Path,
    status: EvidenceGateStatus,
    outcome: DecisionOutcome,
    reason: ReasonCode,
) -> None:
    case = _make_case(tmp_path, evaluator_status=status)

    report = _verify(case)

    assert report.outcome == outcome
    _assert_single_reason(report, reason)


def test_ephemeral_replay_is_rejected_after_shadow_registration(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    first = _verify(case)

    second = _verify(case)

    assert first.outcome == DecisionOutcome.APPROVED_FOR_AUDIT
    assert second.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(second, ReasonCode.REPLAY_DETECTED)


def test_missing_ephemeral_replay_store_requires_review(tmp_path: Path) -> None:
    case = _make_case(tmp_path)

    report = _verify(case, replay_store=None)

    assert report.outcome == DecisionOutcome.REVIEW_REQUIRED
    _assert_single_reason(report, ReasonCode.REPLAY_STORE_UNAVAILABLE)


def test_ephemeral_replay_consumption_is_atomic_within_process() -> None:
    store = InMemoryReplayStore()

    def consume(_: int) -> ReasonCode | None:
        return store.check_and_record(
            decision_id=DECISION_ID,
            nonce=_nonce(),
            subject_key="subject-key",
            scope="E06_5_AUDIT",
            sequence=1,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(consume, range(16)))

    assert results.count(None) == 1
    assert results.count(ReasonCode.REPLAY_DETECTED) == 15


def test_sequence_must_increase_within_ephemeral_store(tmp_path: Path) -> None:
    first_case = _make_case(tmp_path)
    first = _verify(first_case)

    def mutate(statement: dict[str, Any]) -> None:
        statement["predicate"]["decisionId"] = str(UUID("87654321-4321-4321-8321-cba987654321"))
        statement["predicate"]["nonce"] = _nonce(b"m")
        statement["predicate"]["sequence"] = 1

    second_case = _make_case(tmp_path, statement_mutator=mutate)
    second_case.replay_store = first_case.replay_store
    second = _verify(second_case)

    assert first.outcome == DecisionOutcome.APPROVED_FOR_AUDIT
    assert second.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(second, ReasonCode.SEQUENCE_REPLAYED)


def test_unbounded_trusted_file_limit_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    evidence = dict(case.context.evidence)
    binding = evidence["smoke-gate"]
    evidence["smoke-gate"] = TrustedFileBinding(
        binding.path,
        binding.allowed_root,
        max_bytes=9 * 1024 * 1024,
    )
    case.context = ShadowVerificationContext(
        policy_document=case.context.policy_document,
        subjects=case.context.subjects,
        evidence=evidence,
        requester_authorization_id=case.context.requester_authorization_id,
    )

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(report, ReasonCode.EVIDENCE_HASH_MISMATCH)


def test_symlink_evidence_binding_is_rejected_without_following(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text("outside\n", encoding="utf-8")
    linked = tmp_path / "linked-evidence.json"
    linked.symlink_to(outside)
    evidence = dict(case.context.evidence)
    evidence["smoke-gate"] = TrustedFileBinding(linked, tmp_path)
    case.context = ShadowVerificationContext(
        policy_document=case.context.policy_document,
        subjects=case.context.subjects,
        evidence=evidence,
        requester_authorization_id=case.context.requester_authorization_id,
    )

    report = _verify(case)

    assert report.outcome == DecisionOutcome.REJECTED_AUTHENTICATED
    _assert_single_reason(report, ReasonCode.EVIDENCE_HASH_MISMATCH)
    assert outside.read_text(encoding="utf-8") == "outside\n"
