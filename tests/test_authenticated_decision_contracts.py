"""Strict parsing and Pydantic contracts for authenticated research decisions."""

from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from uuid import UUID

import pytest
from pydantic import ValidationError

from src.security.authenticated_decision_contracts import (
    DsseEnvelope,
    InTotoStatement,
    ReasonCode,
    SignedDecisionSet,
)
from src.security.authenticated_decision_json import DecisionJsonError, parse_strict_json

POLICY_SHA256 = "1" * 64
SUBJECT_SHA256 = "2" * 64
EVIDENCE_SHA256 = "3" * 64
DECISION_ID = str(UUID("12345678-1234-4234-9234-123456789abc"))
NONCE = base64.urlsafe_b64encode(b"n" * 32).decode("ascii").rstrip("=")


def _statement() -> dict[str, object]:
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {
                "name": "audit-plan",
                "digest": {"sha256": SUBJECT_SHA256},
            }
        ],
        "predicateType": "https://project-lewis.dev/attestations/research-decision/v1",
        "predicate": {
            "decisionId": DECISION_ID,
            "nonce": NONCE,
            "sequence": 1,
            "scope": "E06_5_AUDIT",
            "requester": {"authorizationId": "authz_requester00000001"},
            "policy": {
                "policyId": "project-lewis/research-decision/v1",
                "version": "1.0.0",
                "sha256": POLICY_SHA256,
                "quorum": {
                    "threshold": 2,
                    "requiredRoles": ["EVIDENCE_BOT", "SCIENTIFIC_APPROVER"],
                },
            },
            "evidence": [
                {
                    "name": "preflight",
                    "sha256": EVIDENCE_SHA256,
                    "mediaType": "application/json",
                }
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


def _dsse(statement: dict[str, object] | None = None) -> dict[str, object]:
    payload = json.dumps(
        statement or _statement(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "payloadType": "application/vnd.in-toto+json",
        "payload": base64.b64encode(payload).decode("ascii"),
        "signatures": [{"keyid": "fixture", "sig": base64.b64encode(b"sig").decode("ascii")}],
    }


def _signed_set() -> dict[str, object]:
    bundle = b'{"fixture":"bundle"}'
    return {
        "format": "project-lewis.dev/signed-decision-set/v1",
        "bundles": [
            {
                "bundleMediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
                "bundleBase64": base64.b64encode(bundle).decode("ascii"),
                "bundleSha256": hashlib.sha256(bundle).hexdigest(),
            }
        ],
    }


def _encode(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")


def test_valid_documents_are_prevalidated_before_strict_pydantic() -> None:
    signed = SignedDecisionSet.model_validate(
        parse_strict_json(_encode(_signed_set()), profile="signed_set"),
        strict=True,
    )
    envelope = DsseEnvelope.model_validate(
        parse_strict_json(_encode(_dsse()), profile="dsse"),
        strict=True,
    )
    statement = InTotoStatement.model_validate(
        parse_strict_json(
            base64.b64decode(envelope.payload, validate=True),
            profile="statement",
        ),
        strict=True,
    )

    assert len(signed.bundles) == 1
    assert statement.predicate.decision_id == DECISION_ID
    assert statement.predicate.sequence == 1


def test_duplicate_json_key_is_rejected_before_pydantic() -> None:
    raw = (
        b'{"format":"project-lewis.dev/signed-decision-set/v1",'
        b'"format":"project-lewis.dev/signed-decision-set/v1","bundles":[]}'
    )

    with pytest.raises(DecisionJsonError) as captured:
        parse_strict_json(raw, profile="signed_set")

    assert captured.value.reason_code == ReasonCode.DUPLICATE_JSON_KEY


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_is_rejected_before_pydantic(constant: str) -> None:
    raw = (
        '{"format":"project-lewis.dev/signed-decision-set/v1",' f'"bundles":{constant}' + "}"
    ).encode("utf-8")

    with pytest.raises(DecisionJsonError) as captured:
        parse_strict_json(raw, profile="signed_set")

    assert captured.value.reason_code == ReasonCode.NON_FINITE_JSON


def test_oversized_integer_is_rejected_without_escaping_parser() -> None:
    raw = _encode(_statement()).replace(b'"sequence":1', b'"sequence":' + b"9" * 5000)

    with pytest.raises(DecisionJsonError) as captured:
        parse_strict_json(raw, profile="statement")

    assert captured.value.reason_code == ReasonCode.RESOURCE_LIMIT_EXCEEDED


def test_excessive_nesting_is_rejected_without_recursion_escape() -> None:
    raw = b"[" * 2000 + b"0" + b"]" * 2000

    with pytest.raises(DecisionJsonError) as captured:
        parse_strict_json(raw, profile="signed_set")

    assert captured.value.reason_code == ReasonCode.RESOURCE_LIMIT_EXCEEDED


@pytest.mark.parametrize("ambiguous", [True, "1", 1.0])
def test_ambiguous_sequence_type_is_rejected_before_pydantic(ambiguous: object) -> None:
    statement = _statement()
    statement["predicate"]["sequence"] = ambiguous  # type: ignore[index]
    raw = json.dumps(statement, separators=(",", ":")).encode("utf-8")

    with pytest.raises(DecisionJsonError) as captured:
        parse_strict_json(raw, profile="statement")

    assert captured.value.reason_code == ReasonCode.AMBIGUOUS_JSON_TYPE


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decisionId", "not-a-uuid"),
        ("nonce", "short"),
        ("sequence", -1),
    ],
)
def test_security_scalar_constraints_are_strict(field: str, value: object) -> None:
    statement = _statement()
    statement["predicate"][field] = value  # type: ignore[index]
    prevalidated = parse_strict_json(_encode(statement), profile="statement")

    with pytest.raises(ValidationError):
        InTotoStatement.model_validate(prevalidated, strict=True)


def test_authorization_record_id_cannot_contain_email_or_raw_identity() -> None:
    statement = _statement()
    requester = statement["predicate"]["requester"]  # type: ignore[index]
    requester["authorizationId"] = "person@example.com"
    prevalidated = parse_strict_json(_encode(statement), profile="statement")

    with pytest.raises(ValidationError):
        InTotoStatement.model_validate(prevalidated, strict=True)


def test_unknown_nested_field_is_rejected_before_pydantic() -> None:
    statement = _statement()
    statement["predicate"]["policy"]["allow"] = True  # type: ignore[index]

    with pytest.raises(DecisionJsonError) as captured:
        parse_strict_json(_encode(statement), profile="statement")

    assert captured.value.reason_code == ReasonCode.UNKNOWN_FIELD
    assert captured.value.path == "$.predicate.policy"


def test_unknown_reason_code_is_rejected_before_pydantic() -> None:
    statement = _statement()
    statement["predicate"]["claimedReasonCodes"] = ["UNREGISTERED_REASON"]  # type: ignore[index]

    with pytest.raises(DecisionJsonError) as captured:
        parse_strict_json(_encode(statement), profile="statement")

    assert captured.value.reason_code == ReasonCode.INPUT_INVALID


def test_wrong_dsse_payload_type_is_rejected_by_strict_contract() -> None:
    envelope = _dsse()
    envelope["payloadType"] = "application/json"
    prevalidated = parse_strict_json(_encode(envelope), profile="dsse")

    with pytest.raises(ValidationError):
        DsseEnvelope.model_validate(prevalidated, strict=True)


def test_models_reject_fields_not_known_to_the_contract() -> None:
    signed_set = _signed_set()
    mutated = deepcopy(signed_set)
    mutated["unexpected"] = "forbidden"

    with pytest.raises(DecisionJsonError) as captured:
        parse_strict_json(_encode(mutated), profile="signed_set")

    assert captured.value.reason_code == ReasonCode.UNKNOWN_FIELD
