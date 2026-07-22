"""Pre-Pydantic JSON rejection boundary for authenticated decisions."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal, NoReturn, cast

from src.security.authenticated_decision_contracts import ReasonCode, SignerRole

MAX_SIGNED_SET_BYTES = 8 * 1024 * 1024
MAX_DSSE_ENVELOPE_BYTES = 2 * 1024 * 1024
MAX_STATEMENT_BYTES = 512 * 1024
MAX_JSON_DEPTH = 24
MAX_BUNDLES = 8
MAX_SUBJECTS = 16
MAX_EVIDENCE = 32
MAX_WAIVERS = 16
MAX_REASON_CODES = 32
MAX_STRING_LENGTH = 4096
MAX_BUNDLE_BASE64_LENGTH = 6_000_000
MAX_PAYLOAD_BASE64_LENGTH = 1_000_000

JsonProfile = Literal["signed_set", "dsse", "statement"]


class DecisionJsonError(ValueError):
    """Rejected JSON carrying a controlled reason code and safe path."""

    def __init__(self, reason_code: ReasonCode, message: str, *, path: str = "$") -> None:
        super().__init__(f"{path}: {message}")
        self.reason_code = reason_code
        self.path = path


def parse_strict_json(raw: bytes, *, profile: JsonProfile) -> dict[str, Any]:
    """Parse JSON once and reject ambiguity before any Pydantic model is called."""
    if not isinstance(raw, bytes):
        raise DecisionJsonError(ReasonCode.AMBIGUOUS_JSON_TYPE, "input must be bytes")
    maximum = {
        "signed_set": MAX_SIGNED_SET_BYTES,
        "dsse": MAX_DSSE_ENVELOPE_BYTES,
        "statement": MAX_STATEMENT_BYTES,
    }[profile]
    if not raw or len(raw) > maximum:
        raise DecisionJsonError(
            ReasonCode.RESOURCE_LIMIT_EXCEEDED,
            f"input size must be between 1 and {maximum} bytes",
        )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise DecisionJsonError(ReasonCode.INPUT_INVALID, "input is not UTF-8") from error
    if text.startswith("\ufeff"):
        raise DecisionJsonError(ReasonCode.INPUT_INVALID, "UTF-8 BOM is prohibited")

    def reject_constant(value: str) -> NoReturn:
        raise DecisionJsonError(
            ReasonCode.NON_FINITE_JSON,
            f"non-finite constant is prohibited: {value}",
        )

    def reject_float(value: str) -> NoReturn:
        raise DecisionJsonError(
            ReasonCode.AMBIGUOUS_JSON_TYPE,
            f"floating-point JSON number is prohibited: {value}",
        )

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=reject_constant,
            parse_float=reject_float,
        )
    except Exception as error:
        if isinstance(error, DecisionJsonError):
            raise
        if isinstance(error, json.JSONDecodeError):
            raise DecisionJsonError(
                ReasonCode.INPUT_INVALID,
                f"malformed JSON at line {error.lineno} column {error.colno}",
            ) from error
        if isinstance(error, (ValueError, RecursionError, MemoryError)):
            raise DecisionJsonError(
                ReasonCode.RESOURCE_LIMIT_EXCEEDED,
                "JSON decoder resource limit exceeded",
            ) from error
        raise DecisionJsonError(
            ReasonCode.INPUT_INVALID,
            "JSON decoder failed closed",
        ) from error
    try:
        _check_depth(parsed, depth=0, path="$")
        validators: dict[JsonProfile, Callable[[Any], None]] = {
            "signed_set": _validate_signed_set,
            "dsse": _validate_dsse,
            "statement": _validate_statement,
        }
        validators[profile](parsed)
        if profile == "statement":
            _freeze_statement_enums(parsed)
        frozen = _freeze_json_arrays(parsed)
        return cast(dict[str, Any], frozen)
    except Exception as error:
        if isinstance(error, DecisionJsonError):
            raise
        if isinstance(error, (RecursionError, MemoryError)):
            raise DecisionJsonError(
                ReasonCode.RESOURCE_LIMIT_EXCEEDED,
                "JSON structural validation resource limit exceeded",
            ) from error
        raise DecisionJsonError(
            ReasonCode.INPUT_INVALID,
            "JSON structural validation failed closed",
        ) from error


def _freeze_statement_enums(value: Any) -> None:
    """Convert already type-checked controlled strings to enum instances explicitly."""
    root = _object(value, path="$")
    predicate = _object(root["predicate"], path="$.predicate")
    policy = _object(predicate["policy"], path="$.predicate.policy")
    quorum = _object(policy["quorum"], path="$.predicate.policy.quorum")
    try:
        quorum["requiredRoles"] = [SignerRole(item) for item in quorum["requiredRoles"]]
        predicate["claimedReasonCodes"] = [
            ReasonCode(item) for item in predicate["claimedReasonCodes"]
        ]
    except ValueError as error:
        raise DecisionJsonError(
            ReasonCode.INPUT_INVALID,
            "controlled enum value is not registered",
        ) from error


def _freeze_json_arrays(value: Any) -> Any:
    """Convert already-validated JSON arrays to immutable tuples without coercing scalars."""
    if type(value) is list:
        return tuple(_freeze_json_arrays(item) for item in value)
    if type(value) is dict:
        return {key: _freeze_json_arrays(item) for key, item in value.items()}
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DecisionJsonError(
                ReasonCode.DUPLICATE_JSON_KEY,
                f"duplicate key: {key}",
            )
        result[key] = value
    return result


def _check_depth(value: Any, *, depth: int, path: str) -> None:
    if depth > MAX_JSON_DEPTH:
        raise DecisionJsonError(
            ReasonCode.RESOURCE_LIMIT_EXCEEDED,
            f"JSON nesting exceeds {MAX_JSON_DEPTH}",
            path=path,
        )
    if type(value) is dict:
        for key, item in value.items():
            _check_depth(item, depth=depth + 1, path=f"{path}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            _check_depth(item, depth=depth + 1, path=f"{path}[{index}]")
    elif type(value) not in {str, int, bool, type(None)}:
        raise DecisionJsonError(
            ReasonCode.AMBIGUOUS_JSON_TYPE,
            f"unsupported JSON type: {type(value).__name__}",
            path=path,
        )


def _validate_signed_set(value: Any) -> None:
    root = _exact_object(value, {"format", "bundles"}, path="$")
    _string(root["format"], path="$.format")
    bundles = _array(root["bundles"], path="$.bundles", minimum=1, maximum=MAX_BUNDLES)
    for index, item in enumerate(bundles):
        path = f"$.bundles[{index}]"
        bundle = _exact_object(
            item,
            {"bundleMediaType", "bundleBase64", "bundleSha256"},
            path=path,
        )
        _string(bundle["bundleMediaType"], path=f"{path}.bundleMediaType")
        _string(
            bundle["bundleBase64"],
            path=f"{path}.bundleBase64",
            maximum=MAX_BUNDLE_BASE64_LENGTH,
        )
        _string(bundle["bundleSha256"], path=f"{path}.bundleSha256")


def _validate_dsse(value: Any) -> None:
    root = _exact_object(value, {"payloadType", "payload", "signatures"}, path="$")
    _string(root["payloadType"], path="$.payloadType")
    _string(root["payload"], path="$.payload", maximum=MAX_PAYLOAD_BASE64_LENGTH)
    signatures = _array(root["signatures"], path="$.signatures", minimum=1, maximum=1)
    for index, item in enumerate(signatures):
        path = f"$.signatures[{index}]"
        signature = _object_with_optional(
            item,
            required={"sig"},
            optional={"keyid"},
            path=path,
        )
        _string(signature["sig"], path=f"{path}.sig", maximum=MAX_PAYLOAD_BASE64_LENGTH)
        if "keyid" in signature:
            _string(signature["keyid"], path=f"{path}.keyid", maximum=256, allow_empty=True)


def _validate_statement(value: Any) -> None:
    root = _exact_object(value, {"_type", "subject", "predicateType", "predicate"}, path="$")
    _string(root["_type"], path="$._type")
    _string(root["predicateType"], path="$.predicateType")
    subjects = _array(root["subject"], path="$.subject", minimum=1, maximum=MAX_SUBJECTS)
    for index, item in enumerate(subjects):
        path = f"$.subject[{index}]"
        subject = _exact_object(item, {"name", "digest"}, path=path)
        _string(subject["name"], path=f"{path}.name", maximum=256)
        digest = _exact_object(subject["digest"], {"sha256"}, path=f"{path}.digest")
        _string(digest["sha256"], path=f"{path}.digest.sha256", maximum=64)
    _validate_predicate(root["predicate"], path="$.predicate")


def _validate_predicate(value: Any, *, path: str) -> None:
    predicate = _exact_object(
        value,
        {
            "decisionId",
            "nonce",
            "sequence",
            "scope",
            "requester",
            "policy",
            "evidence",
            "waivers",
            "validity",
            "claimedReasonCodes",
        },
        path=path,
    )
    for key in ("decisionId", "nonce", "scope"):
        _string(predicate[key], path=f"{path}.{key}", maximum=256)
    _integer(predicate["sequence"], path=f"{path}.sequence")

    requester = _exact_object(
        predicate["requester"],
        {"authorizationId"},
        path=f"{path}.requester",
    )
    _string(
        requester["authorizationId"],
        path=f"{path}.requester.authorizationId",
        maximum=256,
    )

    policy = _exact_object(
        predicate["policy"],
        {"policyId", "version", "sha256", "quorum"},
        path=f"{path}.policy",
    )
    for key in ("policyId", "version", "sha256"):
        _string(policy[key], path=f"{path}.policy.{key}", maximum=256)
    quorum = _exact_object(
        policy["quorum"],
        {"threshold", "requiredRoles"},
        path=f"{path}.policy.quorum",
    )
    _integer(quorum["threshold"], path=f"{path}.policy.quorum.threshold")
    roles = _array(
        quorum["requiredRoles"],
        path=f"{path}.policy.quorum.requiredRoles",
        minimum=2,
        maximum=8,
    )
    for index, role in enumerate(roles):
        _string(role, path=f"{path}.policy.quorum.requiredRoles[{index}]")

    evidence = _array(
        predicate["evidence"],
        path=f"{path}.evidence",
        minimum=1,
        maximum=MAX_EVIDENCE,
    )
    for index, item in enumerate(evidence):
        item_path = f"{path}.evidence[{index}]"
        reference = _exact_object(item, {"name", "sha256", "mediaType"}, path=item_path)
        for key in ("name", "sha256", "mediaType"):
            _string(reference[key], path=f"{item_path}.{key}")

    waivers = _array(
        predicate["waivers"],
        path=f"{path}.waivers",
        minimum=0,
        maximum=MAX_WAIVERS,
    )
    for index, item in enumerate(waivers):
        item_path = f"{path}.waivers[{index}]"
        waiver = _object_with_optional(
            item,
            required={"code", "justification"},
            optional={"evidenceName"},
            path=item_path,
        )
        _string(waiver["code"], path=f"{item_path}.code", maximum=256)
        _string(waiver["justification"], path=f"{item_path}.justification")
        if "evidenceName" in waiver and waiver["evidenceName"] is not None:
            _string(waiver["evidenceName"], path=f"{item_path}.evidenceName")

    validity = _exact_object(
        predicate["validity"],
        {"issuedOn", "notBefore", "expiresOn"},
        path=f"{path}.validity",
    )
    for key in ("issuedOn", "notBefore", "expiresOn"):
        _string(validity[key], path=f"{path}.validity.{key}", maximum=64)

    reasons = _array(
        predicate["claimedReasonCodes"],
        path=f"{path}.claimedReasonCodes",
        minimum=1,
        maximum=MAX_REASON_CODES,
    )
    for index, reason in enumerate(reasons):
        _string(reason, path=f"{path}.claimedReasonCodes[{index}]", maximum=256)


def _exact_object(value: Any, keys: set[str], *, path: str) -> dict[str, Any]:
    return _object_with_optional(value, required=keys, optional=set(), path=path)


def _object_with_optional(
    value: Any,
    *,
    required: set[str],
    optional: set[str],
    path: str,
) -> dict[str, Any]:
    result = _object(value, path=path)
    unknown = set(result) - required - optional
    missing = required - set(result)
    if unknown:
        raise DecisionJsonError(
            ReasonCode.UNKNOWN_FIELD,
            f"unknown fields: {sorted(unknown)}",
            path=path,
        )
    if missing:
        raise DecisionJsonError(
            ReasonCode.INPUT_INVALID,
            f"missing fields: {sorted(missing)}",
            path=path,
        )
    return result


def _object(value: Any, *, path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise DecisionJsonError(
            ReasonCode.AMBIGUOUS_JSON_TYPE,
            "expected object",
            path=path,
        )
    return value


def _array(
    value: Any,
    *,
    path: str,
    minimum: int,
    maximum: int,
) -> list[Any]:
    if type(value) is not list:
        raise DecisionJsonError(
            ReasonCode.AMBIGUOUS_JSON_TYPE,
            "expected array",
            path=path,
        )
    if not minimum <= len(value) <= maximum:
        raise DecisionJsonError(
            ReasonCode.RESOURCE_LIMIT_EXCEEDED,
            f"array length must be between {minimum} and {maximum}",
            path=path,
        )
    return value


def _string(
    value: Any,
    *,
    path: str,
    maximum: int = MAX_STRING_LENGTH,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str:
        raise DecisionJsonError(
            ReasonCode.AMBIGUOUS_JSON_TYPE,
            "expected string",
            path=path,
        )
    if (not allow_empty and not value) or len(value) > maximum:
        raise DecisionJsonError(
            ReasonCode.RESOURCE_LIMIT_EXCEEDED,
            f"string length must be {'0' if allow_empty else '1'}..{maximum}",
            path=path,
        )
    return value


def _integer(value: Any, *, path: str) -> int:
    if type(value) is not int:
        raise DecisionJsonError(
            ReasonCode.AMBIGUOUS_JSON_TYPE,
            "expected integer (boolean/coercion prohibited)",
            path=path,
        )
    return value
