"""Canonical fail-closed CLI for advanced-training v3.1 preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from src.training_integrity.config import load_advanced_training_config
from src.training_integrity.contracts import (
    PreflightEvidenceBundle,
    PreflightReportPublication,
    PretrainingGateMarker,
)
from src.training_integrity.integrity import (
    hash_canonical,
    resolve_project_path,
    sha256_file,
    verify_detached_sha256,
)
from src.training_integrity.preflight import (
    run_project_preflight,
    verify_canonical_preflight_execution,
    verify_complete_component_snapshot,
)

DEFAULT_CONFIG = Path("config/advanced_training_v3.1.yaml")
EXIT_OK = 0
EXIT_ARGUMENT = 2
EXIT_INTEGRITY = 3
EXIT_BLOCKED = 10


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.advanced_training_v3",
        description="Project-Lewis advanced-training v3.1 integrity gate",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="execute pre-training integrity checks")
    preflight.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    preflight.add_argument(
        "--publish-splits",
        action="store_true",
        help="publish the immutable patient-aware v3.1 split bundle",
    )
    status = subparsers.add_parser("status", help="read the persisted preflight decision")
    status.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    verify = subparsers.add_parser(
        "verify-generation",
        help="recompute the generation identity and reject drift",
    )
    verify.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def _load_bundle(config_path: Path) -> PreflightEvidenceBundle:
    config, project_root = load_advanced_training_config(config_path)
    report_path = resolve_project_path(project_root, config.report_json)
    markdown_path = resolve_project_path(project_root, config.report_markdown)
    completion_path = resolve_project_path(project_root, config.report_completion_marker)
    gate_path = resolve_project_path(project_root, config.pretraining_gate_marker)
    try:
        verify_detached_sha256(completion_path)
        publication = PreflightReportPublication.model_validate_json(
            completion_path.read_text(encoding="utf-8")
        )
        verify_detached_sha256(report_path)
        verify_detached_sha256(markdown_path)
        bundle = PreflightEvidenceBundle.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError, ValueError) as error:
        raise ValueError(f"cannot load persisted preflight evidence: {report_path}") from error

    expected_publication = PreflightReportPublication(
        schema_version="preflight-report-publication-v3.1.0",
        generation_id=bundle.generation_manifest.generation_id,
        evidence_bundle_hash=hash_canonical("training-preflight-evidence", bundle),
        report_json_sha256=sha256_file(report_path),
        report_markdown_sha256=sha256_file(markdown_path),
        status="REPORT_BUNDLE_COMPLETE",
    )
    if publication != expected_publication:
        raise ValueError("report completion marker does not bind the report bundle")

    identity_hash = hash_canonical("patient-identity", bundle.patient_identity_manifest)
    split_hash = hash_canonical("patient-split", bundle.patient_split_manifest)
    if bundle.patient_split_manifest.patient_identity_hash != identity_hash:
        raise ValueError("patient identity hash does not bind the persisted split")
    if bundle.generation_manifest.patient_split_hash != split_hash:
        raise ValueError("generation manifest does not bind the persisted split")
    if bundle.generation_manifest.training_config_hash != sha256_file(config_path.resolve()):
        raise ValueError("training config hash drift")
    verify_complete_component_snapshot(project_root, bundle)
    verify_canonical_preflight_execution(config_path.resolve(), bundle)

    if bundle.preflight_report.training_allowed:
        try:
            verify_detached_sha256(gate_path)
            marker = PretrainingGateMarker.model_validate_json(
                gate_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValidationError, ValueError) as error:
            raise ValueError(f"invalid pretraining gate marker: {gate_path}") from error
        expected = PretrainingGateMarker(
            schema_version="pretraining-gate-v3.1.0",
            generation_id=bundle.generation_manifest.generation_id,
            evidence_bundle_hash=hash_canonical("training-preflight-evidence", bundle),
            generation_manifest_hash=hash_canonical(
                "training-generation", bundle.generation_manifest
            ),
            preflight_report_hash=hash_canonical("training-preflight", bundle.preflight_report),
            status="TRAINING_PROVENANCE_VALIDATED",
        )
        if marker != expected:
            raise ValueError("pretraining gate marker does not bind persisted evidence")
    elif gate_path.exists() or gate_path.with_name(f"{gate_path.name}.sha256").exists():
        raise ValueError("blocked preflight has an orphan pass marker")
    return bundle


def _summary(bundle: PreflightEvidenceBundle) -> dict[str, object]:
    report = bundle.preflight_report
    return {
        "generation_id": report.generation_id,
        "final_state": report.final_state,
        "training_allowed": report.training_allowed,
        "blocking_codes": list(report.blocking_codes),
        "confirmatory_patients": bundle.patient_identity_manifest.confirmatory_patient_count,
        "quarantined_records": bundle.patient_identity_manifest.quarantined_record_count,
        "legacy_cross_fold_patients": bundle.legacy_leakage_audit.cross_fold_patient_count,
    }


def _cmd_preflight(args: argparse.Namespace) -> int:
    bundle = run_project_preflight(
        args.config,
        publish_splits=bool(args.publish_splits),
        write_reports=True,
    )
    print(json.dumps(_summary(bundle), sort_keys=True))
    return EXIT_OK if bundle.preflight_report.training_allowed else EXIT_BLOCKED


def _cmd_status(args: argparse.Namespace) -> int:
    bundle = _load_bundle(args.config)
    print(json.dumps(_summary(bundle), sort_keys=True))
    return EXIT_OK if bundle.preflight_report.training_allowed else EXIT_BLOCKED


def _comparable_bundle(bundle: PreflightEvidenceBundle) -> dict[str, object]:
    payload = bundle.model_dump(mode="json")
    payload.pop("generated_at_utc", None)
    return payload


def _cmd_verify(args: argparse.Namespace) -> int:
    stored = _load_bundle(args.config)
    current = run_project_preflight(
        args.config,
        publish_splits=False,
        write_reports=False,
    )
    if _comparable_bundle(stored) != _comparable_bundle(current):
        print(
            json.dumps(
                {
                    "generation_id": stored.generation_manifest.generation_id,
                    "final_state": "GENERATION_DRIFT_DETECTED",
                },
                sort_keys=True,
            )
        )
        return EXIT_INTEGRITY
    print(json.dumps(_summary(stored), sort_keys=True))
    return EXIT_OK if stored.preflight_report.training_allowed else EXIT_BLOCKED


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            return _cmd_preflight(args)
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "verify-generation":
            return _cmd_verify(args)
        raise ValueError(f"unsupported command: {args.command}")
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(
            json.dumps({"final_state": "REVIEW_REQUIRED", "error": str(error)}),
            file=sys.stderr,
        )
        return EXIT_INTEGRITY


if __name__ == "__main__":
    raise SystemExit(main())
