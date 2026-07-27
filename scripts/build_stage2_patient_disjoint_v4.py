#!/usr/bin/env python3
"""Publish the authorized Stage 2 r5 custody generation for E07R."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.stage2_research.integrity import hash_canonical
from src.stage2_research.stage2_custody import (
    build_stage2_custody_bundle,
    publish_stage2_custody_generation,
)
from src.training_integrity.integrity import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT_NPZ = PROJECT_ROOT / "data/features/v3.1.0-r4/finetuning_mitbih_family.npz"
PARENT_PARQUET = PROJECT_ROOT / "data/features/v3.1.0-r4/finetuning_mitbih_family.parquet"
EXPECTED_PARENT_NPZ_SHA256 = "d8ce5061634a22aafc01cc7489552b2b4b1112338bba3c870e5ce22486168f57"
EXPECTED_PARENT_PARQUET_SHA256 = "92e0018a59bf9bad945ac833e038377d256414b2ea63486ce0efc614386b22e3"
DEFAULT_TARGET = PROJECT_ROOT / "data/features/v3.1.0-r5-stage2-pd"
PRODUCER_FILES = (
    "src/stage2_research/e07r_contracts.py",
    "src/stage2_research/stage2_custody.py",
    "src/training_integrity/integrity.py",
    "src/training_integrity/preflight.py",
)


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _producer_hash() -> str:
    payload = {
        "git_head": _git_head(),
        "files": {relative: sha256_file(PROJECT_ROOT / relative) for relative in PRODUCER_FILES},
    }
    return hash_canonical(payload)


def _target(value: str) -> Path:
    candidate = (PROJECT_ROOT / value).resolve()
    data_root = (PROJECT_ROOT / "data/features").resolve()
    if candidate.parent != data_root:
        raise argparse.ArgumentTypeError("target must be one direct data/features generation")
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=_target,
        default=DEFAULT_TARGET,
        help="write-once generation path below data/features",
    )
    args = parser.parse_args(argv)
    bundle = build_stage2_custody_bundle(
        PARENT_NPZ,
        PARENT_PARQUET,
        expected_parent_npz_sha256=EXPECTED_PARENT_NPZ_SHA256,
        expected_parent_parquet_sha256=EXPECTED_PARENT_PARQUET_SHA256,
        source_commit=_git_head(),
        source_manifest_hash=_producer_hash(),
    )
    manifest, complete = publish_stage2_custody_generation(args.target, bundle)
    print(
        json.dumps(
            {
                "status": complete.status,
                "target": str(args.target.relative_to(PROJECT_ROOT)),
                "rows": manifest.row_count,
                "records": manifest.record_count,
                "class_counts": manifest.class_counts,
                "manifest_hash": manifest.manifest_hash,
                "marker_hash": complete.marker_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
