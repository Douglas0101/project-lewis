"""Integrity tests for frozen E07R pins and critical artifacts (SDD FASE 1).

Validates: freeze manifest hashes, read-only protection of pins, patient
mapping presence, splits v4.0 leakage report status, and that models/ is
untouched relative to the freeze manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

FREEZE_MANIFEST = Path("experiments/stage2_v2.4_research/integrity/e07r_freeze_manifest.json")
LEAKAGE_REPORT = Path("experiments/stage2_v2.4_research/integrity/e07r_split_leakage_report.json")
PATIENT_MAPPING = Path("data/metadata/physionet_mitdb_patient_mapping.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def pins() -> list[dict]:
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    return manifest["pins"]


def test_freeze_manifest_hashes_validate(pins):
    mismatches = [
        p["artifact_path"]
        for p in pins
        if _sha256(Path(p["artifact_path"])) != p["sha256"]
    ]
    assert not mismatches, f"hash mismatch: {mismatches[:5]}"


def test_pins_are_read_only(pins):
    writable = [
        p["artifact_path"]
        for p in pins
        if os.stat(p["artifact_path"]).st_mode & 0o222
    ]
    assert not writable, f"writable pins (DEF-002): {writable[:5]}"


def test_pin_overwrite_is_blocked(pins, tmp_path):
    """Writing over a read-only pin must fail (permission denied)."""
    sample = Path(pins[0]["artifact_path"])
    with pytest.raises(PermissionError):
        sample.write_text("corrupt", encoding="utf-8")


def test_models_dir_pins_frozen(pins):
    model_pins = [p for p in pins if p["artifact_path"].startswith("models/")]
    assert model_pins, "freeze manifest should pin models/"
    for pin in model_pins:
        assert _sha256(Path(pin["artifact_path"])) == pin["sha256"], pin["artifact_path"]


def test_patient_mapping_official_source():
    mapping = json.loads(PATIENT_MAPPING.read_text(encoding="utf-8"))
    assert mapping["mapping_policy"] == "official_evidence_required"
    groups = {tuple(g["record_ids"]) for g in mapping["patient_groups"]}
    assert ("201", "202") in groups


def test_leakage_report_v4_pass():
    report = json.loads(LEAKAGE_REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["patient_disjoint"] is True
    assert report["patient_overlap_found"] is False
    assert report["known_group_201_202_respected"] is True
