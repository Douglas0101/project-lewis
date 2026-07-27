from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from src.stage2_research.e07r_contracts import (
    E07RFreezeManifestV4,
    E07RIntegrityViolationV4,
)
from src.stage2_research.e07r_integrity import (
    E07RIntegrityError,
    E07RIntegrityPaths,
    FreezePinSpec,
    _assemble_e07r_freeze_manifest,
    assert_authorized_split_path,
    guard_e07r_write,
    protect_freeze_pins,
    verify_freeze_pins,
)


def _freeze(root: Path) -> tuple[E07RFreezeManifestV4, Path]:
    pin_specs = (
        FreezePinSpec("src/e07r_source.py", "SOURCE", False),
        FreezePinSpec("data/custody.json", "CUSTODY", True),
        FreezePinSpec("data/identity.json", "IDENTITY", True),
        FreezePinSpec("data/split.json", "SPLIT", True),
        FreezePinSpec("data/governance.json", "GOVERNANCE", True),
        FreezePinSpec("data/quarantine.json", "QUARANTINE", True),
        FreezePinSpec("data/legacy.json", "LEGACY_SENTINEL", False),
    )
    for spec in pin_specs:
        path = root / spec.artifact_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{{"role":"{spec.role}"}}\n', encoding="utf-8")
    artifact = root / "data/custody.json"
    manifest = _assemble_e07r_freeze_manifest(
        root,
        pin_specs,
        custody_manifest_hash="1" * 64,
        patient_mapping_hash="2" * 64,
        split_manifest_hash="3" * 64,
        preauthorization_manifest_hash="4" * 64,
    )
    freeze_path = E07RIntegrityPaths(root).freeze_manifest
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    protect_freeze_pins(root, manifest, include_manifest=freeze_path)
    return manifest, artifact


def _events(root: Path) -> list[E07RIntegrityViolationV4]:
    path = E07RIntegrityPaths(root).violation_log
    return [
        E07RIntegrityViolationV4.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_freeze_contract_rejects_missing_required_roles(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    custody = tmp_path / "custody.json"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    custody.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="every required pin role"):
        _assemble_e07r_freeze_manifest(
            tmp_path,
            (
                FreezePinSpec("source.py", "SOURCE", False),
                FreezePinSpec("custody.json", "CUSTODY", True),
            ),
            custody_manifest_hash="1" * 64,
            patient_mapping_hash="2" * 64,
            split_manifest_hash="3" * 64,
            preauthorization_manifest_hash="4" * 64,
        )


def test_freeze_detects_post_freeze_mutation(tmp_path: Path) -> None:
    manifest, artifact = _freeze(tmp_path)
    verify_freeze_pins(tmp_path, manifest)
    assert stat.S_IMODE(artifact.stat().st_mode) & 0o222 == 0

    artifact.chmod(0o644)
    artifact.write_text('{"status":"FAIL"}\n', encoding="utf-8")

    with pytest.raises(E07RIntegrityError, match="frozen artifact"):
        verify_freeze_pins(tmp_path, manifest)


def test_guard_blocks_frozen_write_and_model_promotion(tmp_path: Path) -> None:
    manifest, artifact = _freeze(tmp_path)

    with pytest.raises(E07RIntegrityError, match="frozen"):
        guard_e07r_write(
            tmp_path,
            artifact,
            workflow="E06_5_PD",
            run_id="cell-001",
        )
    with pytest.raises(E07RIntegrityError, match="promotion"):
        guard_e07r_write(
            tmp_path,
            tmp_path / "models/promoted.keras",
            workflow="E06_5_PD",
            run_id="cell-001",
        )

    events = _events(tmp_path)
    assert [event.event_type for event in events] == [
        "FORBIDDEN_WRITE",
        "MODEL_PROMOTION_ATTEMPT",
    ]
    assert all(event.freeze_manifest_hash == manifest.manifest_hash for event in events)


def test_guard_rejects_legacy_split_and_allows_pd_namespace(tmp_path: Path) -> None:
    _freeze(tmp_path)
    legacy = tmp_path / "experiments/stage2_v2.4_research/splits/outer.json"

    with pytest.raises(E07RIntegrityError, match="split use"):
        assert_authorized_split_path(
            tmp_path,
            legacy,
            workflow="E06_5_PD",
            run_id="matrix",
        )

    guard_e07r_write(
        tmp_path,
        tmp_path / "experiments/stage2_v2.4_research/E06_5_PD/cell-001/metrics.json",
        workflow="E06_5_PD",
        run_id="cell-001",
    )
    assert _events(tmp_path)[0].event_type == "LEGACY_SPLIT_USE"
