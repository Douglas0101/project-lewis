"""Testes de isolamento do teste e governança de pilotos (PRD P0-03/P0-05/P0-07).

Sem TensorFlow e sem dataset real: guardas são exercitadas antes dos imports lazy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_pilot_predictions import _iter_partition_ids, export_partition
from scripts.run_pilot_cell import (
    EXIT_GATE_FAILED,
    EXIT_OK,
    evaluate_cell_gate,
    find_predecessor_metrics,
)
from src.governance.freeze_manager import create_model_freeze, is_test_authorized

MANIFEST = Path("data/splits/chapman_paired_v2/manifest.json")


def _metrics(pr_auc: float) -> dict:
    return {"metrics": {"macro_pr_auc": pr_auc, "macro_auroc": 0.85}}


# --- P0-03: guarda do teste ------------------------------------------------
def test_export_test_blocked_without_freeze(tmp_path):
    (tmp_path / "backbone_pretrained.keras").write_bytes(b"fake")
    with pytest.raises(RuntimeError, match="bloqueada"):
        export_partition(tmp_path, MANIFEST, "test")
    assert not is_test_authorized(tmp_path)


def test_export_test_authorized_after_freeze(tmp_path):
    checkpoint = tmp_path / "backbone_pretrained.keras"
    checkpoint.write_bytes(b"fake")
    create_model_freeze(tmp_path, architecture="a1", loss="bce")
    assert is_test_authorized(tmp_path)
    # passa pela guarda (falha depois, no modelo fake) — prova que a guarda abriu
    with pytest.raises(Exception) as excinfo:
        export_partition(tmp_path, MANIFEST, "test")
    assert "bloqueada" not in str(excinfo.value)


def test_freeze_is_write_once(tmp_path):
    (tmp_path / "backbone_pretrained.keras").write_bytes(b"fake")
    create_model_freeze(tmp_path, architecture="a1", loss="bce")
    with pytest.raises(RuntimeError, match="write-once"):
        create_model_freeze(tmp_path, architecture="a1", loss="bce")


# --- P0-05: gate da célula com exit codes ----------------------------------
def test_gate_c1_passes_when_above_baseline():
    passed, _ = evaluate_cell_gate("c1", _metrics(0.675), _metrics(0.646))
    assert passed is True


def test_gate_c1_fails_below_baseline_beyond_noise():
    passed, reason = evaluate_cell_gate("c1", _metrics(0.630), _metrics(0.646))
    assert passed is False
    assert "C0" in reason


def test_gate_c1_tolerates_noise_band():
    passed, _ = evaluate_cell_gate("c1", _metrics(0.644), _metrics(0.646))
    assert passed is True  # Δ = −0,002 < banda 0,005


def test_gate_c2_sanity_floor():
    passed, reason = evaluate_cell_gate("c2", _metrics(0.55), _metrics(0.60))
    assert passed is False
    assert "piso" in reason


def test_gate_without_comparison_or_baseline():
    assert evaluate_cell_gate("c0", _metrics(0.64), None)[0] is True
    assert evaluate_cell_gate("c3", _metrics(0.65), None)[0] is True
    # sem predecessora: aviso, não falha
    assert evaluate_cell_gate("c1", _metrics(0.67), None)[0] is True


def test_exit_codes_constants():
    assert EXIT_OK == 0 and EXIT_GATE_FAILED == 3  # PRD RF-QG-003


# --- predecessora (lookup por pilot_status.json) ----------------------------
def test_find_predecessor_metrics(tmp_path, monkeypatch):
    import scripts.run_pilot_cell as rpc

    for name, cell, pr in (("20260802_035117_pretrain_chapman", "c0", 0.6458),
                           ("20260802_043748_pretrain_chapman", "c1", 0.6749)):
        run = tmp_path / "experiments" / name
        (run / "evaluation_v2").mkdir(parents=True)
        (run / "pilot_status.json").write_text(json.dumps({"cell": cell, "status": "PILOT"}))
        (run / "evaluation_v2" / "metrics.json").write_text(json.dumps(_metrics(pr)))
    monkeypatch.setattr(rpc, "PROJECT_ROOT", tmp_path)
    found = find_predecessor_metrics("c2")
    assert found is not None
    assert found["metrics"]["macro_pr_auc"] == pytest.approx(0.6749)


# --- P0-07: IDs alinhados ---------------------------------------------------
def test_iter_partition_ids_aligned_and_sequential(monkeypatch):
    import src.models.chapman_dataset as cd

    def fake_gen(catalog_path, processed_dir, segment_len, seed=None):
        for rec in ("JS1", "JS1", "JS1", "JS2", "JS2", "JS3"):
            yield None, None, rec

    monkeypatch.setattr(cd, "_record_generator", fake_gen)
    ids = list(_iter_partition_ids({"JS1", "JS3"}, 500, None, None))
    assert ids == [
        ("JS1", "JS1#000"),
        ("JS1", "JS1#001"),
        ("JS1", "JS1#002"),
        ("JS3", "JS3#000"),
    ]
