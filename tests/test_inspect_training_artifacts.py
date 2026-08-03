"""Testes do inspetor de artefatos de treino (scripts/inspect_training_artifacts.py).

Fixtures sintéticas em tmp_path — nenhum run real é tocado (read-only por contrato).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts.inspect_training_artifacts import (
    EXIT_GRAVE,
    EXIT_OK,
    analyze_history,
    check_predictions,
    compare_table,
    inspect_keras_zip,
    inspect_run,
    main,
    select_runs,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _fake_keras(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("metadata.json", json.dumps({"keras_version": "3.0.0", "backend": "tensorflow"}))
        z.writestr(
            "config.json", json.dumps({"class_name": "Functional", "config": {"layers": []}})
        )
        z.writestr("model.weights.h5", b"fake-weights")


def _history(n: int = 30, focal: bool = False) -> dict:
    """History com argmax val_auc_pr na última época e mínimos de perda antes."""
    hist = {
        "loss": [0.5 - i * 0.01 for i in range(n)],
        "val_loss": [0.4 - min(i, 12) * 0.01 for i in range(n)],
        "val_auc_pr": [0.5 + i * 0.005 for i in range(n)],
        "val_auc_roc": [0.7 + i * 0.005 for i in range(n)],
        "learning_rate": [1e-3] * n,
    }
    if focal:
        hist["val_bce_monitor"] = [0.47 - min(i, 12) * 0.01 for i in range(n)]
        hist["bce_monitor"] = [0.5 - i * 0.01 for i in range(n)]
    return {"history": hist}


def _base_run(run: Path, *, loss: str = "bce", es_metric: str = "val_auc_pr") -> None:
    """Run consistente (schema pós-correção): sem nenhuma issue grave."""
    run.mkdir(parents=True)
    _fake_keras(run / "backbone_pretrained.keras")
    _write_json(
        run / "config.json",
        {
            "architecture": "a1",
            "loss": loss,
            "seed": 13,
            "epochs": 30,
            "name": "a1_stable",
        },
    )
    _write_json(
        run / "provenance.json",
        {
            "git_commit": "abc",
            "seed": 13,
            "deterministic_mode": "fast",
            "onednn_enabled": True,
            "runtime": {"profile": "fast", "onednn": True, "deterministic_ops": False},
            "dataset": {"split_policy": "paired manifest chapman-record-disjoint-paired-v2"},
            "training": {
                "architecture": "a1",
                "loss": loss,
                "seed": 13,
                "split_id": "chapman-record-disjoint-paired-v2",
                "early_stopping_metric": es_metric,
            },
            "metrics": {"best_epoch": 30, "val_loss": 0.1, "val_auc_roc": 0.86},
            "qg4": {"pass": True},
            "hashes": {"split_manifest_sha256": "deadbeef"},
        },
    )
    _write_json(run / "qg4_result.json", {"gate": "QG4", "pass": True, "arms": {}})
    _write_json(
        run / "run_status.json",
        {"execution_success": True, "qg4": {"pass": True, "best_epoch": 30}},
    )
    _write_json(run / "history.json", _history(focal=(loss != "bce")))


# --- analyze_history ----------------------------------------------------------
def test_analyze_history_finds_argmax_and_argmin():
    out = analyze_history(_history(focal=True))
    assert out["epochs"] == 30
    assert out["best_val_auc_pr"]["epoch"] == 30
    assert out["best_val_bce_monitor"]["epoch"] == 13
    assert "val_bce_monitor" in out["metrics"]


# --- inspect_keras_zip --------------------------------------------------------
def test_inspect_keras_zip_reads_v3_format(tmp_path):
    keras = tmp_path / "backbone_pretrained.keras"
    _fake_keras(keras)
    out = inspect_keras_zip(keras)
    assert out["keras_class"] == "Functional"
    assert "model.weights.h5" in out["entries"]
    assert "erro" not in out


def test_inspect_keras_zip_flags_bad_file(tmp_path):
    bad = tmp_path / "backbone_pretrained.keras"
    bad.write_bytes(b"not-a-zip")
    assert "erro" in inspect_keras_zip(bad)


# --- run saudável ---------------------------------------------------------------
def test_healthy_run_has_no_grave_issues(tmp_path):
    run = tmp_path / "20260802_990000_pretrain_chapman"
    _base_run(run)
    ficha = inspect_run(run, verify_hashes=True, inspect_model=True)
    graves = [m for s, m in ficha["issues"] if s == "grave"]
    assert graves == []
    assert ficha["checkpoint_epoch"] == 30
    assert ficha["qg4_epoch"] == 30
    assert ficha["keras_zip"]["keras_class"] == "Functional"


# --- divergências graves --------------------------------------------------------
def test_test_npz_without_freeze_is_grave(tmp_path):
    run = tmp_path / "20260802_990001_pretrain_chapman"
    _base_run(run)
    preds = run / "evaluation_v2" / "predictions"
    preds.mkdir(parents=True)
    np.savez(
        preds / "test.npz",
        y_score=np.zeros((4, 5), dtype=np.float32),
        y_true=np.zeros((4, 5), dtype=np.float32),
    )
    ficha = inspect_run(run)
    assert any(s == "grave" and "test.npz" in m and "RF-DATA-005" in m for s, m in ficha["issues"])


def test_hash_mismatch_is_grave(tmp_path):
    run = tmp_path / "20260802_990002_pretrain_chapman"
    _base_run(run)
    prov = json.loads((run / "provenance.json").read_text())
    prov["hashes"]["model_sha256"] = "0" * 64
    _write_json(run / "provenance.json", prov)
    ficha = inspect_run(run, verify_hashes=True)
    assert any(s == "grave" and "model_sha256" in m for s, m in ficha["issues"])
    assert ficha["hash_checks"]["model"] is False


def test_corrupted_json_is_grave(tmp_path):
    run = tmp_path / "20260802_990003_pretrain_chapman"
    _base_run(run)
    (run / "run_status.json").write_text("{invalid", encoding="utf-8")
    ficha = inspect_run(run)
    assert any(s == "grave" and "corrompido" in m for s, m in ficha["issues"])


def test_qg4_without_checkpoint_is_grave(tmp_path):
    run = tmp_path / "20260802_990004_pretrain_chapman"
    _base_run(run)
    (run / "backbone_pretrained.keras").unlink()
    ficha = inspect_run(run)
    assert any(s == "grave" and "sem checkpoint" in m for s, m in ficha["issues"])


# --- avisos (D2/D3/D4/D5/D7) -----------------------------------------------------
def test_runtime_label_mismatch_is_aviso(tmp_path):
    run = tmp_path / "20260802_990005_pretrain_chapman"
    _base_run(run)
    prov = json.loads((run / "provenance.json").read_text())
    prov["deterministic_mode"] = "strict"  # rótulo legado sem bloco runtime
    del prov["runtime"]
    _write_json(run / "provenance.json", prov)
    _write_json(run / "pilot_status.json", {"cell": "c1", "runtime_profile": "fast"})
    ficha = inspect_run(run)
    assert any(s == "aviso" and "runtime_profile=fast" in m for s, m in ficha["issues"])
    assert any(s == "aviso" and "onednn_enabled=true" in m for s, m in ficha["issues"])


def test_qg4_epoch_diverging_from_checkpoint_is_aviso(tmp_path):
    run = tmp_path / "20260802_990006_pretrain_chapman"
    _base_run(run, loss="focal")
    _write_json(
        run / "run_status.json",
        {"execution_success": True, "qg4": {"pass": False, "best_epoch": 13}},
    )
    _write_json(
        run / "qg4_result.json",
        {"gate": "QG4", "pass": False, "arms": {"val_loss": {"observed": 0.47}}},
    )
    ficha = inspect_run(run)
    assert any(
        s == "aviso" and "época do QG4 (13)" in m and "checkpoint (30," in m
        for s, m in ficha["issues"]
    )
    assert any(s == "aviso" and "val_bce_monitor" in m for s, m in ficha["issues"])  # D4


def test_stale_split_policy_is_aviso(tmp_path):
    run = tmp_path / "20260802_990007_pretrain_chapman"
    _base_run(run)
    prov = json.loads((run / "provenance.json").read_text())
    prov["dataset"]["split_policy"] = "record_disjoint (val_ratio=0.1, seeded shuffle)"
    prov["hashes"] = {}
    _write_json(run / "provenance.json", prov)
    ficha = inspect_run(run)
    assert any(s == "aviso" and "split_manifest_sha256" in m for s, m in ficha["issues"])
    assert any(s == "aviso" and "split_policy" in m for s, m in ficha["issues"])


def test_npz_without_ids_is_aviso_and_overlap_is_grave(tmp_path):
    run = tmp_path / "20260802_990008_pretrain_chapman"
    preds = run / "evaluation_v2" / "predictions"
    preds.mkdir(parents=True)
    np.savez(
        preds / "validation.npz",
        y_score=np.full((4, 5), 0.5, dtype=np.float32),
        y_true=np.zeros((4, 5), dtype=np.float32),
    )
    issues: list = []
    summary = check_predictions(run, issues)
    assert any(s == "aviso" and "sem IDs" in m for s, m in issues)
    # com IDs sobrepostos → grave
    rid = np.array(["JS1", "JS1", "JS2", "JS2"])
    sid = np.array(["JS1#000", "JS1#001", "JS2#000", "JS2#001"])
    np.savez(
        preds / "calibration.npz",
        y_score=np.full((4, 5), 0.5, dtype=np.float32),
        y_true=np.zeros((4, 5), dtype=np.float32),
        record_ids=rid,
        segment_ids=sid,
        patient_ids=rid,
    )
    np.savez(
        preds / "validation.npz",
        y_score=np.full((4, 5), 0.5, dtype=np.float32),
        y_true=np.zeros((4, 5), dtype=np.float32),
        record_ids=rid,
        segment_ids=sid,
        patient_ids=rid,
    )
    issues = []
    summary = check_predictions(run, issues)
    assert any(s == "grave" and "overlap" in m for s, m in issues)
    assert summary["overlap_calibration_validation"] == 2  # intersect1d é único: {JS1, JS2}


def test_nonfinite_scores_are_grave(tmp_path):
    run = tmp_path / "20260802_990009_pretrain_chapman"
    preds = run / "evaluation_v2" / "predictions"
    preds.mkdir(parents=True)
    scores = np.full((4, 5), 0.5, dtype=np.float32)
    scores[0, 0] = np.nan
    np.savez(preds / "validation.npz", y_score=scores, y_true=np.zeros((4, 5), dtype=np.float32))
    issues: list = []
    check_predictions(run, issues)
    assert any(s == "grave" and "NaN/Inf" in m for s, m in issues)


# --- seleção e comparação ---------------------------------------------------------
def test_select_runs_latest_tokens(tmp_path):
    exp = tmp_path / "experiments"
    for name, cell, smoke in (
        ("20260802_000001_pretrain_chapman", "c0", False),
        ("20260802_000002_pretrain_chapman", "c1", False),
        ("20260802_000003_pretrain_chapman", "c1", True),
    ):
        run = exp / name
        run.mkdir(parents=True)
        _write_json(run / "pilot_status.json", {"cell": cell, "smoke": smoke})
    selected = select_runs(exp, ["latest-c1", "latest-smoke"])
    assert [p.name for p in selected] == [
        "20260802_000002_pretrain_chapman",
        "20260802_000003_pretrain_chapman",
    ]


def test_compare_table_matrix_fields(tmp_path):
    run = tmp_path / "20260802_990010_pretrain_chapman"
    _base_run(run)
    _write_json(
        run / "evaluation_v2" / "metrics.json",
        {
            "protocol_status": "PROSPECTIVE",
            "split_id": "chapman-record-disjoint-paired-v2",
            "metrics": {
                "macro_pr_auc": 0.69,
                "macro_auroc": 0.85,
                "ece_post_calibration": 0.02,
                "temperature": 0.96,
            },
        },
    )
    ficha = inspect_run(run)
    row = compare_table([ficha])[0]
    assert row["cell"] is None or row["cell"] == "c1"
    assert row["checkpoint_epoch"] == 30
    assert row["macro_pr_auc"] == pytest.approx(0.69)
    assert row["n_grave"] == 0


# --- CLI ---------------------------------------------------------------------------
def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    ok_run = tmp_path / "20260802_990011_pretrain_chapman"
    _base_run(ok_run)
    monkeypatch.setattr("sys.argv", ["inspect", "--run-dir", str(ok_run)])
    assert main() == EXIT_OK
    capsys.readouterr()

    bad_run = tmp_path / "20260802_990012_pretrain_chapman"
    _base_run(bad_run)
    (bad_run / "run_status.json").write_text("{bad", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["inspect", "--run-dir", str(bad_run)])
    assert main() == EXIT_GRAVE
    capsys.readouterr()


def test_main_writes_output_file(tmp_path, monkeypatch):
    run = tmp_path / "20260802_990013_pretrain_chapman"
    _base_run(run)
    out = tmp_path / "audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "inspect",
            "--run-dir",
            str(run),
            "--compare-runs",
            "--output",
            str(out),
            "--format",
            "json",
        ],
    )
    assert main() == EXIT_OK
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["runs"][0]["run_id"] == run.name
    assert payload["comparison"][0]["checkpoint_epoch"] == 30
