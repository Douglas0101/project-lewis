"""Tests for pretrain provenance/artifacts (FASE 4)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.models.pretrain_provenance import (
    build_provenance,
    compute_per_class_metrics,
    sha256_file,
    write_json,
)


def test_sha256_file_roundtrip(tmp_path):
    import hashlib

    path = tmp_path / "x.bin"
    path.write_bytes(b"lewis")
    assert sha256_file(path) == hashlib.sha256(b"lewis").hexdigest()
    assert sha256_file(tmp_path / "missing.bin") == ""


def test_write_json_returns_matching_hash(tmp_path):
    path = tmp_path / "out.json"
    digest = write_json(path, {"a": 1})
    assert sha256_file(path) == digest
    assert json.loads(path.read_text()) == {"a": 1}


def test_compute_per_class_metrics_schema():
    rng = np.random.default_rng(0)
    y_true = np.array(
        [
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1],
        ],
        dtype=int,
    )
    y_score = np.clip(y_true + rng.normal(0, 0.3, y_true.shape), 0, 1)
    result = compute_per_class_metrics(y_true, y_score)

    assert result["threshold"] == 0.5
    per_class = result["per_class"]
    assert set(per_class) == {"NORM", "CD", "MI", "HYP", "STTC"}
    assert per_class["NORM"]["support"] == 3
    assert per_class["CD"]["support"] == 1
    for cls, metrics in per_class.items():
        for key in ("support", "auc_roc", "auc_pr", "precision", "recall", "f1"):
            assert key in metrics, f"{cls}.{key}"
    # HYP has zero positives -> degenerate AUCs must be None, not crash
    assert per_class["HYP"]["auc_roc"] is None
    assert per_class["HYP"]["auc_pr"] is None


def test_build_provenance_required_keys(tmp_path):
    prov = build_provenance(
        run_id="run_x",
        seed=42,
        deterministic_mode="strict",
        train_records=10,
        val_records=2,
        model_info={"name": "m", "params": 1, "estimated_flatbuffer_kb": 1,
                    "input_shape": [None, 500, 1], "num_classes": 5},
        training_info={"epochs": 1, "batch_size": 4, "steps_per_epoch": 1,
                       "validation_steps": 1, "lr_initial": 1e-3,
                       "optimizer": "adam", "loss": "binary_crossentropy"},
        metrics={"best_epoch": 1, "val_loss": 0.5, "val_auc_roc": 0.6, "val_auc_pr": 0.5},
        qg4={"pass": False, "reason": "x"},
        artifacts={"model": "m.keras"},
        hashes={"model_sha256": "abc"},
    )
    for key in (
        "run_id",
        "timestamp_utc",
        "git_commit",
        "git_branch",
        "python_version",
        "tensorflow_version",
        "gpu_available",
        "deterministic_mode",
        "dataset",
        "model",
        "training",
        "metrics",
        "qg4",
        "artifacts",
        "hashes",
        "paper_alignment",
    ):
        assert key in prov, key
    assert prov["dataset"]["reference"] == "Zheng et al., Scientific Data, 2020"
    assert prov["seed"] == 42


@pytest.mark.slow
def test_wrapper_smoke_produces_strict_artifacts():
    """Integration: wrapper smoke run must pass strict artifact validation."""
    import subprocess

    proc = subprocess.run(
        [".venv/bin/python", "scripts/pretrain_wrapper.py", "--smoke"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-2000:]
    check = subprocess.run(
        [".venv/bin/python", "scripts/validate_pretrain_artifacts.py", "--strict"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert check.returncode == 0, check.stdout + check.stderr


# ---------------------------------------------------------------------------
# F2 — QG4 rotula a métrica de perda real e a época do checkpoint (D3/D4)
# ---------------------------------------------------------------------------


def test_write_gate_and_status_labels_bce_monitor_for_focal(tmp_path):
    from src.models.pretrain_provenance import write_gate_and_status

    best = {"best_epoch": 30, "val_loss": 0.476, "val_auc_roc": 0.834}
    write_gate_and_status(
        experiment_dir=tmp_path,
        best=best,
        qg4_cfg={"min_val_auc_roc_macro": 0.85, "max_val_loss": 0.15},
        qg4_pass=False,
        model_promoted=False,
        gate_loss_metric="val_bce_monitor",
        checkpoint_monitor="val_auc_pr",
    )
    qg4 = json.loads((tmp_path / "qg4_result.json").read_text(encoding="utf-8"))
    assert "val_bce_monitor" in qg4["arms"]
    assert "val_loss" not in qg4["arms"]  # rótulo não pode fingir a loss focal
    assert "val_bce_monitor" in qg4["decision_rule"]
    assert "val_auc_pr" in qg4["decision_rule"]  # época = checkpoint, não min perda

    status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert status["qg4"]["best_epoch"] == 30
    assert status["qg4"]["gate_loss_metric"] == "val_bce_monitor"
    assert status["qg4"]["checkpoint_monitor"] == "val_auc_pr"
    assert "val_bce_monitor" in status["qg4"]["reason"]


def test_write_gate_and_status_bce_defaults_stay_compatible(tmp_path):
    from src.models.pretrain_provenance import write_gate_and_status

    best = {"best_epoch": 25, "val_loss": 0.39, "val_auc_roc": 0.826}
    write_gate_and_status(
        experiment_dir=tmp_path,
        best=best,
        qg4_cfg={"min_val_auc_roc_macro": 0.85, "max_val_loss": 0.15},
        qg4_pass=False,
        model_promoted=False,
    )
    qg4 = json.loads((tmp_path / "qg4_result.json").read_text(encoding="utf-8"))
    assert set(qg4["arms"]) == {"val_auc_roc", "val_loss"}
    assert qg4["pass"] is False


# ---------------------------------------------------------------------------
# F3 — split efetivo + manifesto com hash na proveniência (D5)
# ---------------------------------------------------------------------------


def test_build_provenance_records_effective_split_and_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("TF_ENABLE_ONEDNN_OPTS", "1")
    monkeypatch.delenv("TF_DETERMINISTIC_OPS", raising=False)
    prov = build_provenance(
        run_id="run_y",
        seed=13,
        deterministic_mode="fast",
        train_records=35989,
        val_records=4499,
        model_info={"name": "m", "params": 1, "estimated_flatbuffer_kb": 1,
                    "input_shape": [None, 500, 1], "num_classes": 5},
        training_info={"split_id": "chapman-record-disjoint-paired-v2"},
        metrics={"best_epoch": 30, "val_loss": 0.47, "val_auc_roc": 0.83},
        qg4={"pass": False, "reason": "x"},
        artifacts={"model": "m.keras"},
        hashes={"split_manifest_sha256": "deadbeef"},
        split_policy="paired manifest chapman-record-disjoint-paired-v2 "
                     "(train=0.8/validation=0.1/calibration=0.05/test=0.05)",
        runtime={"profile": "fast", "onednn": True, "deterministic_ops": False},
    )
    assert prov["dataset"]["split_policy"].startswith("paired manifest")
    assert prov["hashes"]["split_manifest_sha256"] == "deadbeef"
    assert prov["runtime"]["profile"] == "fast"
    assert prov["runtime"]["deterministic_ops"] is False


def test_runtime_env_snapshot_reads_effective_env(monkeypatch):
    from src.models.pretrain_provenance import runtime_env_snapshot

    monkeypatch.setenv("TF_ENABLE_ONEDNN_OPTS", "0")
    monkeypatch.setenv("TF_DETERMINISTIC_OPS", "1")
    monkeypatch.setenv("TF_NUM_INTRAOP_THREADS", "2")
    snap = runtime_env_snapshot("strict")
    assert snap["profile"] == "strict"
    assert snap["onednn"] is False
    assert snap["deterministic_ops"] is True
    assert snap["intra_threads"] == "2"
