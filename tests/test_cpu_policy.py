"""Testes da política de runtime CPU-only (PRD RF-CPU-001..003)."""

from __future__ import annotations

import os
import sys

import pytest

from src.runtime import cpu_policy


@pytest.fixture(autouse=True)
def _restore_env():
    saved = {
        k: os.environ.get(k)
        for k in (
            "CUDA_VISIBLE_DEVICES",
            "TF_ENABLE_ONEDNN_OPTS",
            "TF_DETERMINISTIC_OPS",
            "TF_NUM_INTRAOP_THREADS",
            "TF_NUM_INTEROP_THREADS",
        )
    }
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_apply_fast_sets_env():
    cfg = cpu_policy.apply("fast", _allow_imported_tf=True)
    assert cfg["onednn"] is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["TF_ENABLE_ONEDNN_OPTS"] == "1"
    assert "TF_DETERMINISTIC_OPS" not in os.environ
    assert os.environ["TF_NUM_INTRAOP_THREADS"] == "2"
    assert os.environ["TF_NUM_INTEROP_THREADS"] == "1"


def test_apply_strict_sets_env():
    cfg = cpu_policy.apply("strict", _allow_imported_tf=True)
    assert cfg["onednn"] is False
    assert os.environ["TF_ENABLE_ONEDNN_OPTS"] == "0"
    assert os.environ["TF_DETERMINISTIC_OPS"] == "1"


def test_apply_unknown_profile_fails():
    with pytest.raises(ValueError, match="perfil desconhecido"):
        cpu_policy.apply("gpu", _allow_imported_tf=True)


def test_apply_after_tf_import_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", object())
    with pytest.raises(RuntimeError, match="TensorFlow já importado"):
        cpu_policy.apply("fast")


def test_verify_fails_when_gpu_visible(monkeypatch):
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    monkeypatch.setattr(
        tf.config, "list_physical_devices", lambda kind: ["GPU:0"] if kind == "GPU" else []
    )
    with pytest.raises(RuntimeError, match="GPU visível"):
        cpu_policy.verify("fast")


def test_verify_passes_cpu_only():
    pytest.importorskip("tensorflow")
    report = cpu_policy.verify("fast")
    assert report["physical_gpus"] == []
    assert report["cuda_disabled"] is True or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
