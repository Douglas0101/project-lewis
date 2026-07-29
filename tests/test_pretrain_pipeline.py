"""Tests for the pretrain wrapper + artifact validator (FASE 3/4, SDD policy).

Exit policy (SDD section 11): default mode returns 0 on execution success
(QG4 pass/fail is a scientific result); ``--enforce-qg4`` returns 10 when
execution succeeded but QG4 failed; real failures (Traceback, crash,
missing artifacts) are never masked.
"""

from __future__ import annotations

import json

from scripts.pretrain_wrapper import (
    QG4_FAIL_EXIT,
    SUCCESS_MARKER,
    TEARDOWN_ERROR,
    decide_exit_code,
)
from scripts.validate_pretrain_artifacts import validate_run_dir

LOG_OK_QG4_PASS = (
    "2026-07-29 | INFO | Pré-treino concluído | loss=0.3 | val_loss=0.1\n"
    "2026-07-29 | INFO | QG4 | best_epoch=5 | val_auc_roc=0.90 | val_loss=0.10 | pass=True\n"
)
LOG_OK_QG4_FAIL = (
    "2026-07-29 | INFO | Pré-treino concluído | loss=0.5 | val_loss=0.5\n"
    "2026-07-29 | INFO | QG4 | best_epoch=3 | val_auc_roc=0.80 | val_loss=0.50 | pass=False\n"
)
LOG_TEARDOWN = (
    LOG_OK_QG4_PASS
    + "Error occurred when finalizing GeneratorDataset iterator: "
    + "FAILED_PRECONDITION: Python interpreter state is not initialized.\n"
)
LOG_CRASH = "Traceback (most recent call last):\nRuntimeError: boom\n"
LOG_SUCCESS_THEN_TRACEBACK = LOG_OK_QG4_PASS + LOG_CRASH


def _decide(rc, log, ok, smoke=False, enforce=False):
    return decide_exit_code(
        returncode=rc,
        log_text=log,
        artifacts_ok=ok,
        smoke=smoke,
        enforce_qg4=enforce,
    )


# ---------------------------------------------------------------------------
# Default mode: execution semantics (QG4 nunca bloqueia)
# ---------------------------------------------------------------------------


def test_default_mode_qg4_pass_returns_zero():
    assert _decide(0, LOG_OK_QG4_PASS, ok=True) == 0


def test_default_mode_qg4_fail_also_returns_zero():
    """QG4 fail é resultado científico, não falha de processo (DEF-008)."""
    assert _decide(0, LOG_OK_QG4_FAIL, ok=True) == 0


def test_teardown_error_with_real_success_returns_zero():
    assert _decide(1, LOG_TEARDOWN, ok=True) == 0


def test_real_crash_keeps_nonzero():
    assert _decide(2, LOG_CRASH, ok=False) == 2
    assert _decide(1, LOG_CRASH, ok=False) == 1


def test_traceback_after_success_is_real_failure():
    """RISK-001: Traceback nunca é mascarado, mesmo após 'concluído'."""
    assert _decide(1, LOG_SUCCESS_THEN_TRACEBACK, ok=True) == 1


def test_success_with_missing_artifacts_fails():
    assert _decide(0, LOG_OK_QG4_PASS, ok=False) == 1


def test_config_error_returns_two():
    assert _decide(2, "usage: pretrain_chapman [-h]\n", ok=False) == 2


# ---------------------------------------------------------------------------
# Gate enforcement (--enforce-qg4): 0 pass / 10 fail
# ---------------------------------------------------------------------------


def test_enforce_qg4_pass_returns_zero():
    assert _decide(0, LOG_OK_QG4_PASS, ok=True, enforce=True) == 0


def test_enforce_qg4_fail_returns_ten():
    assert _decide(0, LOG_OK_QG4_FAIL, ok=True, enforce=True) == QG4_FAIL_EXIT
    assert QG4_FAIL_EXIT == 10


def test_enforce_real_failure_not_masked():
    assert _decide(1, LOG_CRASH, ok=False, enforce=True) == 1


# ---------------------------------------------------------------------------
# Smoke mode: engineering only
# ---------------------------------------------------------------------------


def test_smoke_mode_checks_engineering_only():
    assert _decide(1, LOG_OK_QG4_FAIL, ok=True, smoke=True) == 0


def test_smoke_mode_still_fails_on_crash():
    assert _decide(2, LOG_CRASH, ok=False, smoke=True) == 2


# ---------------------------------------------------------------------------
# validate_run_dir
# ---------------------------------------------------------------------------


def _make_run_dir(tmp_path, *, with_model=True, bad_config=False, strict_files=False):
    run = tmp_path / "run"
    run.mkdir()
    if with_model:
        (run / "backbone_pretrained.keras").write_bytes(b"\x00" * 2048)
    config = {"name": "m", "input_shape": [None, 500, 1], "total_params": 19933}
    (run / "config.json").write_text(
        json.dumps(config) if not bad_config else "{not json", encoding="utf-8"
    )
    (run / "metrics.json").write_text(
        json.dumps({"final_val_loss": 0.5}), encoding="utf-8"
    )
    if strict_files:
        for name in ("provenance.json", "history.json", "metrics_per_class.json",
                     "run_status.json", "qg4_result.json"):
            (run / name).write_text("{}", encoding="utf-8")
    return run


def test_validate_run_dir_ok(tmp_path):
    assert validate_run_dir(_make_run_dir(tmp_path)) == []


def test_validate_run_dir_strict_ok(tmp_path):
    assert validate_run_dir(_make_run_dir(tmp_path, strict_files=True), strict=True) == []


def test_validate_run_dir_strict_missing_status_files(tmp_path):
    problems = validate_run_dir(_make_run_dir(tmp_path), strict=True)
    assert any("run_status.json" in p for p in problems)
    assert any("qg4_result.json" in p for p in problems)


def test_validate_run_dir_missing_model(tmp_path):
    problems = validate_run_dir(_make_run_dir(tmp_path, with_model=False))
    assert any("backbone_pretrained.keras" in p for p in problems)


def test_validate_run_dir_bad_config(tmp_path):
    problems = validate_run_dir(_make_run_dir(tmp_path, bad_config=True))
    assert any("config.json" in p for p in problems)


def test_validate_run_dir_missing_dir(tmp_path):
    problems = validate_run_dir(tmp_path / "nope")
    assert problems


def test_constants_exposed():
    assert "concluído" in SUCCESS_MARKER
    assert "not initialized" in TEARDOWN_ERROR


# ---------------------------------------------------------------------------
# build_subprocess_env (FASE 5 — determinismo antes do TensorFlow)
# ---------------------------------------------------------------------------


def test_strict_env_disables_onednn_before_tf_import():
    from scripts.pretrain_wrapper import build_subprocess_env

    env = build_subprocess_env({"deterministic": {"mode": "strict"}}, seed=13)
    assert env["TF_ENABLE_ONEDNN_OPTS"] == "0"
    assert env["TF_DETERMINISTIC_OPS"] == "1"
    assert env["PYTHONHASHSEED"] == "13"


def test_fast_env_keeps_onednn_and_pins_hashseed():
    import os

    from scripts.pretrain_wrapper import build_subprocess_env

    env = build_subprocess_env({"deterministic": {"mode": "fast"}}, seed=None)
    assert env.get("TF_ENABLE_ONEDNN_OPTS") == os.environ.get("TF_ENABLE_ONEDNN_OPTS")
    assert env["PYTHONHASHSEED"] == "42"


def test_strict_env_uses_config_seed_when_no_override():
    from scripts.pretrain_wrapper import build_subprocess_env

    env = build_subprocess_env(
        {"deterministic": {"mode": "strict"}, "training": {"seed": 99}}, seed=None
    )
    assert env["PYTHONHASHSEED"] == "99"
