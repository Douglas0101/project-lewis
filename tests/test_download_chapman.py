"""Tests for the idempotent Chapman-Shaoxing downloader (Camada 1).

Covers the smart-skip gate (data already present → no network), the
``--force`` bypass, the offline ``--verify`` integrity check, and the
incremental wget source. All tests are offline: every network source is
monkeypatched.
"""

from __future__ import annotations

import sys
import types

import pytest

import src.data.download_chapman as dc
from src.data._downloader import SourceExhausted


@pytest.fixture()
def audit_spy(monkeypatch):
    """Capture audit events instead of writing to data/audit/ingestion.jsonl."""
    events: list[dict] = []
    monkeypatch.setattr(dc, "append_audit", events.append)
    return events


@pytest.fixture()
def populated_raw(tmp_path, monkeypatch):
    """raw_dir with enough synthetic .hea/.mat pairs to satisfy the gate."""
    monkeypatch.setattr(dc, "EXPECTED_MIN_RECORDS", 3)
    raw = tmp_path / "raw_chapman"
    raw.mkdir()
    for i in range(3):
        (raw / f"rec{i}.hea").write_text(
            f"rec{i} 1 500 1000\nrec{i}.mat 16 1000.0(0)/mV 16 0 0 0 0 I\n",
            encoding="utf-8",
        )
        (raw / f"rec{i}.mat").write_bytes(b"")
    return raw


def _forbid_sources(monkeypatch):
    """Make every download source raise if it is called."""

    def _boom(*_args, **_kwargs):
        raise AssertionError("network source must not be called")

    for name in ("_try_mirror", "_try_wget_recursive", "_try_physionet_zip", "_try_kagglehub"):
        monkeypatch.setattr(dc, name, _boom)


def test_gate_skips_download_when_records_present(populated_raw, monkeypatch, audit_spy):
    _forbid_sources(monkeypatch)
    n = dc.download_chapman(raw_dir=populated_raw)
    assert n == 3
    assert any(e.get("event") == "chapman_skip_present" for e in audit_spy)


def test_force_bypasses_gate(populated_raw, monkeypatch, audit_spy):
    called = {"mirror": False}

    def _fake_mirror(raw_dir, mirror):
        called["mirror"] = True
        return True

    monkeypatch.setattr(dc, "_try_mirror", _fake_mirror)
    for name in ("_try_wget_recursive", "_try_physionet_zip", "_try_kagglehub"):

        def _boom(*_args, **_kwargs):
            raise AssertionError("must stop after mirror success")

        monkeypatch.setattr(dc, name, _boom)

    mirror = populated_raw.parent / "chapman_mirror.tar.gz"
    mirror.write_bytes(b"")
    n = dc.download_chapman(raw_dir=populated_raw, mirror_path=mirror, force=True)
    assert called["mirror"], "force=True must run the source cascade"
    assert n == 3
    assert not any(e.get("event") == "chapman_skip_present" for e in audit_spy)


def test_cascade_raises_when_all_sources_fail(tmp_path, monkeypatch, audit_spy):
    monkeypatch.setattr(dc, "EXPECTED_MIN_RECORDS", 3)
    empty = tmp_path / "raw_chapman"
    monkeypatch.setattr(dc, "_try_mirror", lambda *_a: False)
    monkeypatch.setattr(dc, "_try_wget_recursive", lambda *_a: False)
    monkeypatch.setattr(dc, "_try_physionet_zip", lambda *_a: False)
    monkeypatch.setattr(dc, "_try_kagglehub", lambda *_a, **_kw: False)
    with pytest.raises(SourceExhausted, match="all 4 sources failed"):
        dc.download_chapman(raw_dir=empty)


def test_kagglehub_skipped_when_dir_not_empty(tmp_path):
    raw = tmp_path / "raw_chapman"
    raw.mkdir()
    (raw / "anything.hea").write_text("x", encoding="utf-8")
    assert dc._try_kagglehub(raw, force=False) is False


def test_kagglehub_force_download(tmp_path, monkeypatch):
    calls: list[bool] = []
    fake = types.SimpleNamespace(
        dataset_download=lambda _slug, output_dir, force_download=False: calls.append(
            force_download
        )
    )
    monkeypatch.setitem(sys.modules, "kagglehub", fake)
    raw = tmp_path / "raw_chapman"
    raw.mkdir()
    (raw / "stale.hea").write_text("x", encoding="utf-8")
    assert dc._try_kagglehub(raw, force=True) is True
    assert calls == [True]


def test_wget_skips_complete_subsets_and_uses_incremental_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "WGET_SUBSET_MIN_RECORDS", {"chapman_shaoxing": 2, "ningbo": 2})
    monkeypatch.setattr(dc.shutil, "which", lambda _cmd: "/usr/bin/wget")
    cmds: list[list[str]] = []
    monkeypatch.setattr(dc.subprocess, "run", lambda cmd, **kw: cmds.append(cmd))

    raw = tmp_path / "raw_chapman"
    done = raw / "chapman_shaoxing"
    done.mkdir(parents=True)
    for i in range(2):
        (done / f"r{i}.hea").write_text("x", encoding="utf-8")

    assert dc._try_wget_recursive(raw) is True
    assert len(cmds) == 1, "complete subset must be skipped"
    assert "ningbo" in cmds[0][-1]
    assert "-c" in cmds[0] and "-N" in cmds[0], "wget must resume/skip existing files"


def test_verify_ok_with_complete_pairs(populated_raw, monkeypatch):
    monkeypatch.setattr(dc, "_parse_header_sample", lambda files, sample_size: [])
    n, problems = dc.verify_chapman(populated_raw)
    assert n == 3
    assert problems == []


def test_verify_flags_missing_mat_and_low_count(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "EXPECTED_MIN_RECORDS", 5)
    monkeypatch.setattr(dc, "_parse_header_sample", lambda files, sample_size: [])
    raw = tmp_path / "raw_chapman"
    raw.mkdir()
    (raw / "a.hea").write_text(
        "a 1 500 1000\na.mat 16 1000.0(0)/mV 16 0 0 0 0 I\n", encoding="utf-8"
    )
    (raw / "a.mat").write_bytes(b"")
    (raw / "orphan.hea").write_text(
        "orphan 1 500 1000\norphan.mat 16 1000.0(0)/mV 16 0 0 0 0 I\n", encoding="utf-8"
    )

    n, problems = dc.verify_chapman(raw)
    assert n == 2
    assert any("orphan" in p for p in problems)
    assert any("45" in p or "registros" in p or ">=" in p for p in problems)


def test_verify_accepts_declared_signal_name(tmp_path, monkeypatch):
    """WFDB: the .hea declares its signal file (S23074.hea → JS23074.mat)."""
    monkeypatch.setattr(dc, "EXPECTED_MIN_RECORDS", 1)
    monkeypatch.setattr(dc, "_parse_header_sample", lambda files, sample_size: [])
    raw = tmp_path / "raw_chapman"
    raw.mkdir()
    (raw / "S23074.hea").write_text(
        "S23074 1 500 1000\nJS23074.mat 16 1000.0(0)/mV 16 0 0 0 0 I\n",
        encoding="utf-8",
    )
    (raw / "JS23074.mat").write_bytes(b"")

    n, problems = dc.verify_chapman(raw)
    assert n == 1
    assert problems == []


def test_parse_header_sample_records_invalid_headers(tmp_path):
    pytest.importorskip("wfdb")
    bad = tmp_path / "bad.hea"
    bad.write_bytes(b"")
    problems = dc._parse_header_sample([bad], sample_size=50)
    assert problems, "empty header must be reported"


def test_main_verify_exit_codes(populated_raw, monkeypatch, capsys):
    monkeypatch.setattr(dc, "_parse_header_sample", lambda files, sample_size: [])
    monkeypatch.setattr(dc, "_chapman_raw_dir", lambda: populated_raw)
    monkeypatch.setattr(sys, "argv", ["download_chapman", "--verify"])
    assert dc.main() == 0
