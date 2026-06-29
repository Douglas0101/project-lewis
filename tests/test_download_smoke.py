"""Smoke tests for the ingestion/catalog pipeline.

These tests do **not** download the full Chapman/Shaoxing or MIT-BIH datasets.
Instead they create tiny synthetic WFDB records under temporary directories and
verify that the catalog builder and QG0-style integrity checks work end-to-end.

Marker: ``smoke`` (fast, no network).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.data._catalog as catalog_module
from src.data._catalog import RecordMetadata, build_catalog, extract_metadata


@pytest.fixture
def smoke_record_factory():
    """Return a helper that writes a minimal WFDB header (and empty .dat)."""

    def _make(
        root: Path,
        record_name: str,
        *,
        n_sig: int = 1,
        sig_len: int = 1000,
        fs: int = 360,
        signal_names: list[str] | None = None,
        comments: list[str] | None = None,
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        signal_names = signal_names or [f"sig{i}" for i in range(n_sig)]
        header_lines = [
            f"{record_name} {n_sig} {fs} {sig_len} 00:00:00",
        ]
        for name in signal_names:
            header_lines.append(
                f"{name} 16 1000 0 mV"
            )
        for comment in comments or []:
            header_lines.append(f"# {comment}")
        hea_path = root / f"{record_name}.hea"
        hea_path.write_text("\n".join(header_lines) + "\n", encoding="utf-8")
        (root / f"{record_name}.dat").write_bytes(b"")
        return hea_path

    return _make


@pytest.fixture
def smoke_catalog(tmp_path, monkeypatch, smoke_record_factory):
    """Create a tiny multi-dataset tree and patch the catalog globals."""
    raw_chapman = tmp_path / "raw_chapman" / "chapman_shaoxing" / "g1"
    raw_mitbih = tmp_path / "raw_mitbih"
    raw_svdb = tmp_path / "raw_svdb"
    raw_afdb = tmp_path / "raw_afdb"
    raw_incart = tmp_path / "raw_incart"

    smoke_record_factory(
        raw_chapman,
        "SMOKE_C01",
        n_sig=2,
        sig_len=5000,
        fs=500,
        signal_names=["I", "II"],
        comments=["age 55", "sex M", "dx normal"],
    )
    smoke_record_factory(
        raw_mitbih,
        "SMOKE_M01",
        n_sig=1,
        sig_len=2160,
        fs=360,
        signal_names=["MLII"],
        comments=["age 60", "sex F", "dx N"],
    )
    # SVDB / AFDB / INCART intentionally empty in smoke; the catalog should skip
    # them gracefully.

    patched_raw = {
        "chapman": raw_chapman.parent.parent,  # points to raw_chapman
        "mitdb": raw_mitbih,
        "svdb": raw_svdb,
        "afdb": raw_afdb,
        "incart": raw_incart,
        "ptbxl": tmp_path / "raw_ptbxl",
    }
    monkeypatch.setattr(catalog_module, "RAW_DATASETS", patched_raw)

    catalog_path = tmp_path / "smoke_catalog.jsonl"
    return {
        "catalog_path": catalog_path,
        "raw_chapman": raw_chapman,
        "raw_mitbih": raw_mitbih,
    }


@pytest.mark.smoke
@pytest.mark.qg0
def test_smoke_catalog_build(smoke_catalog):
    counts = build_catalog(catalog_path=smoke_catalog["catalog_path"], overwrite=True)
    assert counts["ok"] == 2, f"expected 2 records, got {counts}"
    assert counts["fail"] == 0

    lines = [
        ln
        for ln in smoke_catalog["catalog_path"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(lines) == 2

    required = {
        "record_name",
        "dataset",
        "fs",
        "n_sig",
        "sig_len",
        "duration_sec",
        "units",
        "gains",
        "sig_name",
        "source_path",
    }
    for i, line in enumerate(lines):
        obj = json.loads(line)
        missing = required - obj.keys()
        assert not missing, f"line {i} missing {missing}"
        assert obj["fs"] > 0
        assert obj["n_sig"] > 0
        assert obj["sig_len"] > 0


@pytest.mark.smoke
@pytest.mark.qg0
def test_smoke_extract_metadata(smoke_catalog):
    hea = smoke_catalog["raw_chapman"] / "SMOKE_C01.hea"
    meta: RecordMetadata = extract_metadata(hea, dataset="chapman")
    assert meta.record_name == "SMOKE_C01"
    assert meta.fs == 500
    assert meta.n_sig == 2
    assert meta.sig_len == 5000
    assert meta.duration_sec == 10.0
    assert meta.age == 55
    assert meta.sex == "M"
    assert meta.diagnosis == "normal"


@pytest.mark.smoke
@pytest.mark.qg0
def test_smoke_integrity_files(smoke_catalog):
    """QG0-style integrity check on the tiny synthetic tree."""
    raw_mitbih = smoke_catalog["raw_mitbih"]
    missing = []
    for hea in raw_mitbih.glob("*.hea"):
        rec = hea.stem
        if not (raw_mitbih / f"{rec}.dat").exists():
            missing.append(f"{rec}.dat")
    assert not missing, f"missing data files: {missing}"
