"""Quality Gate QG0 — Project-Lewis Camada 1.

Verifies that the download + integrity layer produced a valid artifact set:

* QG0.1 — record counts per dataset.
* QG0.2 — per-record file integrity (.hea, .dat, .atr / .qrs).
* QG0.3 — ZIP checksums vs ``src/data/checksums.json`` (when ZIPs are cached).
* QG0.4 — ``data/catalog/dataset_catalog.jsonl`` completeness.
* QG0.5 — ``data/.dlq/failed_downloads.jsonl`` is empty/absent.

Most tests skip when ``data/raw_*`` is not populated. To enforce data
presence (CI, pre-commit), set the env var ``LEWIS_REQUIRE_DATA=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.data.download_mitbih import (
    AFDB_ANNOTATIONS_ONLY,
    DATASET_CONFIG,
    RECORDS_AFDB,
    RECORDS_AFDB_PRIMARY,
    RECORDS_INCART,
    RECORDS_MITBIH,
    RECORDS_SVDB,
)

DATA = Path("data")


@pytest.mark.qg0
class TestRecordLists:
    """Synthetic checks for Camada 1 record list correctness."""

    @pytest.mark.qg0
    def test_mitbih_record_count(self) -> None:
        assert len(RECORDS_MITBIH) == 48
        assert DATASET_CONFIG["mitdb"]["expected_count"] == 48

    @pytest.mark.qg0
    def test_svdb_record_count_is_78(self) -> None:
        # Authoritative PhysioNet svdb IDs: 800-812, 820-829, 840-894 (78 records).
        assert len(RECORDS_SVDB) == 78
        assert RECORDS_SVDB[0] == "800"
        assert RECORDS_SVDB[12] == "812"
        assert RECORDS_SVDB[13] == "820"
        assert RECORDS_SVDB[22] == "829"
        assert RECORDS_SVDB[23] == "840"
        assert RECORDS_SVDB[-1] == "894"
        assert DATASET_CONFIG["svdb"]["expected_count"] == 78

    @pytest.mark.qg0
    def test_afdb_record_count_is_25(self) -> None:
        assert len(RECORDS_AFDB_PRIMARY) == 23
        assert len(AFDB_ANNOTATIONS_ONLY) == 2
        assert len(RECORDS_AFDB) == 25
        assert DATASET_CONFIG["afdb"]["expected_count"] == 25
        assert AFDB_ANNOTATIONS_ONLY.issubset(set(RECORDS_AFDB))

    @pytest.mark.qg0
    def test_incart_record_count(self) -> None:
        assert len(RECORDS_INCART) == 75
        assert DATASET_CONFIG["incartdb"]["expected_count"] == 75


RAW = {
    "chapman": DATA / "raw_chapman",
    "mitdb": DATA / "raw_mitbih",
    "svdb": DATA / "raw_svdb",
    "afdb": DATA / "raw_afdb",
    "incart": DATA / "raw_incart",
}
CHECKSUMS = Path("src/data/checksums.json")
CATALOG = DATA / "catalog" / "dataset_catalog.jsonl"
DLQ = DATA / ".dlq" / "failed_downloads.jsonl"


def _has_hea(path: Path) -> bool:
    """True if ``path`` contains at least one ``.hea`` file (anywhere)."""
    if not path.exists():
        return False
    try:
        return any(True for _ in path.rglob("*.hea"))
    except OSError:
        return False


def _has_csv(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return any(True for _ in path.rglob("*.csv"))
    except OSError:
        return False


def _require_or_skip(path: Path, what: str) -> None:
    if path.is_file() and path.exists() and path.stat().st_size > 0:
        return
    if _has_hea(path) or _has_csv(path):
        return
    if os.environ.get("LEWIS_REQUIRE_DATA") == "1":
        pytest.fail(f"{what} missing at {path} — run `make download-all`")
    pytest.skip(f"{what} missing or empty at {path} — run `make download-all`")


@pytest.mark.qg0
def test_qg0_1a_chapman_min_count() -> None:
    raw = RAW["chapman"]
    if not (_has_hea(raw) or _has_csv(raw)):
        if os.environ.get("LEWIS_REQUIRE_DATA") == "1":
            pytest.fail("raw_chapman/ empty — run `make download-chapman`")
        pytest.skip("raw_chapman/ empty — run `make download-chapman`")
    n_hea = sum(1 for _ in raw.rglob("*.hea"))
    n_csv = sum(1 for _ in raw.rglob("*.csv"))
    total = n_hea + n_csv
    # PhysioNet Challenge 2021 bundles Chapman-Shaoxing (10,247) + Ningbo
    # (34,905) for the canonical 45,152 records.
    assert total >= 45_000, (
        f"Chapman has {total} records, expected >= 45.000 "
        "(PhysioNet Challenge 2021: chapman_shaoxing + ningbo)"
    )


@pytest.mark.parametrize(
    "ds,expected",
    [
        ("mitdb", 48),
        ("svdb", 78),
        ("afdb", 25),
        ("incart", 75),
    ],
)
@pytest.mark.qg0
def test_qg0_1_mitbih_family_count(ds: str, expected: int) -> None:
    raw = RAW[ds]
    _require_or_skip(raw, f"raw_{ds}/")
    n = sum(1 for _ in raw.glob("*.hea"))
    assert n == expected, f"{ds} has {n} .hea, expected {expected}"


@pytest.mark.parametrize("ds", ["mitdb", "svdb", "afdb", "incart"])
@pytest.mark.qg0
def test_qg0_2_integridade_arquivos(ds: str) -> None:
    raw = RAW[ds]
    _require_or_skip(raw, f"raw_{ds}/")
    missing_dat: list[str] = []
    missing_hea: list[str] = []
    missing_atr: list[str] = []
    for hea in raw.glob("*.hea"):
        rec = hea.stem
        if not hea.exists():
            missing_hea.append(rec)
        if not (raw / f"{rec}.dat").exists() and rec not in AFDB_ANNOTATIONS_ONLY:
            missing_dat.append(rec)
        if not (raw / f"{rec}.atr").exists() and ds != "afdb":
            missing_atr.append(rec)
    assert not missing_hea, f"{ds}: missing .hea: {missing_hea}"
    assert not missing_dat, f"{ds}: missing .dat: {missing_dat}"
    assert not missing_atr, f"{ds}: missing .atr: {missing_atr}"


@pytest.mark.qg0
def test_qg0_3_checksums() -> None:
    from src.data._compliance import verify_all

    cache = Path("data/.cache/zips")
    if not cache.exists() or not any(cache.glob("*.zip")):
        if os.environ.get("LEWIS_REQUIRE_DATA") == "1":
            pytest.fail("no cached ZIPs under data/.cache/zips — run `make download-all`")
        pytest.skip("no cached ZIPs under data/.cache/zips — run `make download-all`")
    res = verify_all()
    if not res:
        _require_or_skip(CHECKSUMS, "checksums.json")
    failed = [k for k, v in res.items() if not v]
    assert not failed, f"checksum failed for: {failed}"


@pytest.mark.qg0
def test_qg0_4_catalogo_completo() -> None:
    _require_or_skip(CATALOG, "data/catalog/dataset_catalog.jsonl")
    lines = [ln for ln in CATALOG.read_text(encoding="utf-8").splitlines() if ln.strip()]
    expected_min = sum(
        sum(1 for _ in RAW[ds].glob("*.hea"))
        for ds in ("mitdb", "svdb", "afdb", "incart")
        if RAW[ds].exists()
    )
    assert len(lines) >= expected_min, f"catalog has {len(lines)} lines, expected >= {expected_min}"


@pytest.mark.qg0
def test_qg0_5_dlq_vazia() -> None:
    if not DLQ.exists():
        return
    content = DLQ.read_text(encoding="utf-8").strip()
    assert not content, f"DLQ non-empty: {content[:500]}"


@pytest.mark.qg0
def test_catalogo_linhas_validas() -> None:
    _require_or_skip(CATALOG, "data/catalog/dataset_catalog.jsonl")
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
    for i, line in enumerate(CATALOG.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        obj = json.loads(line)
        missing = required - obj.keys()
        assert not missing, f"line {i} missing {missing}: {line[:200]}"


@pytest.mark.qg0
def test_checksums_manifest_schema() -> None:
    _require_or_skip(CHECKSUMS, "src/data/checksums.json")
    from src.data._schemas import validate_checksums_manifest

    assert validate_checksums_manifest(json.loads(CHECKSUMS.read_text(encoding="utf-8")))


@pytest.mark.parametrize("ds", ["mitdb", "svdb", "afdb", "incart"])
@pytest.mark.qg0
def test_wfdb_header_parseable(ds: str) -> None:
    raw = RAW[ds]
    _require_or_skip(raw, f"raw_{ds}/")
    pytest.importorskip("wfdb")
    from src.data._catalog import extract_metadata

    hea_files = sorted(
        h for h in raw.glob("*.hea")
        if h.stem not in AFDB_ANNOTATIONS_ONLY
    )
    assert hea_files, f"no .hea files under {raw}"
    sample = hea_files[0]
    meta = extract_metadata(sample, dataset=ds)
    assert meta.record_name
    assert meta.fs > 0
    assert meta.n_sig > 0
    assert meta.sig_len > 0
