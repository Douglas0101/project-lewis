"""MIT-BIH family downloader (Camada 1 CLI).

Covers the four beat/rhythm datasets: mitdb, svdb, afdb, incartdb.
Per-dataset resolution order: PhysioNet ZIP → wfdb-python dl_database → mirror.

Usage:
    python -m src.data.download_mitbih
    python -m src.data.download_mitbih --datasets mitdb,svdb
"""

from __future__ import annotations

import logging
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

from ._downloader import (
    CACHE_ZIPS_DIR,
    LOGGER,
    DownloadError,
    SourceExhausted,
    _now_iso,
    append_audit,
    circuit_breaker,
    download_url,
    project_root,
)

RECORDS_MITBIH: list[str] = [
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
    "108",
    "109",
    "111",
    "112",
    "113",
    "114",
    "115",
    "116",
    "117",
    "118",
    "119",
    "121",
    "122",
    "123",
    "124",
    "200",
    "201",
    "202",
    "203",
    "205",
    "207",
    "208",
    "209",
    "210",
    "212",
    "213",
    "214",
    "215",
    "217",
    "219",
    "220",
    "221",
    "222",
    "223",
    "228",
    "230",
    "231",
    "232",
    "233",
    "234",
]
RECORDS_SVDB: list[str] = [
    f"{i:03d}"
    for i in (
        list(range(800, 813))  # 800-812  (13)
        + list(range(820, 830))  # 820-829  (10)
        + list(range(840, 895))  # 840-894  (55)
    )
]
"""SVDB records: authoritative list from PhysioNet svdb/1.0.0/RECORDS (78).

The MIT-BIH Supraventricular Arrhythmia Database does not use a contiguous
range. The canonical IDs are 800-812, 820-829 and 840-894, totaling 78
records. Using a naive range(800, 879) would include non-existent IDs and
miss real ones, so we keep the explicit enumeration aligned with the
Camada-01 spec and DATASET_CONFIG["svdb"]["expected_count"].
"""
RECORDS_AFDB_PRIMARY: list[str] = [
    "04015",
    "04043",
    "04048",
    "04126",
    "04746",
    "04908",
    "04936",
    "05091",
    "05121",
    "05261",
    "06426",
    "06453",
    "06995",
    "07162",
    "07859",
    "07879",
    "07910",
    "08215",
    "08219",
    "08378",
    "08405",
    "08434",
    "08455",
]
AFDB_ANNOTATIONS_ONLY: set[str] = {"00735", "03665"}
RECORDS_AFDB: list[str] = RECORDS_AFDB_PRIMARY + sorted(AFDB_ANNOTATIONS_ONLY)
"""AFDB records: 23 signal records + 2 annotations-only records.

The two annotations-only records (00735, 03665) have no .dat file. They are
still part of the canonical 25-record AFDB set, so we include them in the
download list and handle their missing .dat gracefully in the wfdb
fallback.
"""
RECORDS_INCART: list[str] = [f"I{i:02d}" for i in range(1, 76)]

_PHYSIONET = "https://physionet.org/static/published-projects"
MITDB_ZIP = f"{_PHYSIONET}/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"
SVDB_ZIP = f"{_PHYSIONET}/svdb/mit-bih-supraventricular-arrhythmia-database-1.0.0.zip"
AFDB_ZIP = f"{_PHYSIONET}/afdb/mit-bih-atrial-fibrillation-database-1.0.0.zip"
INCART_ZIP = f"{_PHYSIONET}/incartdb/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0.zip"

DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "mitdb": {
        "records": RECORDS_MITBIH,
        "raw_subdir": Path("data/raw_mitbih"),
        "zip_url": MITDB_ZIP,
        "expected_count": 48,
    },
    "svdb": {
        "records": RECORDS_SVDB,
        "raw_subdir": Path("data/raw_svdb"),
        "zip_url": SVDB_ZIP,
        "expected_count": 78,
    },
    "afdb": {
        "records": RECORDS_AFDB,
        "raw_subdir": Path("data/raw_afdb"),
        "zip_url": AFDB_ZIP,
        "expected_count": 25,
    },
    "incartdb": {
        "records": RECORDS_INCART,
        "raw_subdir": Path("data/raw_incart"),
        "zip_url": INCART_ZIP,
        "expected_count": 75,
    },
}

MITBIH_FAMILY_MIRROR = Path("data/mirrors/mitbih_family_mirror.tar.gz")


def _try_wfdb(
    name: str,
    records: list[str],
    raw_dir: Path,
    annotations_only: Optional[set[str]] = None,
) -> bool:
    try:
        import wfdb  # type: ignore
    except ImportError:
        LOGGER.error("wfdb is not installed; cannot use the wfdb source")
        return False
    raw_dir.mkdir(parents=True, exist_ok=True)
    annotations_only = annotations_only or set()
    ok = 0
    for rec in records:
        try:
            wfdb.dl_database(name, [rec], str(raw_dir))
            ok += 1
        except Exception as exc:
            # Annotations-only records have no .dat; try to fetch .hea + .atr.
            if rec in annotations_only:
                try:
                    wfdb.dl_files(name, str(raw_dir), [f"{rec}.hea", f"{rec}.atr"])
                    ok += 1
                    continue
                except Exception as inner_exc:
                    LOGGER.warning(
                        "wfdb annotations-only fallback failed for %s/%s: %s",
                        name,
                        rec,
                        inner_exc,
                    )
            LOGGER.warning("wfdb failed for %s/%s: %s", name, rec, exc)
    return ok == len(records)


def _canonical_files(raw_dir: Path, records: set[str]) -> None:
    """Keep only files belonging to ``records`` and remove nested sub-trees.

    PhysioNet ZIPs may extract documentation folders (``x_mitdb/``,
    ``mitdbdir/``) or old header backups (``.old-headers/``).  This helper
    walks the extracted tree, moves every canonical record file to
    ``raw_dir``, and deletes everything else.
    """
    extensions = {".hea", ".dat", ".atr", ".qrs", ".xws"}
    for path in raw_dir.rglob("*"):
        if path == raw_dir:
            continue
        if path.is_file() and path.suffix.lower() in extensions and path.stem in records:
            dest = raw_dir / path.name
            if dest != path:
                if dest.exists():
                    dest.unlink()
                path.rename(dest)
    # Remove any remaining directories (including hidden ones).
    for path in sorted(raw_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)


def _try_zip(
    zip_url: str,
    raw_dir: Path,
    zip_cache: Path,
    records: Optional[set[str]] = None,
) -> bool:
    if not zip_cache.exists():
        try:
            download_url(zip_url, zip_cache, source="primary")
        except DownloadError:
            return False
    try:
        with zipfile.ZipFile(zip_cache) as zf:
            zf.extractall(raw_dir)  # nosec B202
        if records is not None:
            _canonical_files(raw_dir, records)
        return True
    except zipfile.BadZipFile as exc:
        LOGGER.error("invalid ZIP %s: %s", zip_cache.name, exc)
        zip_cache.unlink(missing_ok=True)
        return False


def _try_mirror(raw_dir: Path, mirror: Path) -> bool:
    if not mirror.exists():
        return False
    try:
        with tarfile.open(mirror) as tf:
            tf.extractall(raw_dir)  # nosec B202
        return True
    except (tarfile.TarError, OSError) as exc:
        LOGGER.error("mirror restore failed: %s", exc)
        return False


def _count_hea(raw_dir: Path) -> int:
    return sum(1 for _ in raw_dir.glob("*.hea"))


@circuit_breaker(threshold=8, cooldown_sec=120.0)
def download_one_dataset(name: str, cfg: dict[str, Any], mirror: Optional[Path] = None) -> int:
    """Download a single dataset using its source cascade.

    Returns the detected record count. Raises :class:`SourceExhausted` if all
    sources fail or return the wrong record count.
    """
    raw_dir: Path = project_root() / cfg["raw_subdir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_cache: Path = project_root() / CACHE_ZIPS_DIR / f"{name}.zip"

    afdb_annotations_only = AFDB_ANNOTATIONS_ONLY if name == "afdb" else None
    records_set = set(cfg["records"])
    sources: list[tuple[str, Callable[[], bool]]] = [
        ("zip", lambda: _try_zip(cfg["zip_url"], raw_dir, zip_cache, records_set)),
        (
            "wfdb",
            lambda: _try_wfdb(
                name, cfg["records"], raw_dir, annotations_only=afdb_annotations_only
            ),
        ),
        ("mirror", lambda: _try_mirror(raw_dir, mirror) if mirror and mirror.exists() else False),
    ]
    for source, fn in sources:
        if not fn():
            continue
        n = _count_hea(raw_dir)
        append_audit(
            {
                "event": f"{name}_done",
                "ts": _now_iso(),
                "source": source,
                "records": n,
            }
        )
        if n == cfg["expected_count"]:
            LOGGER.info("%s OK via %s: %d records", name, source, n)
            return n
        LOGGER.warning(
            "%s via %s returned %d (expected %d)",
            name,
            source,
            n,
            cfg["expected_count"],
        )
    raise SourceExhausted(f"{name}: all 3 sources failed or returned wrong count")


def download_mitbih_family(mirror_path: Optional[Path] = None) -> dict[str, int]:
    """Download all four MIT-BIH-family datasets sequentially.

    Returns ``{name: record_count}`` for each dataset that succeeded.
    """
    mirror = (
        (project_root() / mirror_path) if mirror_path else (project_root() / MITBIH_FAMILY_MIRROR)
    )
    results: dict[str, int] = {}
    for name, cfg in DATASET_CONFIG.items():
        try:
            results[name] = download_one_dataset(name, cfg, mirror=mirror)
        except SourceExhausted as exc:
            LOGGER.error("%s", exc)
            results[name] = -1
    return results


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if argv is None:
        argv = sys.argv[1:]
    selected: Optional[list[str]] = None
    for i, arg in enumerate(argv):
        if arg == "--datasets" and i + 1 < len(argv):
            selected = [d.strip() for d in argv[i + 1].split(",") if d.strip()]
    if selected is not None:
        unknown = [d for d in selected if d not in DATASET_CONFIG]
        if unknown:
            LOGGER.error("unknown datasets: %s (valid: %s)", unknown, list(DATASET_CONFIG))
            return 2
        results: dict[str, int] = {}
        for name in selected:
            try:
                results[name] = download_one_dataset(name, DATASET_CONFIG[name])
            except SourceExhausted as exc:
                LOGGER.error("%s", exc)
                results[name] = -1
    else:
        results = download_mitbih_family()
    failed = [k for k, v in results.items() if v < 0]
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
