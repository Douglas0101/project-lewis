"""Chapman-Shaoxing downloader (Camada 1 CLI).

Resolution order: kagglehub → PhysioNet challenge ZIP → mirror tarball.

Usage:
    python -m src.data.download_chapman
"""

from __future__ import annotations

import logging
import shutil
import subprocess  # nosec B404
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Optional

from ._catalog import RAW_DATASETS
from ._downloader import (
    LOGGER,
    DownloadError,
    SourceExhausted,
    _now_iso,
    append_audit,
    circuit_breaker,
    download_url,
    project_root,
)

CHAPMAN_PN_DB = "challenge-2021/1.0.3"
KAGGLE_SLUG = "erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database"
EXPECTED_MIN_RECORDS = 45_000
CHAPMAN_MIRROR = Path("data/mirrors/chapman_mirror.tar.gz")
CHAPMAN_PN_URL = (
    "https://physionet.org/static/published-projects/challenge-2021/"
    "1.0.3/training/chapman_shaoxing.zip"
)


def _kagglehub_available() -> bool:
    try:
        import kagglehub  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


@circuit_breaker(threshold=5, cooldown_sec=120.0)
def _try_kagglehub(raw_dir: Path) -> bool:
    if not _kagglehub_available():
        return False
    try:
        import kagglehub  # type: ignore

        kagglehub.dataset_download(KAGGLE_SLUG, output_dir=str(raw_dir))
        return True
    except Exception as exc:
        LOGGER.warning("kagglehub download failed: %s", exc)
        return False


def _try_physionet_zip(raw_dir: Path, zip_cache: Path) -> bool:
    if not zip_cache.exists():
        try:
            download_url(CHAPMAN_PN_URL, zip_cache, source="primary")
        except DownloadError:
            return False
    try:
        with zipfile.ZipFile(zip_cache) as zf:
            zf.extractall(raw_dir)  # nosec B202
        return True
    except zipfile.BadZipFile as exc:
        LOGGER.error("invalid Chapman ZIP: %s", exc)
        zip_cache.unlink(missing_ok=True)
        return False


def _try_physionet_wfdb(raw_dir: Path) -> bool:
    """Secondary source: use ``wfdb.dl_database`` to pull the whole tree.

    PhysioNet does not ship a single ZIP for ``challenge-2021/1.0.3`` — the
    Chapman-Shaoxing records live under the ``chapman_shaoxing/`` sub-tree
    (a directory, not an archive). ``wfdb-python`` knows how to enumerate
    and download it record by record.
    """
    try:
        import wfdb  # type: ignore
    except ImportError:
        LOGGER.error("wfdb is not installed; cannot use the wfdb source")
        return False
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "chapman_shaoxing"
    target.mkdir(parents=True, exist_ok=True)
    try:
        wfdb.dl_database(CHAPMAN_PN_DB, dl_dir=str(target), records="chapman_shaoxing")
    except Exception as exc:
        LOGGER.warning("wfdb.dl_database failed for %s: %s", CHAPMAN_PN_DB, exc)
        return False
    return True


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


def _count_records(raw_dir: Path) -> int:
    hea = sum(1 for _ in raw_dir.rglob("*.hea"))
    csv = sum(1 for _ in raw_dir.rglob("*.csv"))
    return hea + csv


def _chapman_raw_dir() -> Path:
    return project_root() / RAW_DATASETS["chapman"]


def _try_wget_recursive(raw_dir: Path) -> bool:
    """Download Chapman-Shaoxing + Ningbo from PhysioNet Challenge 2021.

    The Challenge 2021 training data is split into ``chapman_shaoxing/`` and
    ``ningbo/`` sub-trees, each further divided into ``g#`` folders of up to
    1000 records.  A recursive ``wget`` is the fastest reliable way to mirror
    the ~45k records (5.1 GB) because PhysioNet no longer offers a single ZIP.
    """
    if shutil.which("wget") is None:
        LOGGER.error("wget is not installed; cannot use recursive source")
        return False

    raw_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://physionet.org/files/challenge-2021/1.0.3/training"
    ok = True
    for subset in ("chapman_shaoxing", "ningbo"):
        subset_dir = raw_dir / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "wget",
            "-r",  # recursive
            "-np",  # no parent
            "-nH",  # no host prefix
            "--cut-dirs=5",
            "-R", "index.html*",
            "-P", str(subset_dir),
            f"{base_url}/{subset}/",
        ]
        LOGGER.info("starting recursive wget for %s", subset)
        try:
            subprocess.run(cmd, check=True, timeout=7200)  # nosec B603
        except subprocess.TimeoutExpired:
            LOGGER.warning("recursive wget timed out for %s", subset)
            ok = False
        except subprocess.CalledProcessError as exc:
            LOGGER.warning("recursive wget failed for %s: %s", subset, exc)
            ok = False
    return ok


def download_chapman(
    raw_dir: Optional[Path] = None,
    mirror_path: Optional[Path] = None,
) -> int:
    """Download Chapman-Shaoxing using the source cascade.

    Returns the number of records detected on success. Raises
    :class:`SourceExhausted` if every source fails.
    """
    raw_dir = (project_root() / raw_dir) if raw_dir else _chapman_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    mirror = (project_root() / mirror_path) if mirror_path else (project_root() / CHAPMAN_MIRROR)

    attempts: list[tuple[str, Callable[[], bool]]] = [
        ("physionet-wget", lambda: _try_wget_recursive(raw_dir)),
        ("kagglehub", lambda: _try_kagglehub(raw_dir)),
        ("physionet-wfdb", lambda: _try_physionet_wfdb(raw_dir)),
        ("mirror", lambda: _try_mirror(raw_dir, mirror) if mirror.exists() else False),
    ]
    for source, fn in attempts:
        if not fn():
            continue
        n = _count_records(raw_dir)
        append_audit(
            {
                "event": "chapman_done",
                "ts": _now_iso(),
                "source": source,
                "records": n,
            }
        )
        if n >= EXPECTED_MIN_RECORDS:
            LOGGER.info("Chapman OK via %s: %d records", source, n)
            return n
        LOGGER.warning("Chapman via %s returned %d records (< %d)", source, n, EXPECTED_MIN_RECORDS)
    raise SourceExhausted("Chapman-Shaoxing: all 3 sources failed")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        download_chapman()
        return 0
    except SourceExhausted as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
