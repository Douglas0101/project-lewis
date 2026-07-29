"""Chapman-Shaoxing downloader (Camada 1 CLI).

Idempotent: when the raw directory already holds a complete dataset
(>= EXPECTED_MIN_RECORDS records), the download is skipped and the
command exits 0 without touching the network. Use ``--force`` to
re-download anyway, or ``--verify`` for an offline integrity check.

Resolution order: local mirror → incremental wget (PhysioNet) →
static ZIP (PhysioNet) → kagglehub.

Usage:
    python -m src.data.download_chapman [--force] [--verify]
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import subprocess  # nosec B404
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Optional

from ._catalog import RAW_DATASETS
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

KAGGLE_SLUG = "erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database"
EXPECTED_MIN_RECORDS = 45_000
CHAPMAN_MIRROR = Path("data/mirrors/chapman_mirror.tar.gz")
CHAPMAN_PN_URL = (
    "https://physionet.org/static/published-projects/challenge-2021/"
    "1.0.3/training/chapman_shaoxing.zip"
)
# Minimum .hea count per Challenge 2021 subset to consider it fully mirrored
# (PhysioNet: chapman_shaoxing ~10.3k records, ningbo ~34.9k records).
WGET_SUBSET_MIN_RECORDS = {"chapman_shaoxing": 10_000, "ningbo": 34_000}
VERIFY_HEADER_SAMPLE = 50


def _kagglehub_available() -> bool:
    try:
        import kagglehub  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


@circuit_breaker(threshold=5, cooldown_sec=120.0)
def _try_kagglehub(raw_dir: Path, *, force: bool = False) -> bool:
    if any(raw_dir.iterdir()) and not force:
        LOGGER.info(
            "kagglehub skipped: %s is not empty (use --force to re-download)", raw_dir
        )
        return False
    if not _kagglehub_available():
        return False
    try:
        import kagglehub  # type: ignore

        kagglehub.dataset_download(
            KAGGLE_SLUG, output_dir=str(raw_dir), force_download=force
        )
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

    Incremental: ``-c`` resumes partial files and ``-N`` skips files already
    mirrored locally, so re-runs only fetch what is missing. Subsets whose
    local record count already meets ``WGET_SUBSET_MIN_RECORDS`` are skipped
    entirely.
    """
    if shutil.which("wget") is None:
        LOGGER.error("wget is not installed; cannot use recursive source")
        return False

    raw_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://physionet.org/files/challenge-2021/1.0.3/training"
    ok = True
    for subset, min_records in WGET_SUBSET_MIN_RECORDS.items():
        subset_dir = raw_dir / subset
        existing = sum(1 for _ in subset_dir.rglob("*.hea")) if subset_dir.exists() else 0
        if existing >= min_records:
            LOGGER.info(
                "subset %s already complete (%d >= %d records) — skipped",
                subset,
                existing,
                min_records,
            )
            continue
        subset_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "wget",
            "-r",  # recursive
            "-np",  # no parent
            "-nH",  # no host prefix
            "-c",  # resume partial files
            "-N",  # timestamping: skip files already on disk
            "--tries=3",
            "--waitretry=5",
            "--cut-dirs=5",
            "-R",
            "index.html*",
            "-P",
            str(subset_dir),
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


def _parse_header_sample(hea_files: list[Path], sample_size: int) -> list[str]:
    """Parse a random sample of WFDB headers; return a list of problems."""
    try:
        import wfdb  # type: ignore
    except ImportError:
        LOGGER.warning("wfdb is not installed; header sample parse skipped")
        return []
    problems: list[str] = []
    sample = random.sample(hea_files, k=min(sample_size, len(hea_files)))
    for hea in sample:
        try:
            wfdb.rdheader(str(hea.with_suffix("")))
        except Exception as exc:
            problems.append(f"unparseable header {hea}: {exc}")
    return problems


def _declared_signal_files(hea: Path) -> list[str]:
    """Return the signal filenames declared in a WFDB header (deduplicated)."""
    names: list[str] = []
    try:
        with hea.open("r", encoding="utf-8", errors="replace") as fh:
            next(fh, None)  # record line
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    break
                names.append(line.split()[0])
    except OSError as exc:
        LOGGER.warning("cannot read header %s: %s", hea, exc)
    return list(dict.fromkeys(names))


def verify_chapman(raw_dir: Optional[Path] = None) -> tuple[int, list[str]]:
    """Offline integrity check of the local Chapman dataset.

    Returns ``(record_count, problems)``. Checks: record count vs
    ``EXPECTED_MIN_RECORDS``, the signal file declared by every ``.hea``
    (e.g. ``S23074.hea`` → ``JS23074.mat``), and a random sample of headers
    parseable by wfdb.
    """
    raw_dir = (project_root() / raw_dir) if raw_dir else _chapman_raw_dir()
    problems: list[str] = []
    hea_files = sorted(raw_dir.rglob("*.hea")) if raw_dir.exists() else []
    for hea in hea_files:
        declared = _declared_signal_files(hea)
        if not declared:
            problems.append(f"no signal files declared in {hea}")
            continue
        for sig in declared:
            if not (hea.parent / sig).exists():
                problems.append(f"missing signal file {sig} declared by {hea}")
    n = _count_records(raw_dir)
    if n < EXPECTED_MIN_RECORDS:
        problems.append(f"{n} records found, expected >= {EXPECTED_MIN_RECORDS}")
    problems.extend(_parse_header_sample(hea_files, VERIFY_HEADER_SAMPLE))
    return n, problems


def download_chapman(
    raw_dir: Optional[Path] = None,
    mirror_path: Optional[Path] = None,
    *,
    force: bool = False,
) -> int:
    """Download Chapman-Shaoxing using the source cascade.

    Idempotent: returns immediately (no network) when the local dataset is
    already complete, unless ``force`` is set. Returns the number of records
    detected on success. Raises :class:`SourceExhausted` if every source
    fails.
    """
    raw_dir = (project_root() / raw_dir) if raw_dir else _chapman_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    mirror = (project_root() / mirror_path) if mirror_path else (project_root() / CHAPMAN_MIRROR)

    present = _count_records(raw_dir)
    if not force and present >= EXPECTED_MIN_RECORDS:
        append_audit(
            {
                "event": "chapman_skip_present",
                "ts": _now_iso(),
                "records": present,
            }
        )
        LOGGER.info(
            "Chapman already present (%d >= %d records) — download skipped "
            "(use --force to re-download)",
            present,
            EXPECTED_MIN_RECORDS,
        )
        return present

    zip_cache = project_root() / CACHE_ZIPS_DIR / "chapman_shaoxing.zip"
    attempts: list[tuple[str, Callable[[], bool]]] = [
        ("mirror", lambda: _try_mirror(raw_dir, mirror) if mirror.exists() else False),
        ("physionet-wget", lambda: _try_wget_recursive(raw_dir)),
        ("physionet-zip", lambda: _try_physionet_zip(raw_dir, zip_cache)),
        ("kagglehub", lambda: _try_kagglehub(raw_dir, force=force)),
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
    raise SourceExhausted(f"Chapman-Shaoxing: all {len(attempts)} sources failed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chapman-Shaoxing downloader (Camada 1, idempotent)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even when the local dataset is already complete",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="offline integrity check of the local dataset (no download)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.verify:
        n, problems = verify_chapman()
        if problems:
            for problem in problems[:20]:
                LOGGER.error("verify: %s", problem)
            LOGGER.error("Chapman verify FAILED: %d records, %d problems", n, len(problems))
            return 1
        LOGGER.info("Chapman verify OK: %d records", n)
        return 0
    try:
        download_chapman(force=args.force)
        return 0
    except SourceExhausted as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
