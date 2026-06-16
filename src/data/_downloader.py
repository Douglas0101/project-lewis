"""Resilient download kernel for Project-Lewis Camada 1.

Provides streaming download with retry+backoff, optional SHA256 verification,
resumable HTTP Range support, atomic writes, circuit breaker, audit logging,
and DLQ persistence. All state lives under ``data/`` per Camada 1 spec.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import signal
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Callable, Iterable, Iterator, Optional, Union, cast

LOGGER = logging.getLogger("lewis.camada01")

DLQ_PATH = Path("data/.dlq/failed_downloads.jsonl")
AUDIT_PATH = Path("data/audit/ingestion.jsonl")
CACHE_ZIPS_DIR = Path("data/.cache/zips")
LOCKS_DIR = Path("data/.cache/locks")

CHUNK_BYTES = 64 * 1024
DEFAULT_TIMEOUT_CONNECT = 15
DEFAULT_TIMEOUT_READ = 300
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 32.0
DEFAULT_CB_THRESHOLD = 10
DEFAULT_CB_COOLDOWN = 60.0


class DownloadError(Exception):
    """Base download error."""


class ChecksumMismatch(DownloadError):
    """SHA256 of the downloaded file does not match the expected value."""


class SourceExhausted(DownloadError):
    """All configured sources (primary/secondary/tertiary) failed."""


@dataclass(frozen=True)
class DownloadResult:
    url: str
    dest: Path
    sha256: str
    size_bytes: int
    elapsed_sec: float
    source: str
    resumed: bool
    attempts: int


def project_root() -> Path:
    """Return the project root (the directory containing ``src/``)."""
    return Path(__file__).resolve().parents[2]


def _resolve(p: Path) -> Path:
    """Resolve a path against the project root if not absolute."""
    p = Path(p)
    if p.is_absolute():
        return p
    return project_root() / p


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _audit_path() -> Path:
    return _resolve(AUDIT_PATH)


def _dlq_path() -> Path:
    return _resolve(DLQ_PATH)


def _json_line(event: dict) -> str:
    """Serialize ``event`` to JSON, converting non-serializable objects to str."""
    return json.dumps(event, ensure_ascii=False, default=str) + "\n"


def append_audit(event: dict) -> None:
    """Append a single event JSON line to ``data/audit/ingestion.jsonl``."""
    path = _audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_json_line(event))


def append_dlq(event: dict) -> None:
    """Append a single failure event JSON line to the DLQ."""
    path = _dlq_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_json_line(event))


@contextmanager
def atomic_write(dest: Path) -> Iterator[IO[bytes]]:
    """Write to ``dest`` via a temp file + fsync + atomic rename.

    Yields a binary file handle positioned at offset 0. The caller writes the
    payload; on context exit, data is fsynced and renamed into place. On
    exception the partial file is removed.
    """
    dest = _resolve(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".part.", suffix=".tmp", dir=dest.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            yield fh
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def http_head(url: str, *, timeout: int = DEFAULT_TIMEOUT_CONNECT) -> dict:
    """Issue a HEAD probe and return size/etag/range metadata."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return {
            "size": int(resp.headers.get("Content-Length", "0") or 0),
            "etag": resp.headers.get("ETag"),
            "accept_ranges": "bytes" in (resp.headers.get("Accept-Ranges", "") or "").lower(),
            "url": resp.geturl(),
        }


def _download_streaming(
    url: str,
    dest: Path,
    *,
    expected_sha256: Optional[str] = None,
    resume_from: int = 0,
    timeout: int = DEFAULT_TIMEOUT_READ,
) -> tuple[int, str]:
    """Stream a GET to ``dest``. Returns ``(size_bytes, sha256_hex)``.

    Uses HTTP Range when ``resume_from > 0``. The file is written atomically
    and the SHA256 is computed while streaming. A checksum mismatch deletes
    the file and raises :class:`ChecksumMismatch`.
    """
    req = urllib.request.Request(url)
    if resume_from > 0:
        req.add_header("Range", f"bytes={resume_from}-")
    hasher = hashlib.sha256()
    total = resume_from
    with (
        urllib.request.urlopen(req, timeout=timeout) as resp,  # nosec B310
        atomic_write(dest) as fh,
    ):
        if resume_from > 0:
            resp.read(resume_from)
            fh.seek(resume_from)
        while True:
            chunk = resp.read(CHUNK_BYTES)
            if not chunk:
                break
            fh.write(chunk)
            hasher.update(chunk)
            total += len(chunk)
    digest = hasher.hexdigest()
    if expected_sha256 and digest != expected_sha256:
        _resolve(dest).unlink(missing_ok=True)
        raise ChecksumMismatch(
            f"SHA256 mismatch for {dest.name}: expected "
            f"{expected_sha256[:12]}..., got {digest[:12]}... ({total} bytes)"
        )
    return total, digest


def download_url(
    url: str,
    dest: Path,
    *,
    expected_sha256: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    source: str = "primary",
    resumable: bool = True,
    timeout_read: int = DEFAULT_TIMEOUT_READ,
) -> DownloadResult:
    """Download a single URL with retry+backoff+jitter and optional Range resume.

    Raises :class:`DownloadError` after exhausting retries. The failure is
    also persisted to the DLQ.
    """
    dest = _resolve(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    attempts = 0
    last_err: Optional[BaseException] = None
    resumed = False

    for attempt in range(max_retries):
        attempts = attempt + 1
        try:
            resume_from = 0
            if resumable and dest.exists():
                resume_from = dest.stat().st_size
                resumed = resume_from > 0
            size, sha = _download_streaming(
                url,
                dest,
                expected_sha256=expected_sha256,
                resume_from=resume_from,
                timeout=timeout_read,
            )
            elapsed = time.monotonic() - t0
            result = DownloadResult(
                url=url,
                dest=dest,
                sha256=sha,
                size_bytes=size,
                elapsed_sec=elapsed,
                source=source,
                resumed=resumed,
                attempts=attempts,
            )
            append_audit(
                {
                    "event": "download_ok",
                    "ts": _now_iso(),
                    **asdict(result),
                }
            )
            LOGGER.info(
                "downloaded %s (%d bytes, %.1fs, attempt %d, resumed=%s)",
                dest.name,
                size,
                elapsed,
                attempts,
                resumed,
            )
            return result
        except (urllib.error.URLError, TimeoutError, OSError, ChecksumMismatch) as e:
            last_err = e
            LOGGER.warning(
                "attempt %d/%d failed for %s: %s",
                attempts,
                max_retries,
                dest.name,
                e,
            )
            if attempt < max_retries - 1:
                time.sleep(min(base_delay * (2**attempt), max_delay) + random.uniform(0, 1))

    append_dlq(
        {
            "event": "download_failed",
            "ts": _now_iso(),
            "url": url,
            "dest": str(dest),
            "attempts": attempts,
            "source": source,
            "error": repr(last_err),
        }
    )
    raise DownloadError(f"failed to download {url} after {attempts} attempts: {last_err}")


def download_many(
    jobs: Iterable[tuple[str, Path, Optional[str]]],
    *,
    max_workers: int = 4,
    source: str = "primary",
) -> list[DownloadResult]:
    """Run several ``download_url`` calls in parallel.

    ``jobs`` yields ``(url, dest, expected_sha256_or_None)`` tuples. Errors
    are logged to the DLQ and the failed items are silently dropped from the
    returned list.
    """
    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(download_url, url, dest, expected_sha256=sha, source=source): (url, dest)
            for url, dest, sha in jobs
        }
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except DownloadError:
                pass
    return results


def circuit_breaker(
    threshold: int = DEFAULT_CB_THRESHOLD, cooldown_sec: float = DEFAULT_CB_COOLDOWN
):
    """Decorator that opens a circuit after ``threshold`` consecutive failures.

    While the circuit is open, calls raise :class:`DownloadError` until
    ``cooldown_sec`` has elapsed. The breaker is process-local and shared
    across calls to the same decorated function.
    """
    state: dict[str, float] = {"failures": 0.0, "open_until": 0.0}

    def deco(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            now = time.monotonic()
            if now < state["open_until"]:
                raise DownloadError(
                    f"circuit breaker open for {state['open_until'] - now:.0f}s more"
                )
            try:
                out = fn(*args, **kwargs)
                state["failures"] = 0
                return out
            except Exception:
                state["failures"] += 1
                if state["failures"] >= threshold:
                    state["open_until"] = now + cooldown_sec
                    LOGGER.error(
                        "circuit breaker OPEN for %.0fs after %d failures",
                        cooldown_sec,
                        int(state["failures"]),
                    )
                raise

        return wrapper

    return deco


@contextmanager
def graceful_shutdown() -> Iterator[dict]:
    """Context manager that catches SIGINT/SIGTERM and signals graceful stop.

    Yields a mutable ``{"flag": bool}`` dict that the caller may inspect to
    abort long-running loops. Original signal handlers are restored on exit.
    """
    stop = {"flag": False}

    def _handler(signum, _frame):
        stop["flag"] = True
        LOGGER.warning("signal %s received — will stop after current chunk", signum)

    previous: list[tuple[int, Union[Callable, int, None]]] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous.append(
                (int(sig), signal.signal(sig, _handler))  # type: ignore[arg-type,assignment]
            )
        except (ValueError, OSError):
            pass
    try:
        yield stop
    finally:
        for s, prev in previous:
            if prev is not None and callable(prev):
                try:
                    signal.signal(s, cast(Callable[..., Any], prev))
                except (ValueError, OSError):
                    pass


def ensure_dirs() -> None:
    """Create all Camada 1 standard directories if missing."""
    for d in (CACHE_ZIPS_DIR, LOCKS_DIR, DLQ_PATH.parent, AUDIT_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
