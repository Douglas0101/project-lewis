"""DLQ replay entry point for Project-Lewis Camada 1.

Reads ``data/.dlq/failed_downloads.jsonl`` and retries each failed URL using
the same resilient downloader. Entries that succeed are dropped; entries
that still fail are rewritten back to the DLQ file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ._downloader import (
    DLQ_PATH,
    DownloadError,
    _now_iso,
    append_audit,
    download_url,
    project_root,
)

LOGGER = logging.getLogger("lewis.camada01")


def replay_dlq(dlq_path: Path = DLQ_PATH) -> dict[str, int]:
    """Retry every entry of the DLQ. Returns ``{success, kept, invalid}`` counts."""
    dlq = project_root() / dlq_path
    if not dlq.exists():
        LOGGER.info("DLQ does not exist — nothing to replay")
        return {"success": 0, "kept": 0, "invalid": 0}

    successes = kept = invalid = 0
    kept_lines: list[str] = []
    for raw in dlq.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            entry: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            LOGGER.error("malformed DLQ line: %s (%s)", raw[:120], exc)
            invalid += 1
            continue
        url = entry.get("url")
        dest = entry.get("dest")
        if not url or not dest:
            LOGGER.error("DLQ entry missing url/dest: %s", entry)
            invalid += 1
            continue
        try:
            download_url(url, Path(dest), source=entry.get("source", "replay"))
            successes += 1
        except DownloadError as exc:
            LOGGER.warning("replay failed: %s — %s", url, exc)
            kept_lines.append(raw)
            kept += 1
    dlq.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
    append_audit(
        {
            "event": "dlq_replay_done",
            "ts": _now_iso(),
            "success": successes,
            "kept": kept,
            "invalid": invalid,
        }
    )
    LOGGER.info("DLQ replay: %d ok, %d kept, %d invalid", successes, kept, invalid)
    return {"success": successes, "kept": kept, "invalid": invalid}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    counts = replay_dlq()
    return 0 if counts["kept"] == 0 and counts["invalid"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
