"""Refresh assíncrono da tabela materializada experiment_timeline."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class TimelineRefresher:
    """Atualiza experiment_timeline_materialized a partir da view."""

    def __init__(self, db_path: str | Path, interval_seconds: int = 300):
        self.db_path = str(db_path)
        self.interval = interval_seconds
        self.last_refresh: Optional[datetime] = None

    async def refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            await self._refresh()

    async def _refresh(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.refresh_sync)

    def refresh_sync(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute("PRAGMA cache_size = -1048576")
            conn.execute("BEGIN TRANSACTION")
            conn.execute("DELETE FROM experiment_timeline_materialized")
            conn.execute(
                "INSERT INTO experiment_timeline_materialized SELECT * FROM experiment_timeline"
            )
            conn.execute("COMMIT")
            self.last_refresh = datetime.now(timezone.utc)
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
