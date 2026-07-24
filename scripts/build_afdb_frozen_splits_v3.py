"""Blocked legacy AFDB record-grouped split builder.

The original implementation is preserved at
``scripts/legacy/build_afdb_frozen_splits_v3_unauthenticated.py.txt.gz``.
AFDB remains exploratory until its patient mapping is authenticated.
"""

from __future__ import annotations

import sys

MESSAGE = (
    "LEGACY_V3_SPLIT_BUILDER_BLOCKED: AFDB patient identity is unresolved. "
    "Run `uv run --locked python -m src.cli.advanced_training_v3 preflight` "
    "and retain AFDB in the RHYTHM_EXPLORATORY role."
)


def main() -> int:
    print(MESSAGE, file=sys.stderr)
    return 10


if __name__ == "__main__":
    raise SystemExit(main())
