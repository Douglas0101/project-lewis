"""Blocked legacy record-grouped split builder.

The original implementation is preserved at
``scripts/legacy/build_frozen_splits_v3_unauthenticated.py.txt.gz``.
Use the canonical advanced-training v3.1 preflight instead.
"""

from __future__ import annotations

import sys

MESSAGE = (
    "LEGACY_V3_SPLIT_BUILDER_BLOCKED: record_id is not patient_id. "
    "Run `uv run --locked python -m src.cli.advanced_training_v3 preflight "
    "--publish-splits` to build immutable patient-aware v3.1 splits."
)


def main() -> int:
    print(MESSAGE, file=sys.stderr)
    return 10


if __name__ == "__main__":
    raise SystemExit(main())
