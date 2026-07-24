"""Blocked legacy matrix entry point.

A historical unauthenticated record-grouped snapshot, including its final
pre-archive safety guard, is retained as non-executable text at
``scripts/legacy/run_stage1_matrix_v3_v3.0.0_unauthenticated.py.txt``.
It must not be used for new training or to reinterpret the existing 100 cells.
Task #3 will provide a generation-bound canonical runner after preflight passes.
"""

from __future__ import annotations

import sys

LEGACY_V3_BLOCK_MESSAGE = (
    "LEGACY_V3_TRAINING_BLOCKED: the record-grouped v3 runner is not eligible for new "
    "training. Run `uv run --locked python -m src.cli.advanced_training_v3 preflight` "
    "and wait for task #3 to provide a generation-bound runner."
)


def main() -> int:
    """Refuse all legacy training attempts before importing ML dependencies."""
    print(LEGACY_V3_BLOCK_MESSAGE, file=sys.stderr)
    return 10


if __name__ == "__main__":
    raise SystemExit(main())
