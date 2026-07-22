"""Blocked legacy Stage 1 sanity-cell entry point.

The historical v3 implementation snapshot is archived as non-executable data at
``scripts/legacy/run_stage1_sanity_cell_v3.py.txt.gz``. It uses record-grouped
v3 splits and is ineligible for new evidence generation.
"""

from __future__ import annotations

import sys

LEGACY_V3_BLOCK_MESSAGE = (
    "LEGACY_V3_TRAINING_BLOCKED: the Stage 1 sanity runner uses invalid record-grouped "
    "v3 splits. Run the canonical v3.1 preflight and wait for a generation-bound runner."
)


def main() -> int:
    """Refuse legacy training before importing numerical or ML dependencies."""
    print(LEGACY_V3_BLOCK_MESSAGE, file=sys.stderr)
    return 10


if __name__ == "__main__":
    raise SystemExit(main())
