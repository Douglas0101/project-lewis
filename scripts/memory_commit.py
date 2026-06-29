#!/usr/bin/env python3
"""Registra artefatos de uma run no ArtifactRegistry."""

import argparse
from pathlib import Path

from src.memory.registry import record_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Registra artefato no ArtifactRegistry")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--type", required=True)
    args = parser.parse_args()

    art_id = record_artifact(args.run_id, args.path, args.type)
    print(f"artifact_id={art_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
