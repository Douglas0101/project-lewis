"""CLI para rodar RAGAS eval a partir de golden dataset JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.observability.ragas_eval import LocalRAGASEvaluator, RAGASEvaluator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAGAS eval CLI")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="data/eval/golden_dataset.json",
        help="Caminho do golden dataset JSON",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Usa RAGAS com LLM judge (requer grupo eval)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    path = Path(args.dataset)
    queries = json.loads(path.read_text(encoding="utf-8"))
    evaluator = RAGASEvaluator() if args.llm else LocalRAGASEvaluator()
    result = evaluator.evaluate_batch(queries)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if evaluator.gate_ci(result) else 1


if __name__ == "__main__":
    sys.exit(main())
