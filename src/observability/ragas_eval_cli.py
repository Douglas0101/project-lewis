"""CLI para rodar RAGAS eval a partir de golden dataset JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.observability.ragas_eval import LocalRAGASEvaluator, RAGASEvaluator


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/eval/golden_dataset.json")
    queries = json.loads(path.read_text(encoding="utf-8"))
    use_llm = "--llm" in sys.argv
    evaluator = RAGASEvaluator() if use_llm else LocalRAGASEvaluator()
    result = evaluator.evaluate_batch(queries)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if evaluator.gate_ci(result) else 1


if __name__ == "__main__":
    sys.exit(main())
