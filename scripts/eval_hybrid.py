"""Calcula MRR, Context Precision e Context Recall no golden dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

from src.knowledge.retriever import hybrid_search, search
from src.knowledge.schemas import QueryRequest


def load_golden(path: Path | None = None) -> List[Dict[str, Any]]:
    path = path or Path("data/eval/golden_dataset.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _result_matches(result, expected_chunks: Set[str], expected_sources: Set[str]) -> bool:
    """Verifica se um resultado é relevante por chunk_id ou source."""
    if result.chunk_id in expected_chunks:
        return True
    if result.source in expected_sources:
        return True
    return False


def mrr(queries: List[Dict[str, Any]]) -> float:
    ranks = []
    for q in queries:
        results = hybrid_search(
            QueryRequest(query=q["question"], layer=None, version=None, tags=None, k=5, fetch_k=20)
        )
        expected_chunks = set(q.get("expected_chunk_ids", []))
        expected_sources = set(q.get("expected_sources", []))
        for rank, r in enumerate(results, start=1):
            if _result_matches(r, expected_chunks, expected_sources):
                ranks.append(1.0 / rank)
                break
        else:
            ranks.append(0.0)
    return sum(ranks) / len(ranks) if ranks else 0.0


def context_precision(queries: List[Dict[str, Any]]) -> float:
    precisions = []
    for q in queries:
        results = hybrid_search(
            QueryRequest(query=q["question"], layer=None, version=None, tags=None, k=5, fetch_k=20)
        )
        expected_chunks = set(q.get("expected_chunk_ids", []))
        expected_sources = set(q.get("expected_sources", []))
        retrieved = [r for r in results]
        if not retrieved:
            continue
        relevant = sum(
            1 for r in retrieved if _result_matches(r, expected_chunks, expected_sources)
        )
        precisions.append(relevant / len(retrieved))
    return sum(precisions) / len(precisions) if precisions else 0.0


def context_recall(queries: List[Dict[str, Any]]) -> float:
    recalls = []
    for q in queries:
        results = hybrid_search(
            QueryRequest(query=q["question"], layer=None, version=None, tags=None, k=5, fetch_k=20)
        )
        expected_chunks = set(q.get("expected_chunk_ids", []))
        expected_sources = set(q.get("expected_sources", []))
        if not expected_chunks and not expected_sources:
            continue
        retrieved = list(results)
        relevant = sum(
            1 for r in retrieved if _result_matches(r, expected_chunks, expected_sources)
        )
        recalls.append(relevant / max(len(expected_chunks), len(expected_sources)))
    return sum(recalls) / len(recalls) if recalls else 0.0


def compare_baseline(queries: List[Dict[str, Any]]) -> Dict[str, float]:
    hybrid_mrr = mrr(queries)
    baseline_ranks = []
    for q in queries:
        results = search(
            QueryRequest(query=q["question"], layer=None, version=None, tags=None, k=5, fetch_k=20)
        )
        expected_chunks = set(q.get("expected_chunk_ids", []))
        expected_sources = set(q.get("expected_sources", []))
        for rank, r in enumerate(results, start=1):
            if _result_matches(r, expected_chunks, expected_sources):
                baseline_ranks.append(1.0 / rank)
                break
        else:
            baseline_ranks.append(0.0)
    baseline_mrr = sum(baseline_ranks) / len(baseline_ranks) if baseline_ranks else 0.0
    return {
        "hybrid_mrr": hybrid_mrr,
        "baseline_mrr": baseline_mrr,
        "delta": hybrid_mrr - baseline_mrr,
    }


if __name__ == "__main__":
    queries = load_golden()
    report = {
        "queries": len(queries),
        "mrr": mrr(queries),
        "context_precision": context_precision(queries),
        "context_recall": context_recall(queries),
        "comparison": compare_baseline(queries),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
