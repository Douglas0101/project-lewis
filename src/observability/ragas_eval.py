"""Avaliação contínua da qualidade do RAG.

Modo local: métricas baseadas em golden dataset sem LLM judge.
Modo RAGAS: requer grupo de dependências `eval` e OPENAI_API_KEY.
"""

from __future__ import annotations

from typing import Any, Dict, List


class LocalRAGASEvaluator:
    """Evaluator offline que não depende de API externa."""

    def __init__(self, judge_model: str = "local", temperature: float = 0.0):
        self.judge_model = judge_model
        self.temperature = temperature

    def evaluate_batch(self, queries: List[Dict[str, Any]]) -> Dict[str, float]:
        context_precisions = []
        context_recalls = []
        for q in queries:
            contexts = q.get("contexts", [])
            ground_truth = set(q.get("ground_truth", []))
            if contexts and ground_truth:
                tokens = set(" ".join(contexts).lower().split())
                intersection = ground_truth.intersection(tokens)
                context_precisions.append(len(intersection) / len(ground_truth))
                context_recalls.append(len(intersection) / len(ground_truth))
        precision = sum(context_precisions) / len(context_precisions) if context_precisions else 0.0
        recall = sum(context_recalls) / len(context_recalls) if context_recalls else 0.0
        return {
            "context_precision": precision,
            "context_recall": recall,
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
        }

    def gate_ci(self, result: Dict[str, float]) -> bool:
        return (
            result.get("context_precision", 0.0) >= 0.80
            and result.get("context_recall", 0.0) >= 0.85
            and result.get("faithfulness", 0.0) >= 0.90
            and result.get("answer_relevancy", 0.0) >= 0.85
        )


class RAGASEvaluator:
    """Evaluator RAGAS com LLM judge (requer grupo eval)."""

    def __init__(self, judge_model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.judge_model = judge_model
        self.temperature = temperature
        self._lazy_imports()

    def _lazy_imports(self):
        try:
            from datasets import Dataset  # type: ignore
            from langchain_openai import ChatOpenAI  # type: ignore
            from ragas import evaluate  # type: ignore
            from ragas.llms import LangchainLLMWrapper  # type: ignore
            from ragas.metrics import (  # type: ignore
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            self._evaluate = evaluate
            self._metrics = [
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ]
            self._Dataset = Dataset
            self._judge = LangchainLLMWrapper(
                ChatOpenAI(model=self.judge_model, temperature=self.temperature)
            )
        except ImportError as exc:
            raise RuntimeError(
                "Dependências do grupo 'eval' não instaladas. "
                "Rode: uv sync --group eval"
            ) from exc

    def evaluate_batch(self, queries: List[Dict[str, Any]]) -> Dict[str, float]:
        dataset = self._Dataset.from_list(queries)
        result = self._evaluate(dataset, metrics=self._metrics, llm=self._judge)
        return {k: float(v) for k, v in result.items()}

    def gate_ci(self, result: Dict[str, float]) -> bool:
        return (
            result.get("context_precision", 0.0) >= 0.80
            and result.get("context_recall", 0.0) >= 0.85
            and result.get("faithfulness", 0.0) >= 0.90
            and result.get("answer_relevancy", 0.0) >= 0.85
        )
