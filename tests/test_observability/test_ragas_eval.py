from src.observability.ragas_eval import LocalRAGASEvaluator


def test_local_context_precision():
    queries = [
        {
            "question": "q1",
            "contexts": ["Run run_001 completado. Accuracy final: 0.92."],
            "ground_truth": ["run_001", "0.92"],
        }
    ]
    evaluator = LocalRAGASEvaluator()
    result = evaluator.evaluate_batch(queries)
    assert 0.0 <= result["context_precision"] <= 1.0
    assert 0.0 <= result["context_recall"] <= 1.0
    assert "faithfulness" in result
    assert "answer_relevancy" in result
