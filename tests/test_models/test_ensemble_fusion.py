import numpy as np

from src.models.ensemble_fusion import (
    WeightedEnsemble,
    build_ensemble,
    evaluate_ensemble,
)


class DummyModel:
    def __init__(self, probs):
        self.probs = np.array(probs)

    def predict(self, X):
        return self.probs


def test_ensemble_weights_sum_to_one():
    models = [DummyModel([[0.5, 0.5]]), DummyModel([[0.5, 0.5]])]
    ensemble = WeightedEnsemble(models, weights=[2, 3])
    assert np.isclose(ensemble.weights.sum(), 1.0)
    assert len(ensemble.weights) == len(models)


def test_ensemble_predict_averages_probabilities():
    model_a = DummyModel([[0.8, 0.2], [0.4, 0.6]])
    model_b = DummyModel([[0.2, 0.8], [0.6, 0.4]])
    ensemble = WeightedEnsemble([model_a, model_b], weights=[1, 1])

    result = ensemble.predict(None)
    expected = np.array([[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_allclose(result, expected)

    ensemble_weighted = WeightedEnsemble([model_a, model_b], weights=[1, 3])
    result_weighted = ensemble_weighted.predict(None)
    expected_weighted = np.array(
        [
            [0.8 * 0.25 + 0.2 * 0.75, 0.2 * 0.25 + 0.8 * 0.75],
            [0.4 * 0.25 + 0.6 * 0.75, 0.6 * 0.25 + 0.4 * 0.75],
        ]
    )
    np.testing.assert_allclose(result_weighted, expected_weighted)


def test_evaluate_ensemble_returns_f1_macro():
    model = DummyModel([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    metrics = [{"f1_macro": 0.6}, {"f1_macro": 0.4}]
    ensemble = build_ensemble([model, model], metrics)

    y_true = np.array([0, 1, 0])
    result = evaluate_ensemble(ensemble, X=None, y_true=y_true)

    assert "f1_macro" in result
    assert "report" in result
    assert isinstance(result["f1_macro"], (float, np.floating))
    assert isinstance(result["report"], dict)
    assert result["f1_macro"] > 0.0
