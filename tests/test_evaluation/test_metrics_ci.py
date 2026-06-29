import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.metrics_ci import bootstrap_ci, report_ci


@pytest.fixture
def sample_arrays():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 2, 1, 0, 1, 2, 0, 0, 1, 0])
    return y_true, y_pred


def test_bootstrap_ci_bounds(sample_arrays):
    y_true, y_pred = sample_arrays
    mean, lower, upper = bootstrap_ci(y_true, y_pred, accuracy_score)
    assert lower <= mean <= upper


def test_report_ci_returns_expected_keys(sample_arrays):
    y_true, y_pred = sample_arrays
    metrics = {
        "accuracy": accuracy_score,
        "f1_macro": lambda yt, yp: f1_score(yt, yp, average="macro"),
    }
    report = report_ci(y_true, y_pred, metrics)
    assert set(report.keys()) == set(metrics.keys())
    for key in report:
        assert "mean" in report[key]
        assert "ci_95" in report[key]
        assert len(report[key]["ci_95"]) == 2


def test_bootstrap_ci_reproducible(sample_arrays):
    y_true, y_pred = sample_arrays
    result_1 = bootstrap_ci(y_true, y_pred, accuracy_score, random_state=123)
    result_2 = bootstrap_ci(y_true, y_pred, accuracy_score, random_state=123)
    assert result_1 == result_2
