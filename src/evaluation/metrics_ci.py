from typing import Callable, Dict, Tuple

import numpy as np


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    rng = np.random.RandomState(random_state)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return float(np.mean(scores)), float(lower), float(upper)


def report_ci(
    y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict[str, Callable]
) -> Dict[str, dict]:
    result = {}
    for name, fn in metrics.items():
        mean, lower, upper = bootstrap_ci(y_true, y_pred, fn)
        result[name] = {"mean": mean, "ci_95": [lower, upper]}
    return result
