import numpy as np


class WeightedEnsemble:
    def __init__(self, models, weights):
        self.models = models
        self.weights = np.array(weights) / sum(weights)

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return np.average(preds, axis=0, weights=self.weights)


def build_ensemble(models, metrics):
    weights = [m["f1_macro"] for m in metrics]
    return WeightedEnsemble(models, weights)


def evaluate_ensemble(ensemble, X, y_true, threshold_fn=None):
    y_prob = ensemble.predict(X)
    if threshold_fn:
        y_pred = threshold_fn(y_prob)
    else:
        y_pred = np.argmax(y_prob, axis=-1)
    from sklearn.metrics import classification_report, f1_score

    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "report": classification_report(y_true, y_pred, output_dict=True),
    }
