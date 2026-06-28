import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def confusion_matrix_norm(
    y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray = None
) -> np.ndarray:
    """Return a row-normalized confusion matrix (recall per true class)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / (row_sums + 1e-9)
    return cm_norm


def confusion_matrix_report(
    y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray = None
) -> pd.DataFrame:
    """Return a per-class recall/precision DataFrame from a normalized CM."""
    cm_norm = confusion_matrix_norm(y_true, y_pred, labels=labels)
    recall_per_class = np.diag(cm_norm)
    precision_per_class = cm_norm.max(axis=0)  # simplificado
    return pd.DataFrame(
        {
            "recall": recall_per_class,
            "precision": precision_per_class,
        },
        index=labels,
    )
