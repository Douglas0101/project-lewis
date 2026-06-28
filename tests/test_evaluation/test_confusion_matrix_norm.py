import numpy as np
import pytest

from src.evaluation.confusion_matrix_norm import (
    confusion_matrix_norm,
    confusion_matrix_report,
)


def test_row_normalization_recall():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    cm_norm = confusion_matrix_norm(y_true, y_pred, labels=[0, 1])
    assert cm_norm[0, 0] == pytest.approx(0.5)
    assert cm_norm[1, 1] == pytest.approx(1.0)


def test_confusion_matrix_report_shape():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    labels = [0, 1]
    report = confusion_matrix_report(y_true, y_pred, labels=labels)
    assert list(report.index) == labels
    assert list(report.columns) == ["recall", "precision"]
