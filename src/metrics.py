from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
) -> dict[str, float]:
    """Compute shared cell-level classification metrics."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    if y_true_array.ndim != 1 or y_pred_array.ndim != 1:
        raise ValueError("Metric inputs must be one-dimensional")
    if len(y_true_array) == 0:
        raise ValueError("Metric inputs must not be empty")
    if len(y_true_array) != len(y_pred_array):
        raise ValueError("Metric inputs must have the same length")

    return {
        "macro_f1": float(
            f1_score(y_true_array, y_pred_array, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
    }
