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


def summarize_donor_metrics(
    donor_metrics: Sequence[dict[str, float]],
) -> dict[str, float]:
    """Aggregate per-donor metric dicts into a mean and standard deviation.

    Each donor's macro-F1 is averaged over whatever classes appear in that
    donor's own true/predicted labels (see `compute_classification_metrics`),
    so a class a donor simply lacks does not count as an error. Standard
    deviation is `nan` for a single donor, since spread across donors is
    undefined with one sample.
    """
    if not donor_metrics:
        raise ValueError("At least one donor metric is required")

    macro_f1_values = np.array([donor["macro_f1"] for donor in donor_metrics])
    accuracy_values = np.array([donor["accuracy"] for donor in donor_metrics])
    n_donors = len(donor_metrics)

    return {
        "n_donors": n_donors,
        "macro_f1": float(np.mean(macro_f1_values)),
        "macro_f1_std": (
            float(np.std(macro_f1_values, ddof=1)) if n_donors > 1 else float("nan")
        ),
        "accuracy": float(np.mean(accuracy_values)),
        "accuracy_std": (
            float(np.std(accuracy_values, ddof=1)) if n_donors > 1 else float("nan")
        ),
    }
