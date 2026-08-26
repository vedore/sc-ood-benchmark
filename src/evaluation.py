from __future__ import annotations

import logging
from collections.abc import Mapping
from time import perf_counter

import numpy as np
import pandas as pd

from metrics import compute_classification_metrics, summarize_donor_metrics

LOGGER = logging.getLogger("sc_ood_benchmark")


class RunTimer:
    """Records wall-clock durations for named operations."""

    def __init__(self) -> None:
        self._rows: list[dict[str, str | float]] = []

    def record(self, operation: str, split: str, started_at: float) -> float:
        seconds = perf_counter() - started_at
        self._rows.append({"operation": operation, "split": split, "seconds": seconds})
        LOGGER.info("Completed %s for %s in %.2f seconds", operation, split, seconds)
        return seconds

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)


class SplitEvaluator:
    """Builds split/donor/donor_mean metric rows for one evaluated split.

    `extra_fields` are merged into every row (e.g. a `model` name when
    comparing several classifiers over the same embeddings).
    """

    def __init__(self, extra_fields: Mapping[str, object] | None = None) -> None:
        self.extra_fields = dict(extra_fields or {})

    def evaluate(
        self,
        split: str,
        cell_ids: pd.Index,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        donor_ids: pd.Series,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []

        split_result = compute_classification_metrics(y_true, y_pred)
        rows.append(
            {
                **self.extra_fields,
                "split": split,
                "aggregation": "split",
                "donor_id": "",
                "n_cells": len(cell_ids),
                **split_result,
            }
        )

        donor_results = []
        for donor_id in donor_ids.drop_duplicates():
            donor_mask = donor_ids.eq(donor_id).to_numpy()
            donor_result = compute_classification_metrics(
                y_true[donor_mask], y_pred[donor_mask]
            )
            donor_results.append(donor_result)
            rows.append(
                {
                    **self.extra_fields,
                    "split": split,
                    "aggregation": "donor",
                    "donor_id": donor_id,
                    "n_cells": int(donor_mask.sum()),
                    **donor_result,
                }
            )

        donor_summary = summarize_donor_metrics(donor_results)
        rows.append(
            {
                **self.extra_fields,
                "split": split,
                "aggregation": "donor_mean",
                "donor_id": "",
                "n_cells": len(cell_ids),
                **donor_summary,
            }
        )

        context = " / ".join([*(str(v) for v in self.extra_fields.values()), split])
        LOGGER.info(
            "%s metrics: macro_f1=%.4f accuracy=%.4f",
            context,
            split_result["macro_f1"],
            split_result["accuracy"],
        )
        return rows
