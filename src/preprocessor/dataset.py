from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
from typing_extensions import Self


class DataSet:
    """
        with DataSet(
            "data/dataset.h5ad",
            label_column="cell_type",
            counts_layer="counts",
        ) as dataset:
            X = dataset.X()
            Y = dataset.Y()
            counts = dataset.counts()
    """

    def __init__(
        self,
        filepath: str | Path,
        label_column: str = "cell_type",
        counts_layer: str | None = None,
    ) -> None:
        self.filepath = Path(filepath)
        self.label_column = label_column
        self.counts_layer = counts_layer
        self.data = self._load()

        self._validate()

    def _load(self) -> ad.AnnData:
        if not self.filepath.exists():
            raise FileNotFoundError(self.filepath)

        return ad.read_h5ad(self.filepath, backed="r")

    def _validate(self) -> None:
        if not self.data.obs_names.is_unique:
            raise ValueError("Cell IDs must be unique")

        if self.label_column not in self.data.obs:
            raise ValueError(f"Missing label column: {self.label_column!r}")

        if self.counts_layer is not None and self.counts_layer not in self.data.layers:
            raise ValueError(f"Missing counts layer: {self.counts_layer!r}")

    def X(self) -> Any:
        """Active expression matrix: cells x genes."""
        return self.data.X

    def raw_X(self) -> Any:
        """Explicit count matrix for count-based models such as scVI"""
        if self.data.raw is None:
            raise ValueError("Dataset does not contain data.raw")

        return self.data.raw.X

    def counts(self) -> Any:
        """Explicit count matrix for count-based models such as scVI."""
        if self.counts_layer is None:
            raise ValueError("No counts_layer was configured")

        return self.data.layers[self.counts_layer]

    def Y(self) -> pd.Series:
        """One cell-type label per cell."""
        labels = self.data.obs[self.label_column].copy()
        labels.name = "target"
        return labels

    def close(self) -> None:
        """Close the backed H5AD file."""
        self.data.file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __str__(self) -> str:
        return (
            f"DataSet(cells={self.data.n_obs}, "
            f"genes={self.data.n_vars}, "
            f"label={self.label_column!r})"
        )
