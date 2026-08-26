from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CellEmbeddings:
    matrix: np.ndarray
    cell_ids: pd.Index

    @classmethod
    def load(cls, split_dir: str | Path) -> CellEmbeddings:
        """Load embeddings written by `Representation.transform_to_file`."""
        split_dir = Path(split_dir)
        matrix = np.load(split_dir / "embeddings.npy", mmap_mode="r")
        cell_ids = pd.Index(
            pd.read_csv(
                split_dir / "cell_ids.csv.gz",
                dtype={"cell_id": str},
                compression="gzip",
            )["cell_id"]
        )
        if not cell_ids.is_unique:
            raise ValueError(f"Duplicate cell IDs in {split_dir}")
        if matrix.shape[0] != len(cell_ids):
            raise ValueError(f"Embedding rows and cell IDs do not match in {split_dir}")
        return cls(matrix=matrix, cell_ids=cell_ids)
