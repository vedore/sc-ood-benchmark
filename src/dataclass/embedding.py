from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CellEmbeddings:
    matrix: np.ndarray
    cell_ids: pd.Index
