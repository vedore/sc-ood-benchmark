from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA

from dataclass.embedding import CellEmbeddings
from representations.base import BaseRepresentation


class PCARepresentation(BaseRepresentation):
    """
    - PCARepresentation:
        - Uses normalized adata.X.
        - Selects HVGs using train only.
        - Fits PCA using train only.
        - Stores HVGs and PCA parameters.
    """

    def __init__(
        self, n_hvgs: int = 2000, n_components: int = 20, seed: int = 42
    ) -> None:
        self.n_hvgs = n_hvgs
        self.n_components = n_components
        self.seed = seed

        self.gene_ids_: pd.Index | None = None
        self.pca_: PCA | None = None

    def fit(self, train: ad.AnnData) -> PCARepresentation:
        self.gene_ids_ = self._select_train_hvgs(train, n_hvgs=self.n_hvgs)

        X_train = self._get_expression_matrix(
            train,
            gene_ids=self.gene_ids_,
        )

        self.pca_ = PCA(
            n_components=self.n_components,
            svd_solver="arpack",
            whiten=False,
            random_state=self.seed,
        )
        self.pca_.fit(X_train)
        return self

    def transform(self, data: ad.AnnData) -> CellEmbeddings:
        if self.pca_ is None or self.gene_ids_ is None:
            raise RuntimeError("PCA representation has not been fitted")

        X = self._get_expression_matrix(
            data,
            gene_ids=self.gene_ids_,
        )

        matrix = self.pca_.transform(X).astype(np.float32)

        return CellEmbeddings(
            matrix=matrix,
            cell_ids=data.obs_names.copy(),
        )

    @staticmethod
    def _select_train_hvgs(train: ad.AnnData, n_hvgs: int) -> pd.Index:
        usable = ~train.var["feature_is_filtered"].to_numpy()
        usable_genes = train.var_names[usable]

        hvg_stats = sc.pp.highly_variable_genes(
            train[:, usable_genes],
            n_top_genes=n_hvgs,
            flavor="seurat",
            subset=False,
            inplace=False,
        )

        return hvg_stats.index[hvg_stats["highly_variable"]].copy()

    @staticmethod
    def _get_expression_matrix(
        data: ad.AnnData,
        gene_ids: pd.Index,
    ) -> Any:
        missing = gene_ids.difference(data.var_names)
        if not missing.empty:
            raise ValueError(f"Missing {len(missing)} PCA genes")

        matrix = data[:, gene_ids].X

        if hasattr(matrix, "to_memory"):
            matrix = matrix.to_memory()

        if sparse.issparse(matrix):
            return matrix.tocsr().astype(np.float32)

        return np.asarray(matrix, dtype=np.float32)

    def __str__(self) -> str:
        return (
            f"PCARepresentation(n_hvgs={self.n_hvgs}, "
            f"n_components={self.n_components}, seed={self.seed}, "
            f"fitted={self.pca_ is not None})"
        )
