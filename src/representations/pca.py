from __future__ import annotations

import pickle
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA, IncrementalPCA

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
        self,
        n_hvgs: int = 2000,
        n_components: int = 20,
        seed: int = 42,
        batch_size: int | None = 4096,
        hvg_cache_file: str | Path | None = None,
        incremental: bool = False,
    ) -> None:
        if n_hvgs < 1 or n_components < 1:
            raise ValueError("n_hvgs and n_components must be positive")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive or None")

        self.n_hvgs = n_hvgs
        self.n_components = n_components
        self.seed = seed
        self.batch_size = batch_size
        self.hvg_cache_file = (
            Path(hvg_cache_file) if hvg_cache_file is not None else None
        )
        self.incremental = incremental

        self.gene_ids_: pd.Index | None = None
        self.pca_: PCA | IncrementalPCA | None = None

    def __str__(self) -> str:
        mode = "incremental" if self.incremental else "exact"
        return (
            f"PCARepresentation(n_hvgs={self.n_hvgs}, "
            f"n_components={self.n_components}, seed={self.seed}, "
            f"batch_size={self.batch_size}, mode={mode!r}, "
            f"fitted={self.pca_ is not None})"
        )

    def save(self, path: str | Path) -> None:
        """Save the fitted PCA model, selected genes, and transform settings."""
        if self.pca_ is None or self.gene_ids_ is None:
            raise RuntimeError("PCA representation has not been fitted")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "representation": "PCARepresentation",
            "parameters": {
                "n_hvgs": self.n_hvgs,
                "n_components": self.n_components,
                "seed": self.seed,
                "batch_size": self.batch_size,
                "incremental": self.incremental,
            },
            "gene_ids": self.gene_ids_.tolist(),
            "pca": self.pca_,
        }
        with path.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> PCARepresentation:
        """Load a fitted PCA representation from a trusted local file."""
        path = Path(path)
        with path.open("rb") as file:
            payload = pickle.load(file)

        if not isinstance(payload, dict):
            raise TypeError("Invalid PCA representation file")
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported PCA representation format")
        if payload.get("representation") != "PCARepresentation":
            raise ValueError("File does not contain a PCA representation")

        parameters = payload.get("parameters")
        model = payload.get("pca")
        gene_ids = pd.Index(payload.get("gene_ids", []), dtype="object")
        if not isinstance(parameters, dict):
            raise TypeError("PCA representation parameters are missing")
        if not isinstance(model, (PCA, IncrementalPCA)):
            raise TypeError("PCA representation model is missing")
        if gene_ids.empty or not gene_ids.is_unique:
            raise ValueError("PCA representation genes are invalid")
        if getattr(model, "n_features_in_", None) != len(gene_ids):
            raise ValueError("PCA model and selected genes do not match")

        representation = cls(**parameters)
        representation.gene_ids_ = gene_ids
        representation.pca_ = model
        return representation

    def fit(self, train: ad.AnnData) -> PCARepresentation:
        self.gene_ids_ = self._load_or_select_train_hvgs(train)

        if self.incremental:
            model = IncrementalPCA(
                n_components=self.n_components, batch_size=self.batch_size
            )
            for start, end in self._batch_slices(
                train.n_obs, minimum_size=self.n_components
            ):
                batch = self._get_expression_matrix(
                    train,
                    self.gene_ids_,
                    row_slice=slice(start, end),
                )
                model.partial_fit(self._as_dense(batch))
            self.pca_ = model
            return self

        matrix = self._get_expression_matrix(train, self.gene_ids_)
        model = PCA(
            n_components=self.n_components,
            svd_solver="arpack",
            whiten=False,
            random_state=self.seed,
        )
        model.fit(matrix)
        self.pca_ = model
        return self

    def transform(self, data: ad.AnnData) -> CellEmbeddings:
        matrix = np.empty((data.n_obs, self.n_components), dtype=np.float32)
        self._transform_into(data, matrix)
        return CellEmbeddings(matrix=matrix, cell_ids=data.obs_names.copy())

    def _transform_into(self, data: ad.AnnData, output: np.ndarray) -> None:
        if self.pca_ is None or self.gene_ids_ is None:
            raise RuntimeError("PCA representation has not been fitted")
        if output.shape != (data.n_obs, self.n_components):
            raise ValueError("Output matrix has the wrong shape")

        for start, end in self._batch_slices(data.n_obs):
            batch = self._get_expression_matrix(
                data,
                self.gene_ids_,
                row_slice=slice(start, end),
            )
            if isinstance(self.pca_, IncrementalPCA):
                batch = self._as_dense(batch)
            output[start:end] = self.pca_.transform(batch).astype(
                np.float32, copy=False
            )

    def transform_to_file(
        self,
        data: ad.AnnData,
        matrix_file: str | Path,
        cell_ids_file: str | Path,
    ) -> None:
        """Transform data and stream its matrix and cell IDs to separate files."""
        matrix_file = Path(matrix_file)
        cell_ids_file = Path(cell_ids_file)
        matrix_file.parent.mkdir(parents=True, exist_ok=True)
        cell_ids_file.parent.mkdir(parents=True, exist_ok=True)

        matrix = np.lib.format.open_memmap(
            matrix_file,
            mode="w+",
            dtype=np.float32,
            shape=(data.n_obs, self.n_components),
        )

        self._transform_into(data, matrix)
        matrix.flush()
        pd.DataFrame({"cell_id": data.obs_names}).to_csv(
            cell_ids_file, index=False, compression="gzip"
        )

    def _load_or_select_train_hvgs(self, train: ad.AnnData) -> pd.Index:
        cache_file = self.hvg_cache_file
        if cache_file is not None and cache_file.exists():
            cache = pd.read_csv(cache_file, compression="infer")
            self._validate_hvg_cache(cache, train)
            return pd.Index(cache["gene_id"].astype(str), dtype="object")

        gene_ids = self._select_train_hvgs(train)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "gene_id": gene_ids,
                    "requested_n_hvgs": self.n_hvgs,
                    "train_n_obs": train.n_obs,
                    "train_cell_hash": self._hash_index(train.obs_names),
                    "gene_universe_hash": self._hash_index(train.var_names),
                }
            ).to_csv(cache_file, index=False, compression="gzip")
        return gene_ids

    def _validate_hvg_cache(self, cache: pd.DataFrame, train: ad.AnnData) -> None:
        required = {
            "gene_id",
            "requested_n_hvgs",
            "train_n_obs",
            "train_cell_hash",
            "gene_universe_hash",
        }
        missing_columns = required.difference(cache.columns)
        if missing_columns:
            raise ValueError(
                "HVG cache is missing columns: " + ", ".join(sorted(missing_columns))
            )
        if cache.empty:
            raise ValueError("HVG cache is empty")

        expected = {
            "requested_n_hvgs": str(self.n_hvgs),
            "train_n_obs": str(train.n_obs),
            "train_cell_hash": self._hash_index(train.obs_names),
            "gene_universe_hash": self._hash_index(train.var_names),
        }
        for column, value in expected.items():
            values = cache[column].astype(str).unique()
            if len(values) != 1 or values[0] != value:
                raise ValueError(f"HVG cache does not match train data: {column}")

        gene_ids = pd.Index(cache["gene_id"].astype(str))
        if not gene_ids.is_unique:
            raise ValueError("HVG cache contains duplicate genes")
        missing_genes = gene_ids.difference(train.var_names)
        if not missing_genes.empty:
            raise ValueError(f"HVG cache contains {len(missing_genes)} missing genes")

    def _select_train_hvgs(self, train: ad.AnnData) -> pd.Index:
        if "feature_is_filtered" not in train.var:
            raise ValueError("Missing var column: 'feature_is_filtered'")

        usable = ~train.var["feature_is_filtered"].to_numpy()
        usable_genes = train.var_names[usable]
        if usable_genes.empty:
            raise ValueError("No usable genes remain for HVG selection")

        hvg_data = ad.AnnData(
            X=self._get_expression_matrix(train, usable_genes),
            var=train.var.loc[usable_genes].copy(),
        )
        hvg_stats = sc.pp.highly_variable_genes(
            hvg_data,
            n_top_genes=self.n_hvgs,
            flavor="seurat",
            subset=False,
            inplace=False,
        )

        return hvg_stats.index[hvg_stats["highly_variable"]].copy()

    def _batch_slices(
        self, n_rows: int, minimum_size: int = 1
    ) -> Iterator[tuple[int, int]]:
        if n_rows < minimum_size:
            raise ValueError(f"Need at least {minimum_size} rows, received {n_rows}")
        batch_size = (
            n_rows if self.batch_size is None else max(self.batch_size, minimum_size)
        )
        start = 0
        while start < n_rows:
            end = min(start + batch_size, n_rows)
            if 0 < n_rows - end < minimum_size:
                end = n_rows
            yield start, end
            start = end

    @staticmethod
    def _get_expression_matrix(
        data: ad.AnnData,
        gene_ids: pd.Index,
        row_slice: slice = slice(None),
    ) -> Any:
        missing = gene_ids.difference(data.var_names)
        if not missing.empty:
            raise ValueError(f"Missing {len(missing)} PCA genes")

        if data.isbacked and data.is_view:
            # Backed AnnData forbids view-of-view indexing. Resolve the stable
            # cell and gene IDs against the original object in one operation.
            source = data._adata_ref
            matrix = source[data.obs_names[row_slice], gene_ids].X
        else:
            matrix = data[row_slice, gene_ids].X
        if hasattr(matrix, "to_memory"):
            matrix = matrix.to_memory()
        if sparse.issparse(matrix):
            return matrix.tocsr().astype(np.float32, copy=False)
        return np.asarray(matrix, dtype=np.float32)

    @staticmethod
    def _as_dense(matrix: Any) -> np.ndarray:
        if sparse.issparse(matrix):
            return matrix.toarray()
        return np.asarray(matrix, dtype=np.float32)

    @staticmethod
    def _hash_index(index: pd.Index) -> str:
        digest = sha256()
        for value in index:
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()
