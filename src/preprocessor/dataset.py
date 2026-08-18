from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd


class DataSet:
    def __init__(self, filepath: str | Path) -> None:
        self.data = self.init_dataset(Path(filepath))

    def init_dataset(self, filepath: Path) -> None:
        return ad.read_h5ad(filepath, backed="r")

    def X(self):
        pass

    def Y(self):
        pass


MANIFEST_OBS = [
    "donor_id",
    "sample_id",
    "library_id",
    "institute",
    "Country",
    "self_reported_ethnicity",
    "assay",
    "tissue",
    "disease",
    "author_cell_type",
    "Annotation_Level1",
    "Annotation_Level4",
    "cell_type",
    "cell_type_ontology_term_id",
]


class Manifest:
    def __init__(self, dataset: DataSet | None = None) -> None:
        self.dataset_obj = dataset
        self.manifest: pd.DataFrame | None = None

    def init(self) -> None:
        if self.dataset_obj is None:
            raise ValueError("A dataset is required to initialize a manifest")

        data = self.dataset_obj.data
        manifest = data.obs[MANIFEST_OBS].copy()
        manifest.insert(0, "cell_id", data.obs_names)
        self.manifest = manifest

    def save(self, filepath: str | Path) -> None:
        if self.manifest is None:
            raise ValueError("Manifest has not been initialized or loaded")

        self.manifest.to_csv(Path(filepath), index=False, compression="gzip")

    @classmethod
    def load(cls, filepath: str | Path) -> Manifest:
        manifest = cls()
        manifest.manifest = pd.read_csv(Path(filepath), compression="gzip")
        return manifest
