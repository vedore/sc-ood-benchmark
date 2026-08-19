from __future__ import annotations

from pathlib import Path

import pandas as pd

from preprocessor.dataset import DataSet


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

        filepath = Path(filepath)
        if not filepath.name.endswith(".csv.gz"):
            raise ValueError("Compressed manifest path must end with '.csv.gz'")

        self.manifest.to_csv(filepath, index=False, compression="gzip")

    @classmethod
    def load(cls, filepath: str | Path) -> Manifest:
        manifest = cls()
        manifest.manifest = pd.read_csv(Path(filepath), compression="gzip")
        return manifest
