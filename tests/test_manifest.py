import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd

from preprocessor.dataset import DataSet
from preprocessor.manifest import MANIFEST_OBS, Manifest


class ManifestTest(unittest.TestCase):
    def test_manifest_preserves_dataset_column_names(self) -> None:
        obs = pd.DataFrame(
            {column: [f"{column}-value"] for column in MANIFEST_OBS},
            index=["cell-1"],
        )
        dataset = DataSet.__new__(DataSet)
        dataset.data = ad.AnnData(X=np.zeros((1, 1)), obs=obs)

        manifest = Manifest(dataset)
        manifest.init()

        self.assertEqual(
            manifest.manifest.columns.tolist(),
            ["cell_id", *MANIFEST_OBS],
        )

        with TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.csv.gz"
            manifest.save(output)
            loaded = Manifest.load(output)
            self.assertEqual(
                loaded.manifest.columns.tolist(),
                ["cell_id", *MANIFEST_OBS],
            )

            with self.assertRaisesRegex(ValueError, "csv.gz"):
                manifest.save(Path(directory) / "manifest.csv.gzip")


if __name__ == "__main__":
    unittest.main()
