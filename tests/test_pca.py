import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from representations.pca import PCARepresentation


class PCARepresentationTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        counts = rng.poisson(1.5, size=(60, 20)).astype(np.float32)
        normalized = np.log1p(
            counts / counts.sum(axis=1, keepdims=True) * 10_000
        )
        self.adata = ad.AnnData(
            X=sparse.csr_matrix(normalized),
            obs=pd.DataFrame(index=[f"cell-{index}" for index in range(60)]),
            var=pd.DataFrame(
                {"feature_is_filtered": [False] * 19 + [True]},
                index=[f"gene-{index}" for index in range(20)],
            ),
        )

    def test_fit_and_transform_backed_data(self) -> None:
        with TemporaryDirectory() as directory:
            data_file = Path(directory) / "data.h5ad"
            self.adata.write_h5ad(data_file)
            backed = ad.read_h5ad(data_file, backed="r")
            try:
                representation = PCARepresentation(n_hvgs=10, n_components=3)
                self.assertIn("fitted=False", str(representation))

                representation.fit(backed)
                embeddings = representation.transform(backed)

                self.assertEqual(embeddings.matrix.shape, (60, 3))
                self.assertTrue(embeddings.cell_ids.equals(backed.obs_names))
                self.assertIn("fitted=True", str(representation))
                self.assertNotIn("gene-19", representation.gene_ids_)
            finally:
                backed.file.close()

    def test_transform_requires_fit(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "has not been fitted"):
            PCARepresentation().transform(self.adata)


if __name__ == "__main__":
    unittest.main()
