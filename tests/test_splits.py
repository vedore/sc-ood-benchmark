import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from preprocessor.splits import assign_donor_splits, create_split_views


class AssignDonorSplitsTest(unittest.TestCase):
    def setUp(self) -> None:
        rows = [
            {
                "cell_id": f"cell-{donor}-{cell}",
                "donor_id": donor,
                "institute": "A",
            }
            for donor in range(10)
            for cell in range(2)
        ]
        rows.extend(
            [
                {
                    "cell_id": "shared-a",
                    "donor_id": "shared",
                    "institute": "A",
                },
                {
                    "cell_id": "shared-b",
                    "donor_id": "shared",
                    "institute": "B",
                },
                {
                    "cell_id": "other",
                    "donor_id": "other",
                    "institute": "B",
                },
            ]
        )
        self.manifest = pd.DataFrame(rows)

    def test_split_is_reproducible_and_donor_grouped(self) -> None:
        first = assign_donor_splits(self.manifest, seed=7, institute="A")
        second = assign_donor_splits(self.manifest, seed=7, institute="A")
        assert_frame_equal(first, second)

        used = first[first["split"] != "excluded"]
        self.assertEqual(set(used["split"]), {"train", "dev", "test"})
        self.assertEqual(used.groupby("donor_id")["split"].nunique().max(), 1)

    def test_multisite_and_other_site_donors_are_excluded(self) -> None:
        result = assign_donor_splits(self.manifest, institute="A")
        excluded = result[result["donor_id"].isin(["shared", "other"])]
        self.assertEqual(set(excluded["split"]), {"excluded"})

    def test_required_columns_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "institute"):
            assign_donor_splits(self.manifest.drop(columns="institute"))


class CreateSplitViewsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adata = ad.AnnData(
            X=np.arange(12).reshape(6, 2),
            obs=pd.DataFrame(index=[f"cell-{index}" for index in range(6)]),
        )
        self.split_manifest = pd.DataFrame(
            {
                "cell_id": self.adata.obs_names,
                "split": ["train", "train", "dev", "dev", "test", "test"],
            }
        )

    def _save_manifest(self, directory: str) -> Path:
        split_file = Path(directory) / "split.csv.gz"
        self.split_manifest.to_csv(
            split_file,
            index=False,
            compression="gzip",
        )
        return split_file

    def test_views_follow_manifest_cell_ids(self) -> None:
        with TemporaryDirectory() as directory:
            splits = create_split_views(
                self.adata,
                self._save_manifest(directory),
            )

        self.assertEqual(tuple(splits), ("train", "dev", "test"))
        self.assertEqual(splits["train"].obs_names.tolist(), ["cell-0", "cell-1"])
        self.assertEqual(splits["dev"].obs_names.tolist(), ["cell-2", "cell-3"])
        self.assertEqual(splits["test"].obs_names.tolist(), ["cell-4", "cell-5"])

    def test_cell_id_coverage_is_validated(self) -> None:
        self.split_manifest.loc[5, "cell_id"] = "unknown-cell"
        with TemporaryDirectory() as directory:
            split_file = self._save_manifest(directory)
            with self.assertRaisesRegex(ValueError, "missing=1, extra=1"):
                create_split_views(self.adata, split_file)


if __name__ == "__main__":
    unittest.main()
