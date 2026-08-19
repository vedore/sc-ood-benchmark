import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from preprocessor.splits import assign_donor_splits


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


if __name__ == "__main__":
    unittest.main()
