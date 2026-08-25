import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from preprocessor.standard_scaler import StandardScaler


class StandardScalerTest(unittest.TestCase):
    def test_fit_transform_zero_means_and_unit_variance(self) -> None:
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        transformed = StandardScaler().fit_transform(X)

        np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(transformed.std(axis=0), 1, atol=1e-10)

    def test_transform_uses_fit_statistics_on_new_data(self) -> None:
        train = np.array([[0.0], [10.0]])
        scaler = StandardScaler().fit(train)

        transformed = scaler.transform(np.array([[20.0]]))
        np.testing.assert_allclose(transformed, [[3.0]])

    def test_save_and_load_round_trip(self) -> None:
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        scaler = StandardScaler().fit(X)

        with TemporaryDirectory() as directory:
            scaler_file = Path(directory) / "scaler.pkl"
            scaler.save(scaler_file)
            loaded = StandardScaler.load(scaler_file)

            np.testing.assert_allclose(loaded.transform(X), scaler.transform(X))

    def test_save_requires_fit(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "has not been fitted"):
                StandardScaler().save(Path(directory) / "scaler.pkl")


if __name__ == "__main__":
    unittest.main()
