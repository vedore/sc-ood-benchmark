import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from classifiers.knn import KNNClassifier


class KNNClassifierTest(unittest.TestCase):
    def test_fit_and_predict(self) -> None:
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array(["a", "a", "b", "b"])

        classifier = KNNClassifier({"n_neighbors": 1}).fit(X, y)
        predictions = classifier.predict(X)

        np.testing.assert_array_equal(predictions, y)

    def test_rejects_unknown_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown KNN"):
            KNNClassifier({"invalid": True})

    def test_save_and_load_round_trip(self) -> None:
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array(["a", "a", "b", "b"])

        with TemporaryDirectory() as directory:
            model_file = Path(directory) / "knn.pkl"
            classifier = KNNClassifier({"n_neighbors": 1}).fit(X, y)
            classifier.save(model_file)
            loaded = KNNClassifier.load(model_file)

            np.testing.assert_array_equal(loaded.predict(X), y)

    def test_save_requires_fit(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "has not been fitted"):
                KNNClassifier().save(Path(directory) / "knn.pkl")


if __name__ == "__main__":
    unittest.main()
