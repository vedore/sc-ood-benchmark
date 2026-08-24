import unittest

from metrics import compute_classification_metrics


class ClassificationMetricsTest(unittest.TestCase):
    def test_classification_metrics(self) -> None:
        result = compute_classification_metrics(
            ["a", "a", "b", "b"],
            ["a", "b", "b", "b"],
        )

        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], 11 / 15)

    def test_rejects_different_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            compute_classification_metrics(["a"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
