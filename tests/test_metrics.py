import unittest

from metrics import compute_classification_metrics, summarize_donor_metrics


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


class SummarizeDonorMetricsTest(unittest.TestCase):
    def test_mean_and_std_across_donors(self) -> None:
        summary = summarize_donor_metrics(
            [
                {"macro_f1": 0.6, "accuracy": 0.8},
                {"macro_f1": 0.8, "accuracy": 0.9},
            ]
        )

        self.assertEqual(summary["n_donors"], 2)
        self.assertAlmostEqual(summary["macro_f1"], 0.7)
        self.assertAlmostEqual(summary["accuracy"], 0.85)
        self.assertGreater(summary["macro_f1_std"], 0)

    def test_std_is_nan_for_single_donor(self) -> None:
        summary = summarize_donor_metrics([{"macro_f1": 0.6, "accuracy": 0.8}])

        self.assertEqual(summary["n_donors"], 1)
        self.assertTrue(summary["macro_f1_std"] != summary["macro_f1_std"])

    def test_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one donor metric"):
            summarize_donor_metrics([])


if __name__ == "__main__":
    unittest.main()
