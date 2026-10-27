import unittest

from baselines.score import score, score_axis


class ScoreTests(unittest.TestCase):
    def test_score_axis_counts_errors_reviews_and_unlabeled_records(self):
        result = score_axis("outcome", [
            (True, "pass"),
            (True, "fail"),
            (False, "pass"),
            (False, "unknown"),
            (None, "pass"),
        ])
        self.assertEqual(result["labeled"], 4)
        self.assertEqual(result["unlabeled"], 1)
        self.assertEqual(result["good"], 2)
        self.assertEqual(result["bad"], 2)
        self.assertEqual(result["review_rate"], 0.25)
        self.assertEqual(result["accuracy"], 1 / 3)
        self.assertEqual(result["false_accept_rate"], 0.5)
        self.assertEqual(result["false_reject_rate"], 0.5)

    def test_score_axis_treats_side_effect_as_bad_label(self):
        result = score_axis("side_effect", [(True, "pass"), (False, "fail")])
        self.assertEqual(result["good"], 1)
        self.assertEqual(result["bad"], 1)
        self.assertEqual(result["false_accept_rate"], 1.0)
        self.assertEqual(result["false_reject_rate"], 1.0)

    def test_score_axis_returns_none_for_undefined_rates(self):
        result = score_axis("outcome", [(None, "unknown")])
        self.assertIsNone(result["review_rate"])
        self.assertIsNone(result["accuracy"])
        self.assertIsNone(result["false_accept_rate"])
        self.assertIsNone(result["false_reject_rate"])

    def test_score_computes_every_axis_for_predictions(self):
        labels = {
            "one": {"success": True, "side_effect": False, "looping": False},
            "two": {"success": False, "side_effect": True, "looping": True},
        }
        predictions = {
            "one": {"outcome": "pass", "side_effect": "pass", "looping": "pass"},
            "two": {"outcome": "fail", "side_effect": "fail", "looping": "fail"},
        }
        result = score(labels, predictions)
        self.assertEqual(set(result), {"outcome", "side_effect", "looping"})
        self.assertTrue(all(axis["accuracy"] == 1 for axis in result.values()))


if __name__ == "__main__":
    unittest.main()
