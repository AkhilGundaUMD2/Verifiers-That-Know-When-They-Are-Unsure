import json
import tempfile
import unittest
from pathlib import Path

from compare_outcomes import load_predictions, split_predictions


class CompareOutcomesTests(unittest.TestCase):
    def test_load_predictions_validates_all_axes_and_statuses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm.jsonl"
            path.write_text(
                json.dumps({
                    "id": "one",
                    "verifier": "llm",
                    "outcome": "pass",
                    "side_effect": "unknown",
                    "looping": "fail",
                }) + "\n",
                encoding="utf-8",
            )
            predictions = load_predictions(path)
            self.assertEqual(predictions["one"]["outcome"], "pass")

    def test_load_predictions_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.jsonl"
            row = {
                "id": "one",
                "verifier": "rules",
                "outcome": "fail",
                "side_effect": "pass",
                "looping": "pass",
            }
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
                load_predictions(path)

    def test_split_predictions_groups_and_rejects_missing_assignment(self):
        predictions = {
            "one": {"outcome": "pass"},
            "two": {"outcome": "fail"},
        }
        split = split_predictions(predictions, {"one": "id", "two": "ood"})
        self.assertEqual(set(split), {"id", "ood"})
        with self.assertRaisesRegex(ValueError, "absent from split"):
            split_predictions(predictions, {"one": "id"})


if __name__ == "__main__":
    unittest.main()
