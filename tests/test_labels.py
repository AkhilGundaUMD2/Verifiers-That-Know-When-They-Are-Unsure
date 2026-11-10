import unittest

from baselines.labels import label_rows


class LabelsTests(unittest.TestCase):
    def test_label_rows_keeps_only_id_and_labels(self):
        records = [{"id": "a", "labels": {"success": True}, "steps": [{"axtree": "big"}]}]
        self.assertEqual(list(label_rows(records)),
                         [{"id": "a", "labels": {"success": True}}])


if __name__ == "__main__":
    unittest.main()
