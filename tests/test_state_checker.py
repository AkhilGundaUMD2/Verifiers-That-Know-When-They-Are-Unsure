import unittest

from baselines.state_checker import check_file_exists


class TestFileExists(unittest.TestCase):
    def test_present_file_passes(self):
        result = check_file_exists(
            "/workspace/report.pdf",
            {"/workspace/report.pdf"},
            listing_complete=True,
        )
        self.assertEqual(result["status"], "pass")

    def test_missing_file_fails_with_complete_listing(self):
        result = check_file_exists(
            "/workspace/report.pdf",
            {"/workspace/notes.txt"},
            listing_complete=True,
        )
        self.assertEqual(result["status"], "fail")

    def test_missing_file_is_unknown_with_incomplete_listing(self):
        result = check_file_exists(
            "/workspace/report.pdf",
            {"/workspace/notes.txt"},
            listing_complete=False,
        )
        self.assertEqual(result["status"], "unknown")

    def test_unavailable_listing_is_unknown(self):
        result = check_file_exists("/workspace/report.pdf", None)
        self.assertEqual(result["status"], "unknown")

    def test_present_file_passes_with_incomplete_listing(self):
        result = check_file_exists(
            "/workspace/report.pdf",
            {"/workspace/report.pdf"},
        )
        self.assertEqual(result["status"], "pass")