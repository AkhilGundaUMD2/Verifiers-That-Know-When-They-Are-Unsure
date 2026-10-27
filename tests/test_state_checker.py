import unittest

from baselines.state_checker import (
    check_file_exists,
    check_file_changed,
    check_file_unchanged,
    check_file_absent,
)


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


class TestFileChanged(unittest.TestCase):
    def test_different_hashes_pass(self):
        result = check_file_changed("abc123", "xyz789")
        self.assertEqual(result["status"], "pass")

    def test_identical_hashes_fail(self):
        result = check_file_changed("abc123", "abc123")
        self.assertEqual(result["status"], "fail")

    def test_missing_before_hash_is_unknown(self):
        result = check_file_changed(None, "xyz789")
        self.assertEqual(result["status"], "unknown")

    def test_missing_after_hash_is_unknown(self):
        result = check_file_changed("abc123", None)
        self.assertEqual(result["status"], "unknown")

    def test_both_hashes_missing_is_unknown(self):
        result = check_file_changed(None, None)
        self.assertEqual(result["status"], "unknown")


class TestFileUnchanged(unittest.TestCase):
    def test_identical_hashes_pass(self):
        result = check_file_unchanged("abc123", "abc123")
        self.assertEqual(result["status"], "pass")

    def test_different_hashes_fail(self):
        result = check_file_unchanged("abc123", "xyz789")
        self.assertEqual(result["status"], "fail")

    def test_missing_before_hash_is_unknown(self):
        result = check_file_unchanged(None, "abc123")
        self.assertEqual(result["status"], "unknown")

    def test_missing_after_hash_is_unknown(self):
        result = check_file_unchanged("abc123", None)
        self.assertEqual(result["status"], "unknown")

    def test_both_hashes_missing_is_unknown(self):
        result = check_file_unchanged(None, None)
        self.assertEqual(result["status"], "unknown")


class TestFileAbsent(unittest.TestCase):
    def test_present_file_fails_with_complete_listing(self):
        result = check_file_absent(
            "/workspace/temp.txt",
            {"/workspace/temp.txt"},
            listing_complete=True,
        )
        self.assertEqual(result["status"], "fail")

    def test_present_file_fails_with_incomplete_listing(self):
        result = check_file_absent(
            "/workspace/temp.txt",
            {"/workspace/temp.txt"},
            listing_complete=False,
        )
        self.assertEqual(result["status"], "fail")

    def test_absent_file_passes_with_complete_listing(self):
        result = check_file_absent(
            "/workspace/temp.txt",
            {"/workspace/report.pdf"},
            listing_complete=True,
        )
        self.assertEqual(result["status"], "pass")

    def test_absent_file_is_unknown_with_incomplete_listing(self):
        result = check_file_absent(
            "/workspace/temp.txt",
            {"/workspace/report.pdf"},
            listing_complete=False,
        )
        self.assertEqual(result["status"], "unknown")

    def test_unavailable_listing_is_unknown(self):
        result = check_file_absent("/workspace/temp.txt", None)
        self.assertEqual(result["status"], "unknown")


if __name__ == "__main__":
    unittest.main()