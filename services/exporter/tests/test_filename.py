"""Tests for filename sanitization."""

from exporter.filename import sanitize_filename


class TestSanitizeFilename:
    def test_spaces_replaced(self):
        assert sanitize_filename("My Export", ".tif") == "My_Export.tif"

    def test_special_chars_replaced_and_collapsed(self):
        assert sanitize_filename("my export (v2)!", ".tif") == "my_export_v2.tif"

    def test_empty_string_falls_back(self):
        assert sanitize_filename("", ".tif") == "export.tif"

    def test_all_special_chars_falls_back(self):
        assert sanitize_filename("!!!", ".tif") == "export.tif"

    def test_whitespace_stripped(self):
        assert sanitize_filename("  spaced  ", ".tif") == "spaced.tif"

    def test_hyphens_preserved(self):
        assert sanitize_filename("a---b", ".tif") == "a---b.tif"

    def test_truncated_to_200_chars(self):
        result = sanitize_filename("a" * 300, ".tif")
        assert result == "a" * 200 + ".tif"

    def test_unicode_replaced(self):
        assert sanitize_filename("café", ".tif") == "caf.tif"
