"""
Unit tests for api.db.blobs module.

Tests GCS blob operations with mocked gcsfs client.
"""

from unittest.mock import AsyncMock, patch

import pytest
from api.db import blobs

pytestmark = pytest.mark.anyio


@pytest.fixture
def mock_gcsfs():
    mock = AsyncMock()
    with patch.object(blobs, "gcsfs_client", mock):
        yield mock


class TestConstants:
    def test_grids_bucket_from_config(self):
        from lib.config import GRIDS_BUCKET

        assert blobs.GRIDS_BUCKET == GRIDS_BUCKET

    def test_exports_bucket_from_config(self):
        from lib.config import EXPORTS_BUCKET

        assert blobs.EXPORTS_BUCKET == EXPORTS_BUCKET


class TestDeleteDirectory:
    async def test_deletes_existing_directory(self, mock_gcsfs):
        mock_gcsfs._exists.return_value = True

        await blobs.delete_directory("my-bucket", "some/dir")

        mock_gcsfs._exists.assert_awaited_once_with("my-bucket/some/dir")
        mock_gcsfs._rm.assert_awaited_once_with("my-bucket/some/dir", recursive=True)

    async def test_skips_nonexistent_directory(self, mock_gcsfs):
        mock_gcsfs._exists.return_value = False

        await blobs.delete_directory("my-bucket", "missing/dir")

        mock_gcsfs._exists.assert_awaited_once_with("my-bucket/missing/dir")
        mock_gcsfs._rm.assert_not_awaited()

    async def test_constructs_full_path(self, mock_gcsfs):
        mock_gcsfs._exists.return_value = False

        await blobs.delete_directory("bucket_name", "directory_path")

        mock_gcsfs._exists.assert_awaited_once_with("bucket_name/directory_path")


class TestDeleteFile:
    async def test_deletes_file(self, mock_gcsfs):
        await blobs.delete_file("my-bucket", "path/to/file.txt")

        mock_gcsfs._rm.assert_awaited_once_with("my-bucket/path/to/file.txt")


class TestCheckExists:
    async def test_returns_true_when_exists(self, mock_gcsfs):
        mock_gcsfs._exists.return_value = True

        result = await blobs.check_exists("my-bucket", "existing/path")

        assert result is True
        mock_gcsfs._exists.assert_awaited_once_with("my-bucket/existing/path")

    async def test_returns_false_when_not_exists(self, mock_gcsfs):
        mock_gcsfs._exists.return_value = False

        result = await blobs.check_exists("my-bucket", "missing/path")

        assert result is False


class TestDownloadFile:
    async def test_downloads_file(self, mock_gcsfs):
        await blobs.download_file("bucket/path/file.txt", "/local/file.txt")

        mock_gcsfs._get.assert_awaited_once_with(
            "bucket/path/file.txt", "/local/file.txt"
        )

    async def test_strips_gs_prefix(self, mock_gcsfs):
        await blobs.download_file("gs://bucket/path/file.txt", "/local/file.txt")

        mock_gcsfs._get.assert_awaited_once_with(
            "bucket/path/file.txt", "/local/file.txt"
        )

    async def test_no_strip_without_gs_prefix(self, mock_gcsfs):
        await blobs.download_file("bucket/path/file.txt", "/local/file.txt")

        mock_gcsfs._get.assert_awaited_once_with(
            "bucket/path/file.txt", "/local/file.txt"
        )


class TestDeleteDirectorySafe:
    async def test_successful_delete(self, mock_gcsfs):
        mock_gcsfs._exists.return_value = True

        await blobs.delete_directory_safe("my-bucket", "some/dir")

        mock_gcsfs._rm.assert_awaited_once_with("my-bucket/some/dir", recursive=True)

    async def test_swallows_file_not_found(self, mock_gcsfs):
        mock_gcsfs._exists.side_effect = FileNotFoundError("gone")

        await blobs.delete_directory_safe("my-bucket", "missing/dir")

    async def test_logs_warning_on_other_exception(self, mock_gcsfs, caplog):
        mock_gcsfs._exists.side_effect = RuntimeError("network error")

        with caplog.at_level("WARNING"):
            await blobs.delete_directory_safe("my-bucket", "bad/dir")

        assert "Failed to delete GCS data" in caplog.text
        assert "network error" in caplog.text

    async def test_passes_args_correctly(self, mock_gcsfs):
        mock_gcsfs._exists.return_value = True

        await blobs.delete_directory_safe("specific-bucket", "specific/path")

        mock_gcsfs._exists.assert_awaited_once_with("specific-bucket/specific/path")
        mock_gcsfs._rm.assert_awaited_once_with(
            "specific-bucket/specific/path", recursive=True
        )
