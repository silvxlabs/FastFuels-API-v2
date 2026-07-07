"""Unit tests for lib.gcs.blobs helpers."""

from unittest.mock import MagicMock, patch

from lib.gcs.blobs import storage_size


@patch("lib.gcs.blobs.get_gcsfs_client")
def test_storage_size_sums_prefix_after_invalidating_cache(mock_get_client):
    """storage_size drops the stale listing cache, then sums the whole prefix.

    Invalidating first is what makes it see objects just written; du(total=True)
    sums every object under the path, so it works for a multi-object store.
    """
    fs = MagicMock()
    fs.du.return_value = 4096
    mock_get_client.return_value = fs

    assert storage_size("gs://grids-v2/grid-123") == 4096

    # gs:// prefix stripped before touching the filesystem.
    fs.invalidate_cache.assert_called_once_with("grids-v2/grid-123")
    fs.du.assert_called_once_with("grids-v2/grid-123", total=True)


@patch("lib.gcs.blobs.get_gcsfs_client")
def test_storage_size_accepts_bare_path_and_single_object(mock_get_client):
    """A bare bucket/key path (single object) is passed through unchanged."""
    fs = MagicMock()
    fs.du.return_value = 10
    mock_get_client.return_value = fs

    assert storage_size("pointclouds-v2/pc-1/cloud.laz") == 10

    fs.du.assert_called_once_with("pointclouds-v2/pc-1/cloud.laz", total=True)
