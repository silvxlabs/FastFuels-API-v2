"""Unit tests for lib.gcs.blobs helpers."""

import io
from unittest.mock import MagicMock, patch

from lib.gcs.blobs import _UPLOAD_SLICE_BYTES, storage_size, upload_buffer


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


class _RecordingHandle:
    """Fake write handle that copies each write out of the caller's buffer.

    Deliberately not a MagicMock: a mock retains its call arguments, and the
    arguments here are memoryview slices of the source buffer. Holding one
    keeps the buffer exported, which is exactly what the helper is careful not
    to do.
    """

    def __init__(self):
        self.writes: list[bytes] = []

    def write(self, data):
        self.writes.append(bytes(data))
        return len(data)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture_writes(mock_get_client):
    """Point a patched client at a recording handle, returning it and the fs."""
    handle = _RecordingHandle()
    fs = MagicMock()
    fs.open.return_value = handle
    mock_get_client.return_value = fs
    return fs, handle.writes


@patch("lib.gcs.blobs.get_gcsfs_client")
def test_upload_buffer_writes_every_byte_in_order(mock_get_client):
    fs, writes = _capture_writes(mock_get_client)
    payload = bytes(range(256)) * 1024

    upload_buffer("gs://pointclouds-v2/pc-1/cloud.laz", io.BytesIO(payload))

    fs.open.assert_called_once_with("pointclouds-v2/pc-1/cloud.laz", "wb")
    assert b"".join(writes) == payload


@patch("lib.gcs.blobs.get_gcsfs_client")
def test_upload_buffer_never_hands_over_the_whole_payload_at_once(mock_get_client):
    """The bound is the point: one big write stages a second full copy.

    fsspec's AbstractBufferedFile.write appends to its own buffer and only
    flushes afterwards, once that buffer reaches the block size. Writing a
    gigabyte-scale LAZ in one call therefore doubles peak memory before a byte
    is uploaded, which is the difference between fitting in the worker's memory
    budget and not.
    """
    _, writes = _capture_writes(mock_get_client)
    payload = b"\0" * (_UPLOAD_SLICE_BYTES * 2 + 7)

    upload_buffer("pointclouds-v2/pc-1/cloud.laz", io.BytesIO(payload))

    assert len(writes) == 3
    assert max(len(chunk) for chunk in writes) <= _UPLOAD_SLICE_BYTES
    assert b"".join(writes) == payload


@patch("lib.gcs.blobs.get_gcsfs_client")
def test_upload_buffer_reads_from_the_start_not_the_cursor(mock_get_client):
    """Writers leave the cursor at the end; the whole file still has to upload."""
    _, writes = _capture_writes(mock_get_client)
    buffer = io.BytesIO(b"abcdef")
    buffer.seek(0, io.SEEK_END)

    upload_buffer("pointclouds-v2/pc-1/cloud.laz", buffer)

    assert b"".join(writes) == b"abcdef"
