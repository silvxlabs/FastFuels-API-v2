"""
Unit tests for api/v2/resources/features/cache.py

Pure unit tests with no live server or GCS access. They pin the storage
access pattern that keeps the feature data endpoints off the grpcio poller
race (#265): Parquet must be opened via pyarrow's native ``gs://`` resolution,
never a synchronous ``gcsfs``/``fsspec`` filesystem whose daemon-thread event
loop floods the logs with ``BlockingIOError`` on every subsequent Firestore
RPC.
"""

import inspect

from api.resources.features import cache as features_cache


def test_open_parquet_file_uses_native_gcs_uri_without_fsspec(monkeypatch):
    """Regression for #265.

    ``_open_parquet_file`` must hand pyarrow a ``gs://`` URI and no
    ``filesystem`` kwarg, so pyarrow resolves its native (Arrow C++) GCS
    filesystem. Passing a sync ``gcsfs.GCSFileSystem`` here is what
    reintroduced the poller flood.
    """
    captured = {}

    class _FakeParquetFile:
        def __init__(self, source, *args, **kwargs):
            captured["source"] = source
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr(features_cache.pq, "ParquetFile", _FakeParquetFile)

    features_cache._open_parquet_file.cache_clear()
    try:
        features_cache._open_parquet_file("test-domain", "test-feature")
    finally:
        features_cache._open_parquet_file.cache_clear()

    assert captured["source"].startswith("gs://")
    assert captured["args"] == ()
    assert "filesystem" not in captured["kwargs"]


def test_cache_module_does_not_reference_sync_fsspec_client():
    """The feature data path must not import or construct the sync fsspec
    GCS client (``lib.gcs.get_gcsfs_client`` / ``gcsfs.GCSFileSystem``)."""
    source = inspect.getsource(features_cache)
    assert "get_gcsfs_client" not in source
    assert "GCSFileSystem" not in source
