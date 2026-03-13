import sys

if sys.platform == "win32":
    import atexit
    import io
    import tempfile

    import geopandas as gpd
    import zarr.core.sync as _zarr_sync
    from google.cloud import storage

    _original_read_file = gpd.read_file
    _original_tmp = tempfile.NamedTemporaryFile

    def _patched_tmp(*args, **kwargs):
        kwargs.setdefault("delete", False)
        return _original_tmp(*args, **kwargs)

    tempfile.NamedTemporaryFile = _patched_tmp

    def _patched_read_file(path, **kwargs):
        if isinstance(path, str) and path.startswith("gs://"):
            # Use GCS client directly, no fsspec/async involved
            parts = path[5:].split("/", 1)  # strip "gs://"
            bucket_name, blob_path = parts[0], parts[1]
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            data = blob.download_as_bytes()
            return _original_read_file(io.BytesIO(data), **kwargs)
        return _original_read_file(path, **kwargs)

    gpd.read_file = _patched_read_file

    def _patched_cleanup():
        try:
            loop = _zarr_sync.loop[0] if _zarr_sync.loop else None
            if loop and not loop.is_closed():
                loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass

    # Replace zarr's atexit handler with our patched one
    atexit.unregister(_zarr_sync.cleanup_resources)
    atexit.register(_patched_cleanup)
