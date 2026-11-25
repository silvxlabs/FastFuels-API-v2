"""In-memory stand-in for ``exporter.storage._blob``.

Writes go to an unseekable sink, like the GCS ``BlobWriter``, and an exception
inside ``open()`` stores nothing, like ``BlobWriter.terminate()``. Patch with
``patch("exporter.storage._blob", side_effect=FakeBlob)`` and read the
uploaded bytes from ``UPLOADS[gcs_path]``.
"""

import io
from contextlib import contextmanager
from pathlib import Path

UPLOADS: dict[str, bytes] = {}


class _Sink(io.RawIOBase):
    def __init__(self):
        self.data = bytearray()

    def writable(self):
        return True

    def write(self, b):
        self.data += bytes(b)
        return len(b)


class FakeBlob:
    def __init__(self, gcs_path: str):
        self.gcs_path = gcs_path

    @contextmanager
    def open(self, mode, **kwargs):
        sink = _Sink()
        yield sink
        UPLOADS[self.gcs_path] = bytes(sink.data)

    def upload_from_filename(self, local_path, **kwargs):
        UPLOADS[self.gcs_path] = Path(local_path).read_bytes()
