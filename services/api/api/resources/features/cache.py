"""
api/v2/resources/features/cache.py

Async byte-offset index + warm byte-range fetch for the Feature data
streaming endpoint.

The cache stores only a compact per-feature byte-offset table; raw
GeoJSON bytes are not retained. On cache hit, paginated requests issue
a single async GCS GET with a Range header for just the bytes the page
needs and splice them with the cached header / footer slabs.
"""

import asyncio
import json
from dataclasses import dataclass

import gcsfs
from ring import lru

from lib.config import FEATURES_BUCKET

# Data-streaming invariants — single source of truth.
# The router imports these for FastAPI Query defaults and exception → HTTP
# status mapping; the cache enforces the size cap itself.
MAX_PAGE_SIZE = 5000
DEFAULT_PAGE_SIZE = 1000
MAX_GEOJSON_BYTES = 30 * 1024 * 1024
GEOJSON_MEDIA_TYPE = "application/geo+json"


@dataclass(frozen=True)
class FeatureIndex:
    """Compact byte-offset table for a single GeoJSON FeatureCollection.

    `header_bytes` covers byte 0 through (and including) the `[` that opens
    the `features` array. `footer_bytes` is the matching `]` through EOF,
    closing the array and the outer object. `offsets` records the byte
    range of each Feature object inside the array.

    Splicing `header_bytes + <feature_byte_slice> + footer_bytes` always
    yields a valid stand-alone GeoJSON FeatureCollection.
    """

    header_bytes: bytes
    footer_bytes: bytes
    offsets: tuple[tuple[int, int], ...]

    @property
    def total_features(self) -> int:
        return len(self.offsets)


class InvalidFeatureGeoJSON(Exception):
    """The blob exists but is not a parseable {..., "features": [...]} shape."""


class PageOutOfRange(Exception):
    """`page * size` is beyond the last feature."""

    def __init__(self, page: int, size: int, total: int):
        super().__init__(f"page {page} (size {size}) out of range for {total} features")
        self.page = page
        self.size = size
        self.total = total


class PageTooLarge(Exception):
    """Assembled page exceeds `MAX_GEOJSON_BYTES`."""

    def __init__(self, page_bytes: int, limit: int):
        super().__init__(
            f"page payload {page_bytes} bytes exceeds {limit} bytes; lower `size`"
        )
        self.page_bytes = page_bytes
        self.limit = limit


_WHITESPACE = " \t\n\r"


def _scan_offsets(text: str) -> tuple[list[tuple[int, int]], int, int]:
    """Step through the `features` array recording (start, end) per feature.

    Uses `json.JSONDecoder.raw_decode` (C-implemented) for the actual
    feature parsing; the parsed dicts are immediately discarded — we keep
    only the byte ranges.

    Returns `(offsets, header_end, footer_start)` where `header_end` is the
    index immediately past the opening `[` and `footer_start` is the index
    of the closing `]`.
    """
    decoder = json.JSONDecoder()
    key_pos = text.find('"features"')
    if key_pos < 0:
        raise InvalidFeatureGeoJSON("no top-level 'features' key found")

    n = len(text)
    i = key_pos + len('"features"')
    while i < n and text[i] in _WHITESPACE:
        i += 1
    if i >= n or text[i] != ":":
        raise InvalidFeatureGeoJSON("expected ':' after 'features' key")
    i += 1
    while i < n and text[i] in _WHITESPACE:
        i += 1
    if i >= n or text[i] != "[":
        raise InvalidFeatureGeoJSON("'features' value is not an array")

    header_end = i + 1
    i = header_end

    offsets: list[tuple[int, int]] = []
    while i < n and text[i] in _WHITESPACE:
        i += 1
    while i < n and text[i] != "]":
        try:
            _obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError as exc:
            raise InvalidFeatureGeoJSON(
                f"malformed feature at byte {i}: {exc.msg}"
            ) from exc
        offsets.append((i, end))
        j = end
        while j < n and text[j] in _WHITESPACE:
            j += 1
        if j < n and text[j] == ",":
            j += 1
            while j < n and text[j] in _WHITESPACE:
                j += 1
        i = j

    if i >= n or text[i] != "]":
        raise InvalidFeatureGeoJSON("'features' array never closed")
    footer_start = i
    return offsets, header_end, footer_start


def _build_index_sync(raw: bytes) -> FeatureIndex:
    """CPU-bound index build. Run under `asyncio.to_thread`."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidFeatureGeoJSON(f"GeoJSON is not valid UTF-8: {exc}") from exc
    offsets, header_end, footer_start = _scan_offsets(text)
    return FeatureIndex(
        header_bytes=text[:header_end].encode("utf-8"),
        footer_bytes=text[footer_start:].encode("utf-8"),
        offsets=tuple(offsets),
    )


def _blob_path(domain_id: str, feature_id: str) -> str:
    return f"{FEATURES_BUCKET}/{domain_id}/{feature_id}.geojson"


@lru(maxsize=128, force_asyncio=True)
async def get_feature_index(domain_id: str, feature_id: str) -> FeatureIndex:
    """Read the GeoJSON once, scan for feature byte ranges, drop the raw bytes.

    Cached by `(domain_id, feature_id)`. Cache entries persist for the life
    of the process; features are immutable post-completion so no
    invalidation is needed.

    Raises:
        FileNotFoundError: blob is not present on GCS.
        InvalidFeatureGeoJSON: blob exists but is not a parseable
            `{..., "features": [...]}` shape.
    """
    return await asyncio.to_thread(_read_and_build_index, domain_id, feature_id)


def _read_and_build_index(domain_id: str, feature_id: str) -> FeatureIndex:
    """Sync entrypoint: read the GeoJSON bytes and build the index.

    Uses gcsfs's sync wrapper (`cat_file`, no underscore), which schedules
    the underlying coroutine on fsspec's dedicated internal loop. Calling
    the async `_cat_file` directly from the FastAPI request loop pins the
    aiohttp session to that loop and breaks once a different loop touches
    it (e.g. after a layerset upload uses the same module-level
    `gcsfs_client` in `lib.gcs.blobs`).
    """
    fs = gcsfs.GCSFileSystem()
    raw = fs.cat_file(_blob_path(domain_id, feature_id))
    return _build_index_sync(raw)


def _read_byte_range(domain_id: str, feature_id: str, start: int, end: int) -> bytes:
    """Sync byte-range read; runs under ``asyncio.to_thread`` from the handler."""
    fs = gcsfs.GCSFileSystem()
    return fs.cat_file(_blob_path(domain_id, feature_id), start=start, end=end)


async def fetch_feature_page(
    domain_id: str,
    feature_id: str,
    page: int,
    size: int,
) -> tuple[bytes, int, int]:
    """Return `(page_bytes, num_features, total_features)`.

    `page_bytes` is a self-contained GeoJSON FeatureCollection containing
    the requested page of features and the source file's original top-level
    fields (CRS block etc.) — suitable for streaming directly to the client.

    Raises:
        FileNotFoundError: blob is missing on GCS.
        InvalidFeatureGeoJSON: blob is malformed.
        PageOutOfRange: `page * size >= total_features` and `total_features > 0`.
        PageTooLarge: assembled page exceeds `MAX_GEOJSON_BYTES`.
    """
    index = await get_feature_index(domain_id, feature_id)
    total = index.total_features

    if total == 0:
        if page > 0:
            raise PageOutOfRange(page, size, total)
        page_bytes = index.header_bytes + index.footer_bytes
        if len(page_bytes) > MAX_GEOJSON_BYTES:
            raise PageTooLarge(len(page_bytes), MAX_GEOJSON_BYTES)
        return page_bytes, 0, 0

    start_idx = page * size
    if start_idx >= total:
        raise PageOutOfRange(page, size, total)
    end_idx = min(start_idx + size, total)
    page_offsets = index.offsets[start_idx:end_idx]

    body = await asyncio.to_thread(
        _read_byte_range,
        domain_id,
        feature_id,
        page_offsets[0][0],
        page_offsets[-1][1],
    )
    page_bytes = index.header_bytes + body + index.footer_bytes
    if len(page_bytes) > MAX_GEOJSON_BYTES:
        raise PageTooLarge(len(page_bytes), MAX_GEOJSON_BYTES)
    return page_bytes, end_idx - start_idx, total
