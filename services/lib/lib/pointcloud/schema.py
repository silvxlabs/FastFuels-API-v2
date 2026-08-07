"""What a stored point cloud looks like on disk.

Every service that writes or reads one agrees here and nowhere else: lakitu and
the uploader write it, griddle reads it. Keeping the layout in one module is
what makes "the same format regardless of where the cloud came from" a
constraint rather than a convention — before this it was three services
independently remembering the same column names.

The dataset is Hive-partitioned::

    <id>/cloud.parquet/tile_x=<i>/tile_y=<j>/part-*.parquet
    <id>/cloud.parquet/_manifest.json   tiling and coordinate scaling

Coordinates are stored as LAS scaled int32s, with the scale and offset in
`_manifest.json` and in each file's schema metadata, which keeps them small and
lossless rather than exploding to float64.
"""

import math

import numpy as np

# Directory holding a point cloud's dataset, under `<bucket>/<point_cloud_id>/`.
CLOUD_DIRNAME = "cloud.parquet"

# Position, what the point looks like, and what it is.
#
# `classification` is here because griddle's CHM branches on it: with ground
# returns it builds terrain from the classification, and without them it falls
# back to deriving a ground surface, which is measurably worse. The rest of LAS
# point format 6 -- return numbers, scan angle, point source id, user data, the
# synthetic/keypoint/withheld flags -- is dropped because nothing reads it.
#
# gps_time is dropped too, and it was the expensive one: 23% of the file at
# 3.76 B/pt, 883k distinct values in 1.1M points. LAZ compresses it only because
# LAZ keeps points in acquisition order, where consecutive timestamps differ by
# microseconds; sorting row groups spatially destroys the only structure a codec
# could exploit. Which acquisitions contributed is recorded per-resource in
# `source_extra.datasets`.
_BASE_FIELDS = [
    ("X", "<i4"),
    ("Y", "<i4"),
    ("Z", "<i4"),
    ("intensity", "<u2"),
    ("classification", "u1"),
]

# Colour is written only when the source carries it, so a cloud's columns say
# what it actually has rather than padding with zeros that mean nothing. The
# schema is fixed for a given cloud -- the point format is settled once, before
# any point is read -- so no part of a dataset ever disagrees with another.
_COLOR_FIELDS = [("red", "<u2"), ("green", "<u2"), ("blue", "<u2")]

# Measured on real 16 km2 output. Dictionary beats delta on anything
# low-cardinality -- intensity has only 1,614 distinct values -- and delta beats
# plain on the sorted coordinates. BYTE_STREAM_SPLIT was tried and lost.
DICT_COLUMNS = [
    "lod",
    "classification",
    "intensity",
    "red",
    "green",
    "blue",
]
DELTA_COLUMNS = {
    "X": "DELTA_BINARY_PACKED",
    "Y": "DELTA_BINARY_PACKED",
    "Z": "DELTA_BINARY_PACKED",
}

LOD_LEVELS = 6  # level k holds 1 in 4**(5-k); level 5 is the whole tile


def point_dtype(has_color: bool) -> np.dtype:
    """The record layout a writer is handed, and a reader gets back."""
    return np.dtype(_BASE_FIELDS + (_COLOR_FIELDS if has_color else []))


def columns_for(dtype: np.dtype) -> tuple[str, ...]:
    """The written columns for a record layout: its fields, plus the LOD level."""
    return ("lod", *dtype.names)


def cloud_prefix(bucket: str, point_cloud_id: str) -> str:
    """Full path of a point cloud's dataset."""
    return f"{bucket}/{point_cloud_id}/{CLOUD_DIRNAME}"


def cloud_location(bucket: str, point_cloud_id: str) -> tuple[str, str]:
    """``(bucket, prefix)``, for writers that talk to the storage client directly."""
    return bucket, f"{point_cloud_id}/{CLOUD_DIRNAME}"


def choose_tile_m(extent: float, target: float = 500.0) -> float:
    """Tile size must divide the extent by a power of two.

    Level k's LOD cell is extent/(128 * 2**k), so a tile of extent/2**t spans
    2**(7 + k - t) cells -- an integer for every level only when t is a whole
    number. At a fixed 500 m an extent like 5656.8 m (32 km2) gives 11.3 cells
    per tile, so tiles straddle cells and the pyramid stops being exact.
    """
    if extent <= 0:
        # A cloud with no horizontal extent: every point at one location, or a
        # single point. Degenerate, but a user can upload one, and log2(0) is
        # not a useful way to find out.
        return target
    t = int(round(math.log2(extent / target)))
    t = max(1, min(7, t))
    return extent / (2**t)


def tile_span(low: float, high: float, origin: float, tile_m: float) -> tuple[int, int]:
    """Inclusive tile-index range covering a coordinate span.

    The inverse of how the writer assigns a point to a tile, so a reader asking
    for a box gets exactly the partitions the writer would have put it in.
    """
    return (
        int(np.floor((low - origin) / tile_m)),
        int(np.floor((high - origin) / tile_m)),
    )
