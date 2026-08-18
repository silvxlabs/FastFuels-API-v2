"""Reading a stored point cloud back.

Deliberately narrow: open the standard Parquet index, read the small application
manifest, and pull selected points. Anything about what the caller does with
those points -- the lattice they land on, how they are blocked, what is computed
from them -- stays with the caller, because that is not a property of the format.
"""

import json
import math
import re
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

from lib.errors import ProcessingError
from lib.gcs import get_gcsfs_client
from lib.pointcloud.schema import LOD_LEVELS, point_dtype, tile_span

# The only columns any consumer reads. What that skips is `intensity`, a fifth
# of a tile's stored bytes -- 21.6% measured over two real datasets, against
# 0.05% for `lod`.
POINT_COLUMNS = ["X", "Y", "Z", "classification"]

# Rows decoded at a time by `iter_points`. Peak memory is linear in this and
# wall time is flat across it -- measured on a 6.6M-point tile, 131k rows held
# 0.29 GB and 8.4M held 0.86 GB, all within noise of each other on time. It is
# the whole reason a caller that reduces can bound its own memory.
BATCH_ROWS = 1 << 20


def _unreadable(prefix: str, detail: str) -> ProcessingError:
    return ProcessingError(
        code="POINT_CLOUD_UNREADABLE",
        message="This point cloud's stored data could not be read.",
        suggestion="Recreate the point cloud, then retry.",
        traceback=f"{prefix}: {detail}",
    )


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _vector(manifest: dict, name: str, *, positive: bool = False) -> list[float]:
    value = manifest[name]
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ValueError(f"{name} must contain exactly three numbers")
    result = [float(v) for v in value]
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain only finite numbers")
    if positive and not all(v > 0 for v in result):
        raise ValueError(f"{name} must contain only positive numbers")
    return result


def _validate_manifest(manifest: dict) -> dict:
    """Validate the application metadata Parquet does not carry."""
    if not isinstance(manifest, dict):
        raise ValueError("the manifest root must be an object")
    if "manifest_version" in manifest:
        raise ValueError("manifest_version is not part of the point-cloud format")

    required = {
        "tile_m",
        "tiles",
        "points",
        "lod_levels",
        "scales",
        "offsets",
        "mins",
        "maxs",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")

    tile_m = manifest["tile_m"]
    if (
        isinstance(tile_m, bool)
        or not isinstance(tile_m, (int, float))
        or not math.isfinite(tile_m)
        or tile_m <= 0
    ):
        raise ValueError("tile_m must be a positive finite number")

    _integer(manifest["points"], "points")
    _integer(manifest["tiles"], "tiles")
    if _integer(manifest["lod_levels"], "lod_levels", 1) != LOD_LEVELS:
        raise ValueError(f"lod_levels must equal {LOD_LEVELS}")

    _vector(manifest, "scales", positive=True)
    _vector(manifest, "offsets")
    mins = _vector(manifest, "mins")
    maxs = _vector(manifest, "maxs")
    if any(low > high for low, high in zip(mins, maxs, strict=True)):
        raise ValueError("mins must not exceed maxs")

    return manifest


def read_manifest(prefix: str, filesystem=None) -> dict:
    """Read the dataset manifest, which carries its tiling and coordinate scaling.

    Args:
        prefix: Dataset prefix, e.g. ``<bucket>/<id>/cloud.parquet``.

    Returns:
        The manifest: tiling, coordinate scaling, bounds, and total counts.

    Raises:
        ProcessingError: POINT_CLOUD_UNREADABLE if the cloud has no valid
            manifest in the current format.
    """
    fs = get_gcsfs_client() if filesystem is None else filesystem
    try:
        with _open_stream(fs, f"{prefix}/_manifest.json") as stream:
            return _validate_manifest(json.load(stream))
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as e:
        raise _unreadable(prefix, str(e)) from e


def open_dataset(prefix: str, filesystem=None) -> pa_ds.Dataset:
    """Open the partitioned dataset from its standard global Parquet metadata.

    Args:
        prefix: Dataset prefix.
        filesystem: Override for the filesystem, for reading a local copy.
            Defaults to GCS.
    """
    return pa_ds.parquet_dataset(
        f"{prefix}/_metadata",
        filesystem=get_gcsfs_client() if filesystem is None else filesystem,
        partitioning="hive",
    )


@dataclass(frozen=True)
class PointCloudStorage:
    """One validated, indexed point cloud ready for repeated reads."""

    prefix: str
    manifest: dict
    dataset: pa_ds.Dataset
    columns: tuple[dict, ...]
    tiles: tuple[dict, ...]


_PART_PATH = re.compile(r"(?:^|/)tile_x=(-?\d+)/tile_y=(-?\d+)/part-(\d{5})\.parquet$")


def _stored_columns(dataset: pa_ds.Dataset) -> tuple[dict, ...]:
    """Public point fields and wire dtypes, derived from the Parquet schema."""
    internal = {"lod", "tile_x", "tile_y"}
    columns = tuple(
        {
            "name": field.name,
            "dtype": np.dtype(field.type.to_pandas_dtype()).name,
        }
        for field in dataset.schema
        if field.name not in internal
    )
    layouts = {
        tuple(
            (name, np.dtype(dtype).name)
            for name, (dtype, _) in point_dtype(has_color).fields.items()
        )
        for has_color in (False, True)
    }
    if tuple((column["name"], column["dtype"]) for column in columns) not in layouts:
        raise ValueError("Parquet columns do not match a supported point-cloud layout")
    if dataset.schema.field("lod").type != pa.uint8():
        raise ValueError("Parquet lod must be uint8")
    return columns


def _tile_catalogue(dataset: pa_ds.Dataset, manifest: dict) -> tuple[dict, ...]:
    """Derive occupied tiles and exact cumulative LOD counts from `_metadata`."""
    counts = {}
    parts = {}
    for fragment in sorted(dataset.get_fragments(), key=lambda item: item.path):
        match = _PART_PATH.search(fragment.path)
        if match is None:
            raise ValueError(f"invalid Parquet part path {fragment.path!r}")
        tile = (int(match.group(1)), int(match.group(2)))
        sequence = int(match.group(3))
        if sequence in parts.setdefault(tile, set()):
            raise ValueError(f"duplicate Parquet part {fragment.path!r}")
        parts[tile].add(sequence)
        lod_counts = counts.setdefault(tile, np.zeros(LOD_LEVELS, dtype=np.int64))
        for row_group in fragment.row_groups:
            statistics = row_group.statistics or {}
            lod_statistics = statistics.get("lod")
            if lod_statistics is None or lod_statistics.get(
                "min"
            ) != lod_statistics.get("max"):
                raise ValueError(
                    f"{fragment.path} row groups must have one exact lod value"
                )
            lod = lod_statistics["min"]
            if isinstance(lod, bool) or not isinstance(lod, int):
                raise ValueError(f"{fragment.path} has a non-integer lod statistic")
            if not 0 <= lod < LOD_LEVELS:
                raise ValueError(f"{fragment.path} has lod {lod} outside the format")
            lod_counts[lod] += row_group.num_rows

    tiles = []
    for (tile_x, tile_y), lod_counts in sorted(counts.items()):
        expected_parts = set(range(len(parts[(tile_x, tile_y)])))
        if parts[(tile_x, tile_y)] != expected_parts:
            raise ValueError(f"tile {(tile_x, tile_y)} has non-contiguous part numbers")
        points_by_lod = np.cumsum(lod_counts)
        tiles.append(
            {
                "tile_x": tile_x,
                "tile_y": tile_y,
                "points": int(points_by_lod[-1]),
                "points_by_lod": [int(value) for value in points_by_lod],
            }
        )

    if len(tiles) != manifest["tiles"]:
        raise ValueError(
            f"Parquet has {len(tiles)} tiles but the manifest declares "
            f"{manifest['tiles']}"
        )
    total = sum(tile["points"] for tile in tiles)
    if total != manifest["points"]:
        raise ValueError(
            f"Parquet has {total} points but the manifest declares {manifest['points']}"
        )
    return tuple(tiles)


def open_point_cloud(prefix: str, filesystem=None) -> PointCloudStorage:
    """Open and validate the manifest, global Parquet index, and tile catalogue."""
    fs = get_gcsfs_client() if filesystem is None else filesystem
    try:
        manifest = read_manifest(prefix, fs)
        dataset = open_dataset(prefix, fs)
        return PointCloudStorage(
            prefix=prefix,
            manifest=manifest,
            dataset=dataset,
            columns=_stored_columns(dataset),
            tiles=_tile_catalogue(dataset, manifest),
        )
    except ProcessingError:
        raise
    except Exception as error:
        raise _unreadable(prefix, str(error)) from error


def _open_stream(filesystem, path: str):
    """A seekable handle, on either flavour of filesystem.

    The counterpart to `_read_whole`: this one leaves the reads to pyarrow, so
    it fetches only the byte ranges a selection actually needs.
    """
    if isinstance(filesystem, pa.fs.FileSystem):
        return filesystem.open_input_file(path)
    return filesystem.open(path, "rb")


def _read_whole(filesystem, path: str) -> bytes:
    """One whole-file read, on either flavour of filesystem.

    `open_dataset` takes what pyarrow takes, which is an fsspec filesystem or a
    native pyarrow one, and the two spell this differently. fsspec's `cat_file`
    is a single GET, which is the point on GCS; pyarrow's `readall` fills from
    the stream's own buffer, which is what a local filesystem wants anyway.
    """
    if isinstance(filesystem, pa.fs.FileSystem):
        with filesystem.open_input_file(path) as stream:
            return stream.readall()
    return filesystem.cat_file(path)


def _lod_row_groups(parquet_file, max_lod: int) -> tuple[list[int], bool]:
    """Row groups holding any point at or below `max_lod`, and whether any straddles.

    This is the one predicate the stored layout can actually answer cheaply. The
    writer emits a row group per LOD level that has points, so `lod` statistics
    come out single-valued and the selection is a clean prefix -- reading
    `lod <= 2` of a 6.6 M-point tile touches 103,122 rows. Nothing here depends
    on that holding: a row group straddling the cut is kept and its rows filtered
    afterwards, and one with no statistics at all is kept for the same reason.

    Returns:
        The row-group indices to read, and whether any of them needs its rows
        re-checked against `lod` because its statistics cross `max_lod`.
    """
    metadata = parquet_file.metadata
    column = [
        metadata.schema.column(i).name for i in range(metadata.num_columns)
    ].index("lod")

    keep, straddles = [], False
    for index in range(metadata.num_row_groups):
        statistics = metadata.row_group(index).column(column).statistics
        if statistics is None:
            keep.append(index)
            straddles = True
        elif statistics.min <= max_lod:
            keep.append(index)
            straddles = straddles or statistics.max > max_lod
    return keep, straddles


def find_tile(tiles: tuple[dict, ...], tile_x: int, tile_y: int) -> dict | None:
    """Return one sorted catalogue entry for a tile coordinate."""
    for tile in tiles:
        key = (tile["tile_x"], tile["tile_y"])
        if key == (tile_x, tile_y):
            return tile
        if key > (tile_x, tile_y):
            break
    return None


def read_tile(
    storage: PointCloudStorage,
    tile_x: int,
    tile_y: int,
    columns: list[str] | tuple[str, ...],
    classes=None,
    max_lod: int | None = None,
) -> pa.Table:
    """Read one projected tile through the indexed Arrow Dataset scanner.

    Partition, LOD, and classification filters are pushed into Parquet before
    the selected columns are materialized. Opening `storage.dataset` consumed
    `_metadata`, so this performs no dataset listing or per-part footer reads.

    Args:
        storage: A point cloud from :func:`open_point_cloud`.
        tile_x: Hive tile x coordinate.
        tile_y: Hive tile y coordinate.
        columns: Stored columns to return, in response order.
        classes: Optional ASPRS classification values to keep.
        max_lod: Keep cumulative levels 0 through this level. ``None`` reads
            every point.
    Returns:
        An Arrow table with the requested columns in response order.

    Raises:
        ValueError: The requested columns or LOD are invalid.
        ProcessingError: The indexed Parquet data is missing or unreadable.
    """
    columns = list(columns)
    available = {column["name"] for column in storage.columns}
    if len(columns) != len(set(columns)):
        raise ValueError("columns must not contain duplicates")
    unknown = [name for name in columns if name not in available]
    if unknown:
        raise ValueError(f"unknown point-cloud columns: {', '.join(unknown)}")
    levels = storage.manifest["lod_levels"]
    if max_lod is not None and not 0 <= max_lod < levels:
        raise ValueError(f"max_lod must be between 0 and {levels - 1}")

    selection = (pa_ds.field("tile_x") == tile_x) & (pa_ds.field("tile_y") == tile_y)
    if max_lod is not None:
        selection = selection & (pa_ds.field("lod") <= max_lod)
    if classes is not None:
        selection = selection & pa_ds.field("classification").isin(list(classes))

    try:
        return storage.dataset.to_table(columns=columns, filter=selection)
    except ProcessingError:
        raise
    except Exception as error:
        raise _unreadable(storage.prefix, str(error)) from error


def iter_points(
    dataset,
    manifest: dict,
    bounds: tuple,
    classes=None,
    filesystem=None,
    max_lod: int | None = None,
):
    """Yield ``(x, y, z, classification)`` a batch at a time, tile by tile.

    Prunes on the Hive partition columns, so a caller reads only the tiles it
    touches rather than the whole cloud. Points outside `bounds` but inside those
    tiles come back too: a partition is the finest thing that can be skipped, and
    trimming further here would only hide that from a caller who has to decide
    what falls in its own cells anyway.

    Batched because a caller that reduces -- a minimum or a maximum per cell --
    never needs the tile at once, and batching is what stops its memory tracking
    the tile's point count. A tile is a fixed area, so that count follows the
    cloud's density, which nothing bounds.

    How each tile is fetched follows how much of it is wanted, because the two
    strategies invert. Without an LOD cut the read wants three quarters of the
    file's bytes anyway -- only `lod` and `intensity` go unread -- so it is
    fetched whole in one GET and decoded from memory. With a cut that excludes
    row groups, pyarrow reads only the ranges it needs and the GET would be
    almost all waste. Measured on a 40.6 MiB tile, whole-file fetch against
    ranged: `lod <= 2` 0.94 s against 0.47 s, `lod <= 3` 1.00 against 0.55, but
    the whole file 1.03 against 3.61. A cut that excludes nothing therefore
    falls back to the GET rather than paying round trips for it.

    What is worth knowing either way is where the time goes: 6.6 M points decode
    from a memory buffer in 0.02 s, so a read taking seconds is all fetch and
    there is nothing to win by tuning the decode.

    Args:
        dataset: An open dataset from `open_dataset`.
        manifest: Its manifest, from `read_manifest`.
        bounds: ``(min_x, min_y, max_x, max_y)`` in the cloud's CRS.
        classes: ASPRS classes to keep. None keeps all.
        filesystem: The filesystem `dataset` was opened on. Defaults to GCS.
        max_lod: Coarsest-first LOD cut -- keep levels 0..`max_lod`. None reads
            every level, which is what a caller wanting all the points needs.
            Unlike `classes` and `bounds` this one really does skip stored data
            rather than filter it afterwards; see `_lod_row_groups`.

    Yields:
        Four arrays per batch: x, y and z in world units, and classification.
        Batches left empty by the filters are skipped, so a caller may get none.
    """
    min_x, min_y, max_x, max_y = bounds
    tile_m, origin = manifest["tile_m"], manifest["mins"]
    tx0, tx1 = tile_span(min_x, max_x, origin[0], tile_m)
    ty0, ty1 = tile_span(min_y, max_y, origin[1], tile_m)

    selection = (
        (pa_ds.field("tile_x") >= tx0)
        & (pa_ds.field("tile_x") <= tx1)
        & (pa_ds.field("tile_y") >= ty0)
        & (pa_ds.field("tile_y") <= ty1)
    )

    scales, offsets = manifest["scales"], manifest["offsets"]
    wanted = None if classes is None else list(classes)
    fs = get_gcsfs_client() if filesystem is None else filesystem

    for fragment in dataset.get_fragments(filter=selection):
        row_groups, recheck_lod = None, False
        # An LOD cut is the only thing that makes ranged reads pay, so it is the
        # only thing that opens a stream. Deciding needs the footer, which is
        # why the cut is read off a handle and the whole-file fetch happens
        # afterwards if it turns out nothing was excluded.
        stream = None if max_lod is None else _open_stream(fs, fragment.path)
        try:
            if stream is not None:
                parquet_file = pq.ParquetFile(stream)
                row_groups, recheck_lod = _lod_row_groups(parquet_file, max_lod)
                if not row_groups:
                    continue
                if len(row_groups) == parquet_file.metadata.num_row_groups:
                    stream.close()
                    stream = None
            if stream is None:
                source = pa.BufferReader(pa.py_buffer(_read_whole(fs, fragment.path)))
                parquet_file = pq.ParquetFile(source)

            columns = [*POINT_COLUMNS, "lod"] if recheck_lod else POINT_COLUMNS
            yield from _fold_batches(
                parquet_file,
                columns,
                row_groups,
                wanted,
                recheck_lod,
                max_lod,
                scales,
                offsets,
            )
        finally:
            if stream is not None:
                stream.close()


def _fold_batches(
    parquet_file, columns, row_groups, wanted, recheck_lod, max_lod, scales, offsets
):
    """The per-batch half of `iter_points`, once the source is settled."""
    for batch in parquet_file.iter_batches(
        batch_size=BATCH_ROWS, columns=columns, row_groups=row_groups
    ):
        classification = batch.column("classification").to_numpy(zero_copy_only=False)
        keep = None
        if wanted is not None:
            keep = np.isin(classification, wanted)
        if recheck_lod:
            within = batch.column("lod").to_numpy(zero_copy_only=False) <= max_lod
            keep = within if keep is None else (keep & within)

        if keep is None:
            keep = slice(None)
        elif not keep.any():
            continue
        else:
            classification = classification[keep]

        yield (
            batch.column("X").to_numpy()[keep] * scales[0] + offsets[0],
            batch.column("Y").to_numpy()[keep] * scales[1] + offsets[1],
            batch.column("Z").to_numpy()[keep] * scales[2] + offsets[2],
            classification,
        )


def read_points(
    dataset,
    manifest: dict,
    bounds: tuple,
    classes=None,
    filesystem=None,
    max_lod: int | None = None,
) -> tuple:
    """`iter_points` gathered into one array per column.

    For callers that genuinely want the whole selection at once. A caller that
    reduces should iterate instead, or it pays the memory `iter_points` exists
    to avoid.

    Args:
        dataset: An open dataset from `open_dataset`.
        manifest: Its manifest, from `read_manifest`.
        bounds: ``(min_x, min_y, max_x, max_y)`` in the cloud's CRS.
        classes: ASPRS classes to keep. None keeps all.
        filesystem: The filesystem `dataset` was opened on. Defaults to GCS, as
            `open_dataset` does, so the two are given the same override.
        max_lod: Coarsest-first LOD cut, as `iter_points` takes it.

    Returns:
        Four arrays: x, y and z in world units, and classification.
    """
    batches = list(iter_points(dataset, manifest, bounds, classes, filesystem, max_lod))
    if not batches:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, np.empty(0, dtype=np.uint8)
    return tuple(np.concatenate(column) for column in zip(*batches, strict=True))
