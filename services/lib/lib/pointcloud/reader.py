"""Reading a stored point cloud back.

Deliberately narrow: open the dataset, read its manifest, and pull the points
overlapping a box. Anything about what the caller does with those points -- the
lattice they land on, how they are blocked, what is computed from them -- stays
with the caller, because that is not a property of the format.
"""

import json

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

from lib.errors import ProcessingError
from lib.gcs import get_gcsfs_client
from lib.pointcloud.schema import tile_span

# The only columns any consumer reads. `lod` and `intensity` are never touched,
# which is a quarter of a tile's bytes.
POINT_COLUMNS = ["X", "Y", "Z", "classification"]

# Rows decoded at a time by `iter_points`. Peak memory is linear in this and
# wall time is flat across it -- measured on a 6.6M-point tile, 131k rows held
# 0.29 GB and 8.4M held 0.86 GB, all within noise of each other on time. It is
# the whole reason a caller that reduces can bound its own memory.
BATCH_ROWS = 1 << 20


def read_manifest(prefix: str) -> dict:
    """Read the dataset manifest, which carries its tiling and coordinate scaling.

    Args:
        prefix: Dataset prefix, e.g. ``<bucket>/<id>/cloud.parquet``.

    Returns:
        The manifest: ``tile_m``, ``mins``, ``maxs``, ``scales``, ``offsets``,
        and the point and tile counts.

    Raises:
        ProcessingError: POINT_CLOUD_UNREADABLE if the cloud has no manifest,
            which means it was never written or predates this format.
    """
    try:
        with get_gcsfs_client().open(f"{prefix}/_manifest.json", "rb") as stream:
            return json.load(stream)
    except FileNotFoundError as e:
        raise ProcessingError(
            code="POINT_CLOUD_UNREADABLE",
            message="This point cloud's stored data could not be read.",
            suggestion="Recreate the point cloud, then retry.",
            traceback=f"{prefix}/_manifest.json: {e}",
        ) from e


def open_dataset(prefix: str, filesystem=None) -> pa_ds.Dataset:
    """Open the partitioned dataset.

    Discovers by listing. ``_manifest.json`` is skipped by pyarrow's default
    underscore-prefix exclusion.

    Args:
        prefix: Dataset prefix.
        filesystem: Override for the filesystem, for reading a local copy.
            Defaults to GCS.
    """
    return pa_ds.dataset(
        prefix,
        filesystem=get_gcsfs_client() if filesystem is None else filesystem,
        format="parquet",
        partitioning="hive",
    )


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


def iter_points(dataset, manifest: dict, bounds: tuple, classes=None, filesystem=None):
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

    Each tile's file is fetched whole in one GET rather than left to pyarrow's
    ranged reads, since we need three quarters of its bytes anyway -- `lod` and
    `intensity` are never read. Expect little from that on its own: a dataset
    scan already batches its ranges well, and the two measured within noise of
    each other. It buys the simpler thing, which is that the bytes arrive
    before the decode does, at the price of holding the compressed file for the
    length of the tile.

    What is worth knowing is where the time goes: 6.6 M points decode from a
    memory buffer in 0.02 s, so a read taking seconds is all fetch and there is
    nothing to win by tuning the decode.

    Args:
        dataset: An open dataset from `open_dataset`.
        manifest: Its manifest, from `read_manifest`.
        bounds: ``(min_x, min_y, max_x, max_y)`` in the cloud's CRS.
        classes: ASPRS classes to keep. None keeps all.
        filesystem: The filesystem `dataset` was opened on. Defaults to GCS.

    Yields:
        Four arrays per batch: x, y and z in world units, and classification.
        Batches carrying no wanted class are skipped, so a caller may get none.
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
        source = pa.BufferReader(pa.py_buffer(_read_whole(fs, fragment.path)))
        for batch in pq.ParquetFile(source).iter_batches(
            batch_size=BATCH_ROWS, columns=POINT_COLUMNS
        ):
            classification = batch.column("classification").to_numpy(
                zero_copy_only=False
            )
            if wanted is None:
                keep = slice(None)
            else:
                keep = np.isin(classification, wanted)
                if not keep.any():
                    continue
                classification = classification[keep]
            yield (
                batch.column("X").to_numpy()[keep] * scales[0] + offsets[0],
                batch.column("Y").to_numpy()[keep] * scales[1] + offsets[1],
                batch.column("Z").to_numpy()[keep] * scales[2] + offsets[2],
                classification,
            )


def read_points(
    dataset, manifest: dict, bounds: tuple, classes=None, filesystem=None
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

    Returns:
        Four arrays: x, y and z in world units, and classification.
    """
    batches = list(iter_points(dataset, manifest, bounds, classes, filesystem))
    if not batches:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, np.empty(0, dtype=np.uint8)
    return tuple(np.concatenate(column) for column in zip(*batches, strict=True))
