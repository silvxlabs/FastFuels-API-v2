"""Reading a stored point cloud back.

Deliberately narrow: open the dataset, read its manifest, and pull the points
overlapping a box. Anything about what the caller does with those points -- the
lattice they land on, how they are blocked, what is computed from them -- stays
with the caller, because that is not a property of the format.
"""

import json

import numpy as np
import pyarrow.dataset as pa_ds

from lib.errors import ProcessingError
from lib.gcs import get_gcsfs_client
from lib.pointcloud.schema import tile_span


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

    ``_metadata`` and ``_manifest.json`` are skipped by pyarrow's default
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


def read_points(dataset, manifest: dict, bounds: tuple, classes=None) -> tuple:
    """Read ``(x, y, z, classification)`` for the partitions overlapping `bounds`.

    Prunes on the Hive partition columns, so a caller reads only the tiles it
    touches rather than the whole cloud. Points outside `bounds` but inside those
    tiles come back too: a partition is the finest thing that can be skipped, and
    trimming further here would only hide that from a caller who has to decide
    what falls in its own cells anyway.

    Only the columns these consumers use are read; colour is never touched.

    Args:
        dataset: An open dataset from `open_dataset`.
        manifest: Its manifest, from `read_manifest`.
        bounds: ``(min_x, min_y, max_x, max_y)`` in the cloud's CRS.
        classes: ASPRS classes to keep, pushed into the read. None keeps all.

    Returns:
        Four arrays: x, y and z in world units, and classification.
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
    if classes is not None:
        selection = selection & pa_ds.field("classification").isin(list(classes))

    table = dataset.to_table(
        columns=["X", "Y", "Z", "classification"], filter=selection
    )
    if table.num_rows == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, np.empty(0, dtype=np.uint8)

    scales, offsets = manifest["scales"], manifest["offsets"]
    return (
        table.column("X").to_numpy() * scales[0] + offsets[0],
        table.column("Y").to_numpy() * scales[1] + offsets[1],
        table.column("Z").to_numpy() * scales[2] + offsets[2],
        table.column("classification").to_numpy(),
    )
