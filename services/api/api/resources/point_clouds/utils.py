"""Point-cloud API validation, selection, and response encoding."""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from fastapi import HTTPException, status
from pydantic import ValidationError

from api.db.documents import get_document_async
from api.resources.point_clouds.cache import (
    get_point_cloud_storage,
    get_point_cloud_tile,
)
from api.resources.point_clouds.schema import (
    PointCloud,
    PointCloudDataMetadata,
    PointCloudTileDataResponse,
    PointCloudTileMetadata,
)
from lib.config import POINT_CLOUDS_COLLECTION
from lib.errors import ProcessingError
from lib.pointcloud.reader import PointCloudStorage, find_tile

MAX_BINARY_BYTES = 30 * 1024 * 1024
MAX_JSON_SCALARS = 1_000_000


@dataclass
class TileSelection:
    """A validated tile request ready to scan and encode."""

    point_cloud_id: str
    storage: PointCloudStorage
    tile: dict
    columns: list[str]
    classes: list[int] | None
    lod: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return tile_bounds(self.storage, self.tile)

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        return {
            column["name"]: np.dtype(column["dtype"])
            for column in self.storage.columns
            if column["name"] in self.columns
        }


def tile_bounds(
    storage: PointCloudStorage, tile: dict
) -> tuple[float, float, float, float]:
    """Return one indexed tile's bounds in cloud coordinates."""
    manifest = storage.manifest
    min_x = manifest["mins"][0] + tile["tile_x"] * manifest["tile_m"]
    min_y = manifest["mins"][1] + tile["tile_y"] * manifest["tile_m"]
    return min_x, min_y, min_x + manifest["tile_m"], min_y + manifest["tile_m"]


def _unprocessable(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail
    )


def _unreadable(error: ProcessingError) -> HTTPException:
    detail = error.message
    if error.suggestion:
        detail = f"{detail} {error.suggestion}"
    return _unprocessable(detail)


async def load_point_cloud(
    owner_id: str,
    domain_id: str,
    point_cloud_id: str,
) -> tuple[PointCloud, PointCloudStorage]:
    """Authorize and open a completed point cloud's indexed storage."""
    _, snapshot = await get_document_async(
        POINT_CLOUDS_COLLECTION,
        point_cloud_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    try:
        resource = PointCloud.model_validate(snapshot.to_dict())
    except ValidationError as error:
        raise _unprocessable(
            f"Point cloud {point_cloud_id} has invalid resource metadata."
        ) from error

    if not resource.checksum:
        raise _unprocessable(f"Point cloud {point_cloud_id} has no content checksum.")
    if resource.georeference is None:
        raise _unprocessable(f"Point cloud {point_cloud_id} has no georeference.")
    if resource.summary is None:
        raise _unprocessable(f"Point cloud {point_cloud_id} has no summary.")

    try:
        storage = await get_point_cloud_storage(point_cloud_id, resource.checksum)
    except ProcessingError as error:
        raise _unreadable(error) from error

    if resource.summary.point_count != storage.manifest["points"]:
        raise _unprocessable(
            f"Point cloud {point_cloud_id} metadata does not match its stored data. "
            "Recreate the point cloud, then retry."
        )
    return resource, storage


async def select_tile(
    owner_id: str,
    domain_id: str,
    point_cloud_id: str,
    tile_x: int,
    tile_y: int,
    lod: int | None,
    classes: str | None,
    columns: str | None,
) -> TileSelection:
    """Authorize a cloud and validate a tile selection against its index."""
    _, storage = await load_point_cloud(owner_id, domain_id, point_cloud_id)
    tile = find_tile(storage.tiles, tile_x, tile_y)
    if tile is None:
        raise _unprocessable(
            f"Point cloud {point_cloud_id} has no tile ({tile_x}, {tile_y})."
        )

    lod = storage.manifest["lod_levels"] - 1 if lod is None else lod
    if not 0 <= lod < storage.manifest["lod_levels"]:
        raise _unprocessable(
            f"lod must be between 0 and {storage.manifest['lod_levels'] - 1} "
            f"for point cloud {point_cloud_id}."
        )

    if classes is not None:
        try:
            class_values = [int(value.strip()) for value in classes.split(",")]
        except ValueError as error:
            raise _unprocessable(
                "classes must be comma-separated integers from 0 through 255."
            ) from error
        if not class_values or any(not 0 <= value <= 255 for value in class_values):
            raise _unprocessable("classes must contain integers from 0 through 255.")
        classes = sorted(set(class_values))

    available = [column["name"] for column in storage.columns]
    if columns is None:
        columns = available
    else:
        columns = [column.strip() for column in columns.split(",")]
        if any(not column for column in columns):
            raise _unprocessable("columns must not contain empty names.")
        if len(columns) != len(set(columns)):
            raise _unprocessable("columns must not contain duplicates.")
    unknown = [column for column in columns if column not in available]
    if unknown:
        raise _unprocessable(
            f"Unknown point-cloud columns: {', '.join(unknown)}. "
            f"Available columns: {', '.join(available)}."
        )
    return TileSelection(
        point_cloud_id=point_cloud_id,
        storage=storage,
        tile=tile,
        columns=columns,
        classes=classes,
        lod=lod,
    )


async def read_tile(selection: TileSelection) -> pa.Table:
    """Read a selection and verify its unfiltered count against the index."""
    try:
        data = await get_point_cloud_tile(
            selection.storage,
            selection.tile["tile_x"],
            selection.tile["tile_y"],
            selection.columns,
            selection.classes,
            selection.lod,
        )
    except ProcessingError as error:
        raise _unreadable(error) from error

    expected = selection.tile["points_by_lod"][selection.lod]
    if selection.classes is None and data.num_rows != expected:
        raise _unprocessable(
            f"Point cloud {selection.point_cloud_id} tile data does not match its "
            "index. Recreate the point cloud, then retry."
        )
    return data


def metadata_response(
    resource: PointCloud, storage: PointCloudStorage
) -> PointCloudDataMetadata:
    """Build the public tile catalogue from validated storage metadata."""
    manifest = storage.manifest
    bounds = resource.georeference.bounds
    return PointCloudDataMetadata(
        tile_m=manifest["tile_m"],
        lod_levels=manifest["lod_levels"],
        crs=resource.georeference.crs,
        bounds=(bounds[0], bounds[1], bounds[3], bounds[4]),
        scales=manifest["scales"],
        offsets=manifest["offsets"],
        columns={column["name"]: column["dtype"] for column in storage.columns},
        tiles=[
            PointCloudTileMetadata(
                tile_x=tile["tile_x"],
                tile_y=tile["tile_y"],
                bounds=tile_bounds(storage, tile),
                points_by_lod=tile["points_by_lod"],
            )
            for tile in storage.tiles
        ],
    )


def json_response(
    selection: TileSelection, data: pa.Table
) -> PointCloudTileDataResponse:
    """Encode a tile selection as the public columnar JSON model."""
    dtypes = {column["name"]: column["dtype"] for column in selection.storage.columns}
    manifest = selection.storage.manifest
    return PointCloudTileDataResponse(
        tile_x=selection.tile["tile_x"],
        tile_y=selection.tile["tile_y"],
        bounds=selection.bounds,
        lod=selection.lod,
        classes=selection.classes,
        scales=manifest["scales"],
        offsets=manifest["offsets"],
        columns={name: dtypes[name] for name in selection.columns},
        data={name: data.column(name).to_pylist() for name in selection.columns},
    )


def check_json_size(selection: TileSelection, rows: int | None = None) -> None:
    """Reject a known or actual JSON selection above the scalar limit."""
    if rows is None:
        if selection.classes is not None:
            return
        rows = selection.tile["points_by_lod"][selection.lod]
    _check_size(
        rows * len(selection.columns),
        MAX_JSON_SCALARS,
        "JSON point-cloud data",
        "Lower lod, filter classes/columns, or request /binary.",
    )


def check_binary_size(selection: TileSelection, rows: int | None = None) -> None:
    """Reject a known or actual binary selection above the byte limit."""
    if rows is None:
        if selection.classes is not None:
            return
        rows = selection.tile["points_by_lod"][selection.lod]
    _check_size(
        rows * sum(dtype.itemsize for dtype in selection.dtypes.values()),
        MAX_BINARY_BYTES,
        "Binary point-cloud data",
        "Lower lod or filter classes/columns.",
    )


def _check_size(actual: int, limit: int, what: str, hint: str) -> None:
    if actual > limit:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"{what} ({actual}) exceeds API response limit ({limit}). {hint}",
        )


def binary_payload(selection: TileSelection, data: pa.Table) -> bytes:
    """Concatenate selected columns as little-endian typed-array blocks."""
    dtypes = selection.dtypes
    return b"".join(
        data.column(name)
        .to_numpy(zero_copy_only=False)
        .astype(dtypes[name].newbyteorder("<"), copy=False)
        .tobytes()
        for name in selection.columns
    )


def binary_headers(selection: TileSelection, count: int) -> dict[str, str]:
    """Describe a binary tile body in browser-readable response headers."""
    manifest = selection.storage.manifest
    dtypes = selection.dtypes
    return {
        "X-Data-Columns": ",".join(selection.columns),
        "X-Data-Dtypes": ",".join(dtypes[name].name for name in selection.columns),
        "X-Data-Count": str(count),
        "X-Data-Tile": f"{selection.tile['tile_x']},{selection.tile['tile_y']}",
        "X-Data-Bounds": ",".join(str(value) for value in selection.bounds),
        "X-Data-LOD": str(selection.lod),
        "X-Data-Classes": (
            "all"
            if selection.classes is None
            else ",".join(str(value) for value in selection.classes)
        ),
        "X-Data-Scales": ",".join(str(value) for value in manifest["scales"]),
        "X-Data-Offsets": ",".join(str(value) for value in manifest["offsets"]),
    }
