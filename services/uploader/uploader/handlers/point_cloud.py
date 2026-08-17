"""
Point cloud upload handler for the Uploader service.

Ingests an uploaded point cloud (LAS or LAZ), validates that it is a readable
cloud with a coordinate reference system, reprojects it to the domain CRS when
necessary, and stores it as a partitioned Parquet dataset in
POINT_CLOUDS_BUCKET. Unlike raster grids — where reprojection means resampling
and is therefore rejected on a CRS mismatch — point reprojection is an exact
per-point coordinate transform, so mismatched uploads are transformed rather
than refused. The stored cloud is always in the domain CRS.

The stored format is `lib.pointcloud`, the same one lakitu writes for 3DEP.
That is deliberate: a point cloud is one resource type, and a reader should not
have to ask where it came from. Everything about the layout — the columns, the
tiling, the encodings — is defined once in `lib.pointcloud.schema`.

Everything streams: the staged upload is read from GCS in bounded chunks
(laspy + gcsfs) and written straight out as partitions, so nothing is ever held
whole. No local disk is used — on Cloud Run the filesystem is RAM-backed, and
per-instance memory is the budget this handler is designed around.

**A compressed upload already in the domain CRS used to be copied server-side
without rewriting. That is gone.** Every upload is now decoded and re-encoded,
because the stored format is no longer the uploaded one. The cost is smaller
than it looks: the copy path already decoded every point to census it, so what
is added is the encode, not the decode.
"""

from datetime import UTC, datetime

import laspy
import numpy as np
from pyproj import CRS as PyprojCRS
from pyproj import Transformer

from lib.config import (
    DOMAINS_COLLECTION,
    POINT_CLOUDS_BUCKET,
    POINT_CLOUDS_COLLECTION,
)
from lib.errors import ProcessingError
from lib.firestore import get_document, update_document
from lib.gcs import delete_file, get_gcsfs_client
from lib.laz import (
    build_output_header,
    normalize_record,
    point_format_id_for_dimensions,
)
from lib.pointcloud.schema import cloud_location, point_dtype
from lib.pointcloud.writer import write_parquet

# ~2M points/chunk keeps the streaming passes around 100 MB of working memory.
_CHUNK_POINTS = 2_000_000
_INT32_MAX = np.iinfo(np.int32).max


def handle_point_cloud(
    resource_id: str, bucket: str, object_name: str, doc: dict
) -> None:
    """Ingest an uploaded point cloud and store it as Parquet in the domain CRS.

    Args:
        resource_id: Point cloud document ID in Firestore.
        bucket: GCS bucket holding the staged upload (UPLOADS_BUCKET).
        object_name: Full GCS object path, e.g. "pointclouds/{id}/upload".
        doc: Point cloud document loaded from Firestore.
    """
    src = f"{bucket}/{object_name}"

    try:
        domain_crs_name = _domain_crs_name(doc["domain_id"])
        domain_crs = PyprojCRS.from_user_input(domain_crs_name)

        with get_gcsfs_client().open(src, "rb") as stream:
            with _open_cloud(stream) as reader:
                src_crs = _require_crs(reader.header)
                transformer = None
                if not src_crs.equals(domain_crs, ignore_axis_order=True):
                    # Exact per-point transform to the domain CRS (horizontal
                    # only — elevations pass through).
                    transformer = Transformer.from_crs(
                        src_crs, domain_crs, always_xy=True
                    )
                summary, bounds, size_bytes = _store(
                    reader, domain_crs, transformer, resource_id
                )

        update_document(
            POINT_CLOUDS_COLLECTION,
            resource_id,
            {
                "status": "completed",
                "modified_on": datetime.now(UTC),
                "georeference": {"crs": domain_crs_name, "bounds": bounds},
                "summary": summary,
                "size_bytes": size_bytes,
                "progress": {"message": "Complete", "percent": 100},
            },
        )
    finally:
        try:
            delete_file(f"gs://{bucket}/{object_name}")
        except Exception:
            pass


def _domain_crs_name(domain_id: str) -> str:
    """Return the domain's CRS name (e.g. 'EPSG:32612') from its document.

    A domain document without a CRS is an internal invariant violation, so a
    malformed document raises (KeyError/TypeError) rather than falling back.
    """
    _, snapshot = get_document(DOMAINS_COLLECTION, domain_id)
    domain = snapshot.to_dict()
    return domain["crs"]["properties"]["name"]


def _open_cloud(source) -> laspy.LasReader:
    """Open a LAS/LAZ stream for chunked reading.

    Args:
        source: Seekable binary stream or local path.

    Raises:
        ProcessingError: UNREADABLE_POINT_CLOUD if the source is not a
            readable LAS/LAZ file.
    """
    try:
        return laspy.open(source)
    except Exception as e:
        raise ProcessingError(
            code="UNREADABLE_POINT_CLOUD",
            message="The uploaded file could not be read as a point cloud.",
            suggestion="Upload a valid LAS or LAZ file.",
            traceback=str(e),
        )


def _require_crs(header: laspy.LasHeader) -> PyprojCRS:
    """Return the cloud's CRS, resolving compound CRSs to their horizontal part.

    Only the horizontal part is ever used: elevations pass through untouched, so
    a declared vertical CRS has nothing to act on.

    Raises:
        ProcessingError: MISSING_CRS when the file carries no CRS — without a
            source CRS the cloud cannot be reprojected to the domain CRS.
    """
    try:
        crs = header.parse_crs()
    except Exception:
        crs = None
    if crs is None:
        raise ProcessingError(
            code="MISSING_CRS",
            message=(
                "The point cloud has no coordinate reference system. Assign a "
                "CRS before uploading."
            ),
            suggestion=(
                "Set the CRS in your processing software, e.g. "
                "`pdal translate in.laz out.laz --writers.las.a_srs=EPSG:<code>`."
            ),
        )
    if crs.is_compound:
        crs = crs.sub_crs_list[0]
    return crs


def _store(reader, dst_crs, transformer, resource_id):
    """Stream the cloud into a partitioned Parquet dataset in the domain CRS.

    Normalises through `lib.laz` exactly as the 3DEP path does, so an upload and
    a fetch produce the same columns from the same source dimensions. That is
    also what makes classification uniform: LAS formats 0-5 pack it in a byte
    with the synthetic and withheld flags, and only the canonical format
    separates them.

    The canonical header is used for the point *format* only. The source's
    physical precision is kept: X/Y scales are converted when reprojection
    changes coordinate units, and Z keeps its original scale and offset because
    elevation is not transformed. The dataset records those values in its
    manifest, so a reader decodes with whatever this cloud was stored at.

    What does not survive: extra dimensions, and gps_time. Both were preserved
    when this wrote LAZ, and neither has anywhere to go in a fixed schema.

    Args:
        reader: Open LAS/LAZ reader positioned at the start of the points.
        dst_crs: CRS to store in (the domain CRS).
        transformer: Coordinate transform to apply, or None to keep coords.
        resource_id: Point cloud id, which decides where the dataset is written.

    Returns:
        ``(summary, bounds, size_bytes)``.
    """
    src_header = reader.header
    point_format_id = point_format_id_for_dimensions(
        src_header.point_format.dimension_names
    )
    bounds_2d = _output_bounds(src_header, transformer)
    header = build_output_header(dst_crs, bounds_2d, point_format_id=point_format_id)
    offsets = np.asarray(header.offsets).copy()
    offsets[2] = src_header.offsets[2]
    header.offsets = offsets
    header.scales = _storage_scales(
        src_header, dst_crs, transformer, bounds_2d, offsets
    )
    dtype = point_dtype("red" in header.point_format.dimension_names)

    info = {
        "mins": np.array([bounds_2d[0], bounds_2d[1], src_header.mins[2]]),
        "maxs": np.array([bounds_2d[2], bounds_2d[3], src_header.maxs[2]]),
        "scales": np.asarray(header.scales),
        "offsets": np.asarray(header.offsets),
    }

    def records():
        for points in reader.chunk_iterator(_CHUNK_POINTS):
            x = np.asarray(points.x)
            y = np.asarray(points.y)
            z = np.asarray(points.z)
            if transformer is not None:
                x, y = transformer.transform(x, y)
            record = normalize_record(points, header, x, y, z)
            out = np.empty(len(record), dtype=dtype)
            for name in dtype.names:
                out[name] = record.array[name]
            yield out

    bucket, prefix = cloud_location(POINT_CLOUDS_BUCKET, resource_id)
    result = write_parquet(records(), info, bucket, prefix)
    return result["summary"], result["bounds"], result["output_bytes"]


def _storage_scales(src_header, dst_crs, transformer, bounds, offsets):
    """Coordinate scales in stored CRS units, preserving source precision."""
    scales = np.asarray(src_header.scales, dtype=np.float64).copy()
    src_crs = _require_crs(src_header)

    if transformer is not None and _horizontal_units(src_crs) != _horizontal_units(
        dst_crs
    ):
        min_x, min_y = src_header.mins[:2]
        max_x, max_y = src_header.maxs[:2]
        xs = np.array([min_x, min_x, max_x, max_x, (min_x + max_x) / 2])
        ys = np.array([min_y, max_y, min_y, max_y, (min_y + max_y) / 2])

        out_x, out_y = transformer.transform(xs, ys)
        x_step_x, x_step_y = transformer.transform(xs + scales[0], ys)
        y_step_x, y_step_y = transformer.transform(xs, ys + scales[1])

        # Each output coordinate can depend on both source axes. Their absolute
        # contributions bound the source quantization error after reprojection.
        converted_x = np.abs(x_step_x - out_x) + np.abs(y_step_x - out_x)
        converted_y = np.abs(x_step_y - out_y) + np.abs(y_step_y - out_y)
        scales[0] = _smallest_positive(converted_x)
        scales[1] = _smallest_positive(converted_y)

    # The output offsets anchor X/Y near their minima. Raise an unusually fine
    # scale only when the projected span would otherwise exceed signed int32.
    for axis, (low, high) in enumerate(
        ((bounds[0], bounds[2]), (bounds[1], bounds[3]))
    ):
        required = max(abs(low - offsets[axis]), abs(high - offsets[axis])) / _INT32_MAX
        scales[axis] = max(scales[axis], np.nextafter(required, np.inf))

    return scales


def _horizontal_units(crs) -> tuple[float, ...]:
    """Conversion factors identifying the CRS's two horizontal axis units."""
    return tuple(float(axis.unit_conversion_factor) for axis in crs.axis_info[:2])


def _smallest_positive(values) -> float:
    usable = np.asarray(values)
    usable = usable[np.isfinite(usable) & (usable > 0)]
    if not usable.size:
        raise ValueError("coordinate transform collapsed a source scale increment")
    return float(usable.min())


def _output_bounds(src_header, transformer) -> tuple:
    """Horizontal bounds of the stored cloud, in the destination CRS.

    Transforms all four corners rather than just the minimum, because a
    reprojected rectangle is not a rectangle and its extreme is not necessarily
    a corner of the source one. Only used to place the tile origin and choose the
    tile size, both of which tolerate being slightly generous; the bounds
    actually reported come from the points themselves.
    """
    min_x, min_y = src_header.mins[0], src_header.mins[1]
    max_x, max_y = src_header.maxs[0], src_header.maxs[1]
    if transformer is None:
        return (min_x, min_y, max_x, max_y)
    xs, ys = transformer.transform(
        [min_x, min_x, max_x, max_x], [min_y, max_y, min_y, max_y]
    )
    return (min(xs), min(ys), max(xs), max(ys))
