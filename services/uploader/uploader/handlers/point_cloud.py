"""
Point cloud upload handler for the Uploader service.

Ingests an uploaded point cloud (LAS or LAZ), validates that it is a readable
cloud with a coordinate reference system, reprojects it to the domain CRS when
necessary, and stores it as LAZ in POINT_CLOUDS_BUCKET. Unlike raster grids —
where reprojection means resampling and is therefore rejected on a CRS
mismatch — point reprojection is an exact per-point coordinate transform, so
mismatched uploads are transformed rather than refused. The stored cloud is
always in the domain CRS.

Everything streams: the staged upload is read from GCS in bounded chunks
(laspy + gcsfs), and rewritten output is built in an in-memory buffer (LAZ
writers need a seekable target, which GCS is not). No local disk is used —
on Cloud Run the filesystem is RAM-backed, and per-instance memory is the
budget this handler is designed around. Peak usage is bounded by the upload
size cap, not the point count.

A compressed upload already in the domain CRS is copied server-side without
rewriting. Stored format is LAZ, not COPC: every maintained COPC writer needs
the native PDAL stack plus scratch space ~8x the compressed input, which is
RAM on Cloud Run. COPC is planned as a lossless LAZ -> COPC batch upgrade once
that build can run on real disk (see the service README).
"""

import io
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

_OUTPUT_FILENAME = "cloud.laz"
# ~2M points/chunk keeps the streaming passes around 100 MB of working memory.
_CHUNK_POINTS = 2_000_000


def handle_point_cloud(
    resource_id: str, bucket: str, object_name: str, doc: dict
) -> None:
    """Ingest an uploaded point cloud and store it as LAZ in the domain CRS.

    Args:
        resource_id: Point cloud document ID in Firestore.
        bucket: GCS bucket holding the staged upload (UPLOADS_BUCKET).
        object_name: Full GCS object path, e.g. "pointclouds/{id}/upload".
        doc: Point cloud document loaded from Firestore.
    """
    src = f"{bucket}/{object_name}"
    dest = f"{POINT_CLOUDS_BUCKET}/{resource_id}/{_OUTPUT_FILENAME}"

    try:
        domain_crs_name = _domain_crs_name(doc["domain_id"])
        domain_crs = PyprojCRS.from_user_input(domain_crs_name)

        with get_gcsfs_client().open(src, "rb") as stream:
            with _open_cloud(stream) as reader:
                src_crs = _require_crs(reader.header)
                if src_crs.equals(domain_crs, ignore_axis_order=True):
                    if reader.header.are_points_compressed:
                        # Already LAZ in the domain CRS: census the points,
                        # then copy the bytes server-side — no rewrite.
                        stats = _census(reader)
                        bounds = [*reader.header.mins, *reader.header.maxs]
                    else:
                        # Uncompressed LAS: recompress to LAZ, same coords.
                        buf, stats, bounds = _rewrite(reader, domain_crs, None)
                else:
                    # CRS mismatch: exact per-point transform to the domain
                    # CRS (horizontal only — elevations pass through).
                    transformer = Transformer.from_crs(
                        src_crs, domain_crs, always_xy=True
                    )
                    buf, stats, bounds = _rewrite(reader, domain_crs, transformer)

        if stats["rewritten"]:
            with get_gcsfs_client().open(dest, "wb") as out:
                out.write(buf.getbuffer())
        else:
            get_gcsfs_client().copy(src, dest)

        update_document(
            POINT_CLOUDS_COLLECTION,
            resource_id,
            {
                "status": "completed",
                "modified_on": datetime.now(UTC),
                "georeference": {"crs": domain_crs_name, "bounds": bounds},
                "summary": {
                    "point_count": stats["point_count"],
                    "point_classes": stats["point_classes"],
                    "density": stats["density"],
                },
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


def _census(reader: laspy.LasReader) -> dict:
    """Single chunked pass over a reader: point count and classification set.

    Density is points per square meter over the header's horizontal extent.
    """
    classes: set[int] = set()
    count = 0
    for points in reader.chunk_iterator(_CHUNK_POINTS):
        classes |= set(np.unique(np.asarray(points.classification)).tolist())
        count += len(points)

    header = reader.header
    area = (header.maxs[0] - header.mins[0]) * (header.maxs[1] - header.mins[1])
    return {
        "rewritten": False,
        "point_count": count,
        "point_classes": sorted(int(c) for c in classes),
        "density": (count / area) if area > 0 else 0.0,
    }


def _rewrite(
    reader: laspy.LasReader,
    dst_crs: PyprojCRS,
    transformer: Transformer | None,
) -> tuple[io.BytesIO, dict, list[float]]:
    """Stream a cloud into an in-memory LAZ, optionally reprojecting it.

    Reads in bounded chunks, transforms X/Y when a transformer is given
    (elevations pass through untouched), and writes compressed LAZ to a
    seekable in-memory buffer. Statistics and bounds are computed from the
    written (output-CRS) coordinates, so what is reported is what was stored.

    Args:
        reader: Open LAS/LAZ reader positioned at the start of the points.
        dst_crs: CRS recorded on the output header (the domain CRS).
        transformer: Coordinate transform to apply, or None to keep coords.

    Returns:
        (buffer, stats dict, [minx, miny, minz, maxx, maxy, maxz]).
    """
    src_header = reader.header
    header = laspy.LasHeader(
        version=src_header.version, point_format=src_header.point_format
    )
    header.scales = src_header.scales
    # Offsets must be near the data and fixed before writing; the transformed
    # header minimum is exact for transformer=None and close enough otherwise.
    if transformer is not None:
        ox, oy = transformer.transform(src_header.mins[0], src_header.mins[1])
    else:
        ox, oy = src_header.mins[0], src_header.mins[1]
    header.offsets = [ox, oy, src_header.mins[2]]
    header.add_crs(dst_crs)

    classes: set[int] = set()
    count = 0
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])

    buf = io.BytesIO()
    with laspy.open(
        buf, mode="w", header=header, do_compress=True, closefd=False
    ) as writer:
        for points in reader.chunk_iterator(_CHUNK_POINTS):
            x = np.asarray(points.x)
            y = np.asarray(points.y)
            if transformer is not None:
                x, y = transformer.transform(x, y)
            points.change_scaling(scales=header.scales, offsets=header.offsets)
            points.x = x
            points.y = y

            z = np.asarray(points.z)
            mins = np.minimum(mins, [x.min(), y.min(), z.min()])
            maxs = np.maximum(maxs, [x.max(), y.max(), z.max()])
            classes |= set(np.unique(np.asarray(points.classification)).tolist())
            count += len(points)
            writer.write_points(points)

    if count == 0:
        mins = np.array(header.offsets, dtype=float)
        maxs = np.array(header.offsets, dtype=float)

    area = (maxs[0] - mins[0]) * (maxs[1] - mins[1])
    stats = {
        "rewritten": True,
        "point_count": count,
        "point_classes": sorted(int(c) for c in classes),
        "density": (count / area) if area > 0 else 0.0,
    }
    buf.seek(0)
    return buf, stats, [*mins.tolist(), *maxs.tolist()]
