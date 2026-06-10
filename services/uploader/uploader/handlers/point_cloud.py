"""
Point cloud upload handler for the Uploader service.

Ingests an uploaded point cloud (LAS / LAZ / COPC), validates it is a readable
cloud whose CRS matches the domain CRS, extracts metadata, and stores a Cloud
Optimized Point Cloud (COPC) in POINT_CLOUDS_BUCKET. As in the grid upload
handler, the file must already be in the domain's CRS — every resource in a
domain shares one CRS — so a mismatch is rejected rather than reprojected.

Reads the staged file directly from GCS over GDAL's ``/vsigs`` (no full
download). LAS/LAZ inputs are transcoded to COPC in a single PDAL pass; the COPC
writer needs a seekable local file, so it writes to an ephemeral-disk scratch
mount (real disk, off the instance memory budget — Cloud Run's ``/tmp`` is
RAM-backed) before the result is uploaded to GCS. A file already in COPC form is
copied through server-side without a rebuild.
"""

import json
import os
from datetime import UTC, datetime

import pdal
from pyproj import CRS as PyprojCRS
from pyproj.exceptions import CRSError

from lib.config import (
    DOMAINS_COLLECTION,
    POINT_CLOUDS_BUCKET,
    POINT_CLOUDS_COLLECTION,
)
from lib.errors import ProcessingError
from lib.firestore import get_document, update_document
from lib.gcs import delete_file, gcsfs_client, upload_file

# Ephemeral-disk scratch mount configured on the uploader Cloud Run service.
# Overridable for local/CI runs that lack the mount. Must NOT default to /tmp:
# /tmp is RAM-backed and would put the COPC build back on the memory budget.
_SCRATCH_DIR = os.environ.get("POINT_CLOUD_SCRATCH_DIR", "/scratch")

_OUTPUT_FILENAME = "cloud.copc.laz"


def handle_point_cloud(
    resource_id: str, bucket: str, object_name: str, doc: dict
) -> None:
    """Ingest an uploaded point cloud and store it as COPC.

    Args:
        resource_id: Point cloud document ID in Firestore.
        bucket: GCS bucket holding the staged upload (UPLOADS_BUCKET).
        object_name: Full GCS object path, e.g. "pointclouds/{id}/upload.laz".
        doc: Point cloud document loaded from Firestore.
    """
    fmt = (doc.get("source") or {}).get("format")
    input_path = f"/vsigs/{bucket}/{object_name}"
    dest = f"gs://{POINT_CLOUDS_BUCKET}/{resource_id}/{_OUTPUT_FILENAME}"

    scratch_path = None
    try:
        domain_crs = _domain_crs(doc["domain_id"])
        if fmt == "copc":
            # Already cloud-optimized: inspect metadata, then (after the CRS
            # check) copy the bytes through server-side — no rebuild, no scratch.
            stats = _inspect_and_transcode(input_path, None)
            _require_crs_match(stats["crs"], domain_crs)
            gcsfs_client.copy(
                f"{bucket}/{object_name}",
                f"{POINT_CLOUDS_BUCKET}/{resource_id}/{_OUTPUT_FILENAME}",
            )
        else:
            # Transcode LAS/LAZ -> COPC on ephemeral-disk scratch, validate the
            # CRS, then upload. A mismatch discards the scratch build (cleaned up
            # below) and never reaches POINT_CLOUDS_BUCKET.
            os.makedirs(_SCRATCH_DIR, exist_ok=True)
            scratch_path = os.path.join(_SCRATCH_DIR, f"{resource_id}.copc.laz")
            stats = _inspect_and_transcode(input_path, scratch_path)
            _require_crs_match(stats["crs"], domain_crs)
            upload_file(scratch_path, dest)

        update_document(
            POINT_CLOUDS_COLLECTION,
            resource_id,
            {
                "status": "completed",
                "modified_on": datetime.now(UTC),
                "georeference": {"crs": stats["crs"], "bounds": stats["bounds"]},
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
        if scratch_path and os.path.exists(scratch_path):
            try:
                os.remove(scratch_path)
            except OSError:
                pass


def _domain_crs(domain_id: str) -> str:
    """Return the domain's CRS as an EPSG authority string (e.g. 'EPSG:32612')."""
    _, snapshot = get_document(DOMAINS_COLLECTION, domain_id)
    domain = snapshot.to_dict()
    crs = domain.get("crs")
    if isinstance(crs, dict):
        return crs["properties"]["name"]
    return crs or "EPSG:4326"


def _require_crs_match(cloud_crs: str, domain_crs: str) -> None:
    """Reject a point cloud whose CRS does not match its domain's CRS.

    In v2 every resource in a domain shares the domain's CRS, so an upload must
    already be in that CRS (mirrors the grid upload handler — no reprojection).

    Raises:
        ProcessingError: CRS_MISMATCH when the codes differ.
    """
    if cloud_crs != domain_crs:
        raise ProcessingError(
            code="CRS_MISMATCH",
            message=(
                f"Point cloud CRS ({cloud_crs}) does not match the domain CRS "
                f"({domain_crs}). Reproject the file to the domain CRS before "
                "uploading."
            ),
            suggestion=(
                "Reproject with PDAL, e.g. `pdal translate in.laz out.laz "
                f"reprojection --filters.reprojection.out_srs={domain_crs}`."
            ),
        )


def _inspect_and_transcode(input_path: str, output_path: str | None) -> dict:
    """Read a point cloud, extract metadata, and optionally write COPC.

    Runs a single PDAL pass — ``readers.las`` -> ``filters.stats`` (-> a COPC
    writer when ``output_path`` is given). Pure: touches no GCS or Firestore, so
    it is directly unit-testable on local files. ``input_path`` / ``output_path``
    may be local paths or GDAL ``/vsi`` paths.

    Args:
        input_path: Source point cloud (local or ``/vsigs`` path).
        output_path: Where to write the COPC, or ``None`` to inspect only.

    Returns:
        ``{"crs", "bounds", "point_count", "point_classes", "density"}``.

    Raises:
        ProcessingError: UNREADABLE_POINT_CLOUD if the file cannot be read;
            MISSING_CRS / UNRESOLVABLE_CRS if it carries no usable CRS.
    """
    stages: list[dict] = [
        {"type": "readers.las", "filename": input_path},
        {
            "type": "filters.stats",
            "dimensions": "X,Y,Z,Classification",
            # Enumerate per-value counts for Classification so we can report the
            # exact set of ASPRS classes present.
            "count": "Classification",
        },
    ]
    if output_path is not None:
        stages.append(
            {"type": "writers.copc", "filename": output_path, "forward": "all"}
        )

    pipeline = pdal.Pipeline(json.dumps(stages))
    try:
        point_count = pipeline.execute()
    except RuntimeError as e:
        raise ProcessingError(
            code="UNREADABLE_POINT_CLOUD",
            message="The uploaded file could not be read as a point cloud.",
            suggestion="Upload a valid LAS, LAZ, or COPC file.",
            traceback=str(e),
        )

    metadata = pipeline.metadata
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    meta = metadata["metadata"]

    las = meta["readers.las"]
    if isinstance(las, list):
        las = las[0]

    srs = las.get("srs") or {}
    wkt = ""
    if isinstance(srs, dict):
        wkt = srs.get("wkt") or srs.get("compoundwkt") or ""
    crs = _wkt_to_epsg(wkt)

    bounds = [
        float(las["minx"]),
        float(las["miny"]),
        float(las["minz"]),
        float(las["maxx"]),
        float(las["maxy"]),
        float(las["maxz"]),
    ]
    point_classes = _classification_set(meta)
    area = (bounds[3] - bounds[0]) * (bounds[4] - bounds[1])
    density = (point_count / area) if area > 0 else 0.0

    return {
        "crs": crs,
        "bounds": bounds,
        "point_count": int(point_count),
        "point_classes": point_classes,
        "density": float(density),
    }


def _classification_set(meta: dict) -> list[int]:
    """Extract the sorted set of ASPRS Classification codes from stats metadata."""
    stats = meta.get("filters.stats", {})
    if isinstance(stats, list):
        stats = stats[0]
    for dim in stats.get("statistic", []):
        if dim.get("name") != "Classification":
            continue
        counts = dim.get("counts")
        if counts:
            values = set()
            for entry in counts:
                if isinstance(entry, dict):
                    values.add(int(round(float(entry["value"]))))
                else:
                    # Some PDAL builds emit "value/count" strings.
                    values.add(int(round(float(str(entry).split("/")[0]))))
            return sorted(values)
        # Fallback if per-value counts are absent: the endpoints only.
        lo = int(round(float(dim.get("minimum", 0))))
        hi = int(round(float(dim.get("maximum", lo))))
        return sorted({lo, hi})
    return []


def _wkt_to_epsg(wkt: str) -> str:
    """Resolve a PDAL SRS WKT string to an ``EPSG:xxxxx`` authority code.

    Compound CRSs (horizontal + vertical) resolve to their horizontal component,
    since a point cloud's georeference records a single horizontal+vertical CRS
    code.

    Raises:
        ProcessingError: MISSING_CRS if no CRS is present; UNRESOLVABLE_CRS if a
            CRS is present but maps to no EPSG code.
    """
    if not wkt or not wkt.strip():
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
    try:
        crs = PyprojCRS.from_wkt(wkt)
    except CRSError as e:
        raise ProcessingError(
            code="UNRESOLVABLE_CRS",
            message="The point cloud's coordinate reference system could not be parsed.",
            traceback=str(e),
        )

    if crs.is_compound:
        crs = crs.sub_crs_list[0]

    epsg = crs.to_epsg()
    if epsg is None:
        auth = crs.to_authority(min_confidence=70)
        if auth and auth[0] == "EPSG":
            epsg = int(auth[1])
    if epsg is None:
        raise ProcessingError(
            code="UNRESOLVABLE_CRS",
            message=(
                "The point cloud's CRS does not correspond to a known EPSG code. "
                "Reproject to a standard projected CRS (e.g. the appropriate UTM "
                "zone) before uploading."
            ),
        )
    return f"EPSG:{epsg}"
