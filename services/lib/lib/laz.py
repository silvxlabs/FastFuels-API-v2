"""
Canonical point schema for point cloud workers.

Every point v2 stores is normalized through here, so that a cloud assembled
from many sources looks exactly like a cloud rewritten from one upload. Workers
supply points; this module owns the schema they are converted to. Storage is
Parquet (`lib.pointcloud.writer`) — the header built here is a conversion
target, never written to a file.

The canonical schema is **LAS 1.4, point format 6, with no extra dimensions**.
Point format 6 is a lossless superset of the formats USGS and most vendors
publish (0-5): classification widens from 5 bits to a full byte, return numbers
widen, and GPS time is always present. Normalizing on it means a single write
path regardless of what came in, and — more importantly — it is the only way to
merge points from two sources at all: laspy's record conversion compares point
formats and extra dimensions, so even two point-format-1 sources disagree if
one carries an extra dimension the other lacks.
"""

import math
from collections.abc import Iterable

import laspy
import numpy as np
from laspy.header import GpsTimeType
from laspy.point.record import PackedPointRecord, ScaleAwarePointRecord

CANONICAL_VERSION = "1.4"
CANONICAL_POINT_FORMAT_ID = 6

# Point format 6 carries no colour. Sources that do are promoted so a merge
# never silently discards it: 7 adds RGB to 6, and 8 adds near-infrared to 7.
_POINT_FORMAT_WITH_RGB = 7
_POINT_FORMAT_WITH_NIR = 8

# Millimetre quantization. At int32 this reaches +/- 2147 km from the header
# offset, which no terrestrial coordinate inside a domain can approach, and it
# is finer than the 0.01 m most source files use, so nothing is lost.
CANONICAL_SCALE = 0.001

# Point format 6+ stores the scan angle as an int16 in increments of 0.006
# degrees; formats 0-5 store it as int8 degrees.
_SCAN_ANGLE_INCREMENT = 0.006


def point_format_id_for_dimensions(dimension_names: Iterable[str]) -> int:
    """Return the canonical point format that preserves the named dimensions.

    Point format 6 is the floor. Colour and near-infrared are promoted rather
    than silently flattened, since dropping either is data loss that only shows
    up long after the fact.

    Names are matched case-insensitively so a format description can come from
    laspy (``red``) or from a file's own schema (``Red``).

    Args:
        dimension_names: Dimension names across every source being merged.

    Returns:
        6, 7, or 8.
    """
    names = {name.lower() for name in dimension_names}
    if names & {"nir", "infrared", "near_infrared"}:
        return _POINT_FORMAT_WITH_NIR
    if "red" in names:
        return _POINT_FORMAT_WITH_RGB
    return CANONICAL_POINT_FORMAT_ID


def build_output_header(
    crs,
    bounds: tuple[float, float, float, float] | tuple[float, ...],
    *,
    point_format_id: int = CANONICAL_POINT_FORMAT_ID,
    gps_standard_time: bool = True,
) -> laspy.LasHeader:
    """Build the canonical output header for a cloud in a known extent.

    This imposes a format: canonical point format, millimetre scaling, and an
    offset anchored to the extent. That is what makes points from several
    acquisitions writable into one file. A caller rewriting a *single* file
    should not use it — reproduce that file's own header instead, or its
    scaling and attributes are silently replaced (see the uploader).

    Scale and offset are fixed up front from the extent rather than derived from
    the data. The header is built in the worker pool initializer, before any
    node has been decoded, so deriving the offset from the points would mean
    buffering them all first. Anchoring to the extent is exact for any point
    inside it and makes the output byte-reproducible for a given domain.

    Args:
        crs: Output CRS, as anything ``laspy.LasHeader.add_crs`` accepts
            (a pyproj CRS).
        bounds: Horizontal extent the points fall inside, as
            ``(min_x, min_y, ...)``. Only the two minima are read.
        point_format_id: Output point format. Defaults to the canonical 6; pass
            the result of ``point_format_id_for_dimensions`` when the sources
            may carry colour.
        gps_standard_time: Whether GPS timestamps are Adjusted Standard GPS
            Time. True for USGS 3DEP and for essentially all modern data. A
            fresh laspy header would otherwise claim GPS Week Time, silently
            misdating every point.

    Returns:
        A LAS 1.4 header in the requested point format, carrying the CRS and
        scaling.
    """
    header = laspy.LasHeader(
        version=CANONICAL_VERSION, point_format=laspy.PointFormat(point_format_id)
    )
    header.scales = [CANONICAL_SCALE] * 3
    header.offsets = [math.floor(bounds[0]), math.floor(bounds[1]), 0.0]
    if gps_standard_time:
        header.global_encoding.gps_time_type = GpsTimeType.STANDARD
    header.add_crs(crs)
    return header


def normalize_record(
    source: PackedPointRecord,
    header: laspy.LasHeader,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> ScaleAwarePointRecord:
    """Convert a source point record into the canonical output format.

    Copies every attribute the output format has and the source can supply, then
    writes the caller's coordinates. Coordinates are passed in rather than read
    from ``source`` because callers reproject: the source record's coordinates
    are in the source CRS, and the output header's offsets are in the output
    CRS.

    Attributes with no counterpart in the output format are dropped. That
    includes ``OriginId``, which EPT files carry: it indexes the source file
    list of one acquisition, so after merging two acquisitions the same value
    means two different flightlines. Provenance belongs on the resource, not on
    the points.

    Args:
        source: Record read from a source file, in any point format.
        header: Canonical header from ``build_output_header``.
        x: Output-CRS x coordinates, one per source point.
        y: Output-CRS y coordinates, one per source point.
        z: Output-CRS z coordinates, one per source point.

    Returns:
        A record in the header's point format and scaling, ready to write.
    """
    out = ScaleAwarePointRecord.zeros(len(source), header=header)
    out.copy_fields_from(source)

    # copy_fields_from matches by dimension name, so the scan angle is dropped
    # silently when the source uses the older name. Convert it explicitly.
    source_names = set(source.point_format.dimension_names)
    if "scan_angle" in header.point_format.dimension_names:
        if "scan_angle" not in source_names and "scan_angle_rank" in source_names:
            rank = np.asarray(source["scan_angle_rank"], dtype=np.float64)
            out["scan_angle"] = np.round(rank / _SCAN_ANGLE_INCREMENT).astype(np.int16)

    # copy_fields_from also copies raw X/Y/Z, which are encoded against the
    # source's scale and offset and are meaningless here. Overwrite through the
    # scaled views, which range-check the conversion and raise rather than
    # silently wrapping past int32.
    out.x = x
    out.y = y
    out.z = z
    return out
