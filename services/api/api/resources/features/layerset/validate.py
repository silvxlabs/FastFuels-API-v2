"""
api/v2/resources/features/layerset/validate.py

Single, geopandas-backed validation helper for layerset uploads.

Mirrors ``api.resources.domains.validate.validate_domain``: the router hands
the raw GeoJSON dict to ``validate_layerset`` and gets back a GeoDataFrame
(ready to write to GeoParquet), the parsed CRS, and the union bounds in one
pass — instead of the router hand-rolling CRS parsing, a projected-CRS check,
and a per-feature bounds loop.

Unlike ``validate_domain``, this stays GDAL-free: the GeoDataFrame is built
with ``gpd.GeoDataFrame.from_features`` (pure shapely + pandas), never
``gpd.read_file`` (fiona/pyogrio → GDAL). The API service must not pull GDAL
at runtime; see the service-boundary note in the repo CLAUDE.md.
"""

import logging
import math

import geopandas as gpd
import pyproj
from fastapi import HTTPException, status
from pyproj.exceptions import CRSError

logger = logging.getLogger(__name__)

# GeoJSON spec default when no crs block is declared on the FeatureCollection.
_DEFAULT_CRS = "EPSG:4326"


class LayersetValidationResult:
    """Processed result of a layerset upload validation.

    Attributes:
        gdf: The features as a GeoDataFrame, tagged with ``crs`` and ready to
            write to GeoParquet via ``gdf.to_parquet``.
        crs: The parsed, projected pyproj CRS.
        crs_string: Canonical ``"EPSG:<code>"`` form of ``crs``, suitable for
            the stored ``FeatureGeoreference.crs``. Falls back to ``crs.srs``
            when the CRS has no EPSG authority.
        bounds: The ``(minx, miny, maxx, maxy)`` union bounding box across
            every feature geometry, in the layerset's own CRS. ``None`` when
            no feature carries a non-empty geometry.
    """

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        crs: pyproj.CRS,
        crs_string: str,
        bounds: tuple[float, float, float, float] | None,
    ):
        self.gdf = gdf
        self.crs = crs
        self.crs_string = crs_string
        self.bounds = bounds


def validate_layerset(geojson: dict) -> LayersetValidationResult:
    """Parse, CRS-check, and compute bounds for a layerset upload.

    1. Parse the CRS from ``geojson["crs"]["properties"]["name"]``, defaulting
       to ``EPSG:4326`` (the GeoJSON spec default when no crs block is given).
       Both bare ``"EPSG:32612"`` and the URN form
       ``"urn:ogc:def:crs:EPSG::32612"`` are accepted via
       ``pyproj.CRS.from_user_input``.
    2. Reject a geographic CRS. ``fastfuels_core.rasterize_layerset`` rejects
       geographic CRSes at rasterize time (resolution would be in degrees, not
       meters); validating here turns a deferred worker crash into an
       immediate, actionable ``422`` for the caller.
    3. Build a GeoDataFrame from ``geojson["features"]`` tagged with the parsed
       CRS, ready to write to GeoParquet.
    4. Compute the union bounds via ``gdf.total_bounds`` (``None`` when every
       geometry is empty).

    Args:
        geojson: A GeoJSON FeatureCollection dict.

    Returns:
        A :class:`LayersetValidationResult`.

    Raises:
        HTTPException: 422 if the CRS is unparseable or geographic.
    """
    crs_name = (geojson.get("crs") or {}).get("properties", {}).get(
        "name"
    ) or _DEFAULT_CRS

    try:
        crs = pyproj.CRS.from_user_input(crs_name)
    except CRSError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Could not parse CRS {crs_name!r}: {exc}. Declare a "
                "projected CRS on the GeoJSON's top-level `crs` block "
                "(e.g. `EPSG:32612` for UTM 12N)."
            ),
        ) from exc

    if crs.is_geographic:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Layerset CRS is geographic ({crs_name}). Rasterization "
                "requires a projected CRS so cell sizes are in meters. "
                "Reproject the GeoJSON to a UTM (or other projected) CRS "
                "and declare it on the FeatureCollection's `crs` block."
            ),
        )

    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs=crs)

    # Normalize to the canonical "EPSG:<code>" form. pyproj's str(crs)/.srs
    # echoes the input (e.g. the URN), but the stored georeference and the
    # rest of the API speak "EPSG:<code>".
    authority = crs.to_authority()
    crs_string = ":".join(authority) if authority else crs.srs

    # gdf.total_bounds is [nan, nan, nan, nan] when every geometry is empty.
    minx, miny, maxx, maxy = (float(v) for v in gdf.total_bounds)
    bounds = None if math.isnan(minx) else (minx, miny, maxx, maxy)

    return LayersetValidationResult(
        gdf=gdf,
        crs=crs,
        crs_string=crs_string,
        bounds=bounds,
    )
