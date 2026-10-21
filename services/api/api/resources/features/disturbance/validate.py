"""
api/v2/resources/features/disturbance/validate.py

Geopandas-backed validation helper for disturbance uploads.

Like ``validate_layerset``, this stays GDAL-free: the GeoDataFrame is built with
``gpd.GeoDataFrame.from_features`` (pure shapely + pandas) and written with
pyarrow. Unlike layerset, any CRS is accepted: the polygons are only reprojected
to the grid they are used with, so cell sizes in meters aren't needed here.
"""

import geopandas as gpd
import pyproj
from fastapi import HTTPException, status
from pyproj.exceptions import CRSError

from lib.domain_utils import parse_domain_gdf

# GeoJSON spec default when no crs block is declared on the FeatureCollection.
_DEFAULT_CRS = "EPSG:4326"


class DisturbanceValidationResult:
    """Processed result of a disturbance upload validation.

    Attributes:
        gdf: The polygons as a GeoDataFrame tagged with ``crs``, with one
            ``fdist`` column, ready to write to GeoParquet.
        crs: The parsed pyproj CRS of the upload.
        crs_string: Canonical ``"EPSG:<code>"`` form of ``crs``, falling back to
            ``crs.srs`` when the CRS has no EPSG authority.
        bounds: ``(minx, miny, maxx, maxy)`` of every polygon, in the upload's
            own CRS.
    """

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        crs: pyproj.CRS,
        crs_string: str,
        bounds: tuple[float, float, float, float],
    ):
        self.gdf = gdf
        self.crs = crs
        self.crs_string = crs_string
        self.bounds = bounds


def validate_disturbance(geojson: dict, domain: dict) -> DisturbanceValidationResult:
    """Parse the CRS, build the GeoDataFrame, and check it overlaps the domain.

    The CRS comes from ``geojson["crs"]["properties"]["name"]``, defaulting to
    ``EPSG:4326`` (the GeoJSON spec default). Both ``"EPSG:32612"`` and the URN
    form are accepted.

    Args:
        geojson: A GeoJSON FeatureCollection dict.
        domain: The domain document the polygons are uploaded to.

    Raises:
        HTTPException: 422 if the CRS is unparseable, or if no polygon
            intersects the domain.
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
                f"Could not parse CRS {crs_name!r}: {exc}. Declare a valid CRS "
                "on the GeoJSON's top-level `crs` block (e.g. `EPSG:4326` or "
                "`EPSG:32612`)."
            ),
        ) from exc

    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs=crs)
    if "year" in gdf:
        gdf["year"] = gdf["year"].astype("Int64")

    _require_overlap_with_domain(gdf, domain)

    authority = crs.to_authority()
    crs_string = ":".join(authority) if authority else crs.srs
    bounds = tuple(float(v) for v in gdf.total_bounds)

    return DisturbanceValidationResult(
        gdf=gdf, crs=crs, crs_string=crs_string, bounds=bounds
    )


def _require_overlap_with_domain(gdf: gpd.GeoDataFrame, domain: dict) -> None:
    """422 unless at least one polygon intersects the domain."""
    domain_geometry = parse_domain_gdf(domain).to_crs(gdf.crs).union_all()
    if not gdf.intersects(domain_geometry).any():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "None of the polygons intersect the domain. Check that they are "
                "in the declared CRS and cover the domain's area."
            ),
        )
