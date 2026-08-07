"""
LANDFIRE coverage pre-flight check.

Before creating a grid from a LANDFIRE product, checks whether a domain
falls inside LANDFIRE's currently-covered area for the requested
product/version (annual products) or season (Seasonal Fuels) -- so a
request outside coverage fails fast and clearly and a domain straddling
covered and uncovered ground is flagged as partial rather than treated
as fully covered.

GeoArea boundaries and Seasonal Fuels boundaries are two different
datasets. GeoAreas are named regions (SW, NW, ...) that tile all of
CONUS; annual-product coverage depends on which GeoAreas LANDFIRE
Product Service currently serves for a given product/version, checked
live via `list_products()` (a product served CONUS-wide skips the GeoArea
check entirely). Seasonal Fuels data comes as one polygon per season.

Both boundary sets are loaded from a data file at import time, in
EPSG:5070 -- LANDFIRE's native CRS. Callers must pass `geometry` already
reprojected to EPSG:5070; this module does no reprojection of its own since
a bare shapely geometry carries no CRS to reproject from.

This module only matters for the LFPS on-demand path. Products/versions
already pre-staged as CONUS-wide COGs are fetched directly from GCS and
are covered by definition -- there's no GeoArea gap to check.
"""

from __future__ import annotations

from enum import StrEnum

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from lib.config import RASTERS_BUCKET
from lib.landfire_lfps import list_products

_BOUNDARY_CRS_EPSG = 5070


def _load_boundaries(filename: str, key: str) -> dict[str, BaseGeometry]:
    """Load a GeoParquet boundary file into a {key property value: geometry} dict.

    Reprojects to EPSG:5070 if the source CRS differs.
    """
    gdf = gpd.read_parquet(f"gs://{RASTERS_BUCKET}/{filename}")
    if gdf.crs is not None and gdf.crs.to_epsg() != _BOUNDARY_CRS_EPSG:
        gdf = gdf.to_crs(epsg=_BOUNDARY_CRS_EPSG)
    return dict(zip(gdf[key], gdf.geometry))


GEOAREA_BOUNDARIES: dict[str, BaseGeometry] = _load_boundaries(
    "LF_GeoAreas.parquet", key="geoArea"
)

SEASONAL_BOUNDARIES: dict[str, BaseGeometry] = _load_boundaries(
    "LF_SeasonalRegions.parquet", key="season"
)


class CoverageStatus(StrEnum):
    """How much of a geometry falls inside a covered area.

    NO_SUCH_PRODUCT is distinct from NONE: NONE means LFPS serves this
    product/version (or product/version/season), just not where the
    geometry is; NO_SUCH_PRODUCT means LFPS isn't serving it anywhere at
    all right now (e.g. a season that hasn't been published yet).
    """

    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"
    NO_SUCH_PRODUCT = "no_such_product"


def covers_annual(product: str, version: str, geometry: BaseGeometry) -> CoverageStatus:
    """Check LFPS coverage for an annual product/version against a geometry.

    Args:
        product: Product acronym, matched case-insensitively against
            `LfpsProduct.acronym` (e.g. "fbfm40" matches "FBFM40").
        version: LANDFIRE version year (e.g. "2025") -- matched against
            LFPS's "LF{version}" format.
        geometry: Domain geometry, already reprojected to EPSG:5070.

    Returns:
        CoverageStatus. NO_SUCH_PRODUCT if no non-Seasonal-Fuels LFPS
        entry matches product/version at all. FULL if `geo_areas` is
        "All", or if every GeoArea the geometry touches is served. NONE
        if it touches no served GeoArea. PARTIAL otherwise.
    """
    matching_product = next(
        (
            p
            for p in list_products()
            if p.theme != "Seasonal Fuels"
            and p.acronym.lower() == product.lower()
            and p.version == f"LF{version}"
        ),
        None,
    )
    if matching_product is None:
        return CoverageStatus.NO_SUCH_PRODUCT

    if matching_product.geo_areas.strip() == "All":
        return CoverageStatus.FULL

    covered_geoareas = {area.strip() for area in matching_product.geo_areas.split(",")}

    remaining = geometry
    touched_covered = False
    for code, boundary in GEOAREA_BOUNDARIES.items():
        if not boundary.intersects(geometry):
            continue
        if code in covered_geoareas:
            touched_covered = True
            if not remaining.is_empty:
                remaining = remaining.difference(boundary)

    if not touched_covered:
        return CoverageStatus.NONE
    return CoverageStatus.FULL if remaining.is_empty else CoverageStatus.PARTIAL


def covers_seasonal(
    product: str, version: str, season: str, geometry: BaseGeometry
) -> CoverageStatus:
    """Check Seasonal Fuels coverage for a product/version/season against a geometry.

    Not every season is necessarily live at a given time -- e.g. fall may
    not yet be published -- so live availability is checked first, via
    `LfpsProduct.season`, before geometry is even looked at.

    Args:
        product: Product acronym, matched case-insensitively against
            `LfpsProduct.acronym` (e.g. "fbfm40" matches "FBFM40").
        version: LANDFIRE version year (e.g. "2025") -- matched against
            LFPS's "LF{version}" format.
        season: A key in SEASONAL_BOUNDARIES (e.g. "SP").
        geometry: Domain geometry, already reprojected to EPSG:5070.

    Returns:
        CoverageStatus. NO_SUCH_PRODUCT if `season` isn't currently live
        for `product`/`version`. Otherwise FULL, PARTIAL, or NONE.
    """
    is_live = any(
        p.season == season
        and p.acronym.lower() == product.lower()
        and p.version == f"LF{version}"
        for p in list_products()
    )
    if not is_live:
        return CoverageStatus.NO_SUCH_PRODUCT

    boundary = SEASONAL_BOUNDARIES[season]
    if not boundary.intersects(geometry):
        return CoverageStatus.NONE
    remaining = geometry.difference(boundary)
    return CoverageStatus.FULL if remaining.is_empty else CoverageStatus.PARTIAL
