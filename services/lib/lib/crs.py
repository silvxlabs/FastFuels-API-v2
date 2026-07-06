"""Semantic CRS equality.

Equivalent CRS spellings — e.g. ``EPSG:5070`` and ``urn:ogc:def:crs:EPSG::5070``
— denote the same coordinate reference system but differ textually, so a raw
string ``==`` wrongly reports a mismatch. Every CRS-equality check that gates a
rejection (upload, export, compose) should route through :func:`crs_equal`.

Imports only ``pyproj`` (which bundles PROJ, not GDAL), so this module is safe to
import from the GDAL-free API service.
"""

from pyproj import CRS


def crs_equal(a: str | None, b: str | None) -> bool:
    """True if two CRS identifiers denote the same coordinate reference system.

    Compares semantically via pyproj so equivalent spellings match — e.g. a grid
    georeference stored as ``EPSG:5070`` and a domain CRS stored in OGC URN form
    ``urn:ogc:def:crs:EPSG::5070`` are the same CRS. The None branch just guards a
    missing CRS (e.g. ``geo.get("crs")`` → None).
    """
    if a is None or b is None:
        return a == b
    return CRS.from_user_input(a) == CRS.from_user_input(b)
