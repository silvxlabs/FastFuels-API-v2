"""Comparing coordinate reference systems, and moving coordinates between them.

Equivalent CRS spellings — e.g. ``EPSG:5070`` and ``urn:ogc:def:crs:EPSG::5070``
— denote the same coordinate reference system but differ textually, so a raw
string ``==`` wrongly reports a mismatch. Every CRS-equality check that gates a
rejection (upload, export, compose) should route through :func:`crs_equal`.

:func:`reproject` is for the bulk case, where the cost of PROJ per point starts
to matter: a point cloud job moves hundreds of millions of them.

Imports only ``pyproj`` (which bundles PROJ, not GDAL) and ``numpy``, so this
module is safe to import from the GDAL-free API service.
"""

import numpy as np
from pyproj import CRS

# Cells per axis of the grid the fit is sampled on. Six unknowns against 25
# points, so the fit is comfortably overdetermined.
_FIT_GRID = 5

# How far the fit may sit from PROJ before it is thrown away, in target units.
# Point clouds quantise to `lib.laz.CANONICAL_SCALE`, 0.001 m, so a coordinate
# already moves up to 0.0005 m when it is stored. This sits five times inside
# that, which keeps the approximation below the noise the storage format adds
# rather than merely near it.
DEFAULT_TOLERANCE = 1e-4


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


def _apply(coeffs, dx, dy):
    """Evaluate both fitted quadratics on coordinates already centred on the box."""
    (cx, cy, mx, my) = coeffs
    xx, xy, yy = dx * dx, dx * dy, dy * dy
    return (
        mx + cx[0] + cx[1] * dx + cx[2] * dy + cx[3] * xx + cx[4] * xy + cx[5] * yy,
        my + cy[0] + cy[1] * dx + cy[2] * dy + cy[3] * xx + cy[4] * xy + cy[5] * yy,
    )


def _fit(transformer, gx, gy, x0, y0):
    """Least-squares quadratics for the map over a box, or None if PROJ balked."""
    tx, ty = transformer.transform(gx + x0, gy + y0)
    if not (np.isfinite(tx).all() and np.isfinite(ty).all()):
        return None
    # Both sides are centred first. On raw projected coordinates x**2 is ~1e14
    # and the normal equations lose every digit that matters — fitted that way,
    # the quadratic scores worse than a plane.
    design = np.stack([np.ones(gx.size), gx, gy, gx * gx, gx * gy, gy * gy], axis=1)
    mx, my = tx.mean(), ty.mean()
    cx = np.linalg.lstsq(design, tx - mx, rcond=None)[0]
    cy = np.linalg.lstsq(design, ty - my, rcond=None)[0]
    return cx, cy, mx, my


def reproject(transformer, x, y, *, tolerance=DEFAULT_TOLERANCE):
    """Reproject coordinate arrays, approximating the map where that stays exact.

    A projected-to-projected map is smooth, so across a bounded box a quadratic
    matches it very closely. Fitting one on a coarse grid and evaluating it per
    point measured 5.7x faster than putting every point through PROJ — 88 against
    15 M points/s — which is worth having when a job moves hundreds of millions.
    GDAL warps the same way, in ``GDALApproxTransform``.

    The fit is checked against PROJ at points it was not fitted on, and the exact
    transform is returned whenever the check fails, so `tolerance` is a guarantee
    rather than an assumption. Boxes wide enough to fail it do exist: the error
    grows with the cube of the span, passing the default near 13 km. EPT nodes
    are far smaller than that, but the shallow ones are not.

    Args:
        transformer: A pyproj ``Transformer``, built with ``always_xy=True``.
        x: Source x coordinates, as a float64 array.
        y: Source y coordinates, as a float64 array.
        tolerance: Largest positional error to accept, in the target's units.

    Returns:
        ``(x, y)`` in the transformer's target CRS.
    """
    if x.size == 0:
        return transformer.transform(x, y)

    x0 = (float(x.min()) + float(x.max())) / 2.0
    y0 = (float(y.min()) + float(y.max())) / 2.0
    # One square covering the whole bounding box, so no point extrapolates.
    half = max(float(x.max()) - x0, float(y.max()) - y0)
    if not np.isfinite(half) or half <= 0.0:
        # A single point, every point in one place, or a NaN in the input.
        return transformer.transform(x, y)

    axis = np.linspace(-half, half, _FIT_GRID)
    gx, gy = (a.ravel() for a in np.meshgrid(axis, axis))
    coeffs = _fit(transformer, gx, gy, x0, y0)
    if coeffs is None:
        return transformer.transform(x, y)

    # Checked on a denser grid than the fit used, so the samples mostly fall
    # between the nodes the fit was constrained at — testing it only where it was
    # fitted would measure lstsq rather than the approximation. The grid has to
    # reach the boundary: a least-squares residual peaks near the edges, and an
    # interior-only check passed a 16 km box whose true error was 1.8x tolerance.
    check = np.linspace(-half, half, _FIT_GRID * 2)
    hx, hy = (a.ravel() for a in np.meshgrid(check, check))
    ex, ey = transformer.transform(hx + x0, hy + y0)
    px, py = _apply(coeffs, hx, hy)
    if not np.isfinite(ex).all() or np.hypot(px - ex, py - ey).max() > tolerance:
        return transformer.transform(x, y)

    return _apply(coeffs, x - x0, y - y0)
