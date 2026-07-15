"""
api/v2/resources/grids/exports/alignment.py

Shared lattice construction and alignment checks for combined grid exports
(QUIC-Fire, landscape).

Combined exports never resample or reproject: every role grid must already be
CRS-equal to the export lattice, at the export cell size, offset by an integer
number of cells, and cover the full export extent. These helpers build the
horizontal export lattice and enforce that contract, raising
`HTTPException(422)` with a remediation pointer on violation.

A lattice dict has keys: `nx`, `ny`, `dx`, `dy`, `transform` (6-element
row-major affine), and `crs`.
"""

from math import ceil, isclose

from fastapi import HTTPException, status

from api.resources.grids.utils import validate_grid_has_georeference
from lib.crs import crs_equal

# 1 µm tolerance for transform-coefficient comparisons.
TOL = 1e-6


def domain_crs_string(domain: dict) -> str | None:
    """Extract the EPSG-style CRS string from the Domain's GeoJSON CRS dict."""
    crs = domain.get("crs")
    if isinstance(crs, dict):
        return crs.get("properties", {}).get("name")
    return crs


def build_domain_lattice(domain: dict, cell: float) -> dict:
    """Tile the Domain bbox at `cell` meters, padded outward to whole cells.

    The lattice is anchored at the bbox's northwest corner (min-x, max-y after
    padding), matching the anchor that domain-target grid alignment produces.
    """
    minx, miny, maxx, maxy = domain["bbox"]
    nx = max(1, ceil((maxx - minx) / cell))
    ny = max(1, ceil((maxy - miny) / cell))
    return {
        "nx": nx,
        "ny": ny,
        "dx": float(cell),
        "dy": float(cell),
        "transform": [cell, 0.0, float(minx), 0.0, -cell, float(miny) + ny * cell],
        "crs": domain_crs_string(domain),
    }


def build_grid_lattice(grid_doc: dict, grid_id: str) -> dict:
    """Adopt an existing grid's horizontal lattice verbatim.

    Raises:
        HTTPException(422): If the grid has no georeference.
    """
    validate_grid_has_georeference(grid_doc, grid_id)
    geo = grid_doc["georeference"]
    transform = [float(c) for c in geo["transform"][:6]]
    return {
        "nx": int(geo["shape"][-1]),
        "ny": int(geo["shape"][-2]),
        "dx": abs(transform[0]),
        "dy": abs(transform[0]),
        "transform": transform,
        "crs": geo.get("crs"),
    }


def check_role_lattice_alignment(
    grid_data: dict,
    grid_id: str,
    role_name: str,
    lattice: dict,
    *,
    export_label: str,
    resolution_hint: str,
) -> None:
    """Per role: CRS, cell size, integer-cell lattice offset, coverage.

    Args:
        grid_data: Grid document data from Firestore.
        grid_id: Grid ID (for error messages).
        role_name: Role the grid fills on the export request.
        lattice: Export lattice dict (see module docstring).
        export_label: Export name used in error messages (e.g. "QUIC-Fire").
        resolution_hint: Copy-pasteable request fix for a resolution mismatch,
            with a `{res}` placeholder for the role grid's resolution (e.g.
            '"alignment": {{"dx": {res}, "dy": {res}}}').

    Raises:
        HTTPException(422): On any alignment violation.
    """
    geo = grid_data["georeference"]
    gtransform = geo["transform"]
    gcrs = geo.get("crs")
    crs = lattice["crs"]
    dx = lattice["dx"]
    minx = lattice["transform"][2]
    maxy = lattice["transform"][5]
    maxx = minx + lattice["nx"] * dx
    miny = maxy - lattice["ny"] * dx

    if not crs_equal(gcrs, crs):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"{export_label} export CRS mismatch: grid '{role_name}' ({grid_id}) "
            f"is in {gcrs}, but this export builds the {export_label} grid in "
            f"{crs}. All role grids must be in {crs}; rebuild this "
            f"grid in that CRS.",
        )

    gdx = abs(float(gtransform[0]))
    gdy = abs(float(gtransform[4]))
    if not isclose(gdx, dx, abs_tol=TOL) or not isclose(gdy, dx, abs_tol=TOL):
        grid_res = f"{gdx}" if isclose(gdx, gdy, abs_tol=TOL) else f"{gdx}x{gdy}"
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"{export_label} export resolution mismatch: grid '{role_name}' "
            f"({grid_id}) is {grid_res} m, but this export builds a "
            f"{dx} m {export_label} grid. To export at {gdx} m, add "
            f"{resolution_hint.format(res=gdx)} to your request; "
            f"otherwise provide role grids built at {dx} m.",
        )

    offset_x = (float(gtransform[2]) - minx) / dx
    offset_y = (maxy - float(gtransform[5])) / dx
    if not isclose(offset_x, round(offset_x), abs_tol=TOL) or not isclose(
        offset_y, round(offset_y), abs_tol=TOL
    ):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"{export_label} export lattice mismatch: grid '{role_name}' "
            f"({grid_id}) is offset from the {export_label} grid by a "
            f"non-integer number of cells (x {offset_x:.3f}, y {offset_y:.3f}). "
            f'Rebuild it domain-anchored (alignment.target="domain") or resample '
            f"it onto the domain lattice so its origin lands on whole cells.",
        )

    gh, gw = int(geo["shape"][-2]), int(geo["shape"][-1])
    gminx = float(gtransform[2])
    gmaxy = float(gtransform[5])
    gmaxx = gminx + gw * gdx
    gminy = gmaxy - gh * gdy
    if not (
        gminx <= minx + TOL
        and gminy <= miny + TOL
        and gmaxx >= maxx - TOL
        and gmaxy >= maxy - TOL
    ):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"{export_label} export coverage gap: grid '{role_name}' ({grid_id}) "
            f"does not cover the full {export_label} grid extent. Rebuild this grid "
            f"over the whole domain (or with more buffer cells) so it spans the "
            f"export area; resampling will not extend coverage.",
        )
