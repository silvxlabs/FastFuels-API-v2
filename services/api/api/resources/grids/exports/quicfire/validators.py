"""
api/v2/resources/grids/exports/quicfire/validators.py

Validation for the QUIC-Fire export endpoint.

The orchestrator `validate_quicfire_request` wires together a few pure
helpers, each unit-testable without Firestore or the async runtime:

* `_load_all_grids` — Firestore lookup for every grid the request touches
  (roles + the alignment target when `target="grid"`).
* `_check_role_contract` — per role: band membership, unit, dimensionality.
* `_build_fire_grid` — derives the fire grid's georeference from the
  alignment + the canopy grid's vertical.
* `_check_role_alignment` — per role: CRS, cell size, integer-cell lattice
  offset, horizontal coverage.
* `_check_3d_role_vertical` — per 3D role: `nz` / `dz` / `z_origin` match
  the fire grid's vertical.
* `_check_cell_count_cap` — total cell count under the v1 cap.
* `_make_source` — assemble the persisted `QuicfireExportSource`.

Every helper raises `HTTPException(422)` on failure.
"""

from math import ceil, isclose

from fastapi import HTTPException, status

from api.db.documents import get_document_async
from api.resources.grids.exports.quicfire.schema import (
    FieldSource,
    QUICFireExportAlignmentDomainTarget,
    QuicfireExportRequest,
    QuicfireExportSource,
)
from api.resources.grids.utils import (
    validate_band_unit,
    validate_grid_dimensionality,
    validate_grid_has_band,
    validate_grid_has_georeference,
)
from lib.config import GRIDS_COLLECTION

# Per-role unit and dimensionality contract.
_ROLE_CONTRACT: dict[str, tuple[int, str]] = {
    "canopy_bulk_density": (3, "kg/m³"),
    "canopy_moisture": (3, "%"),
    "canopy_savr": (3, "m⁻¹"),
    "surface_fuel_load": (2, "kg/m²"),
    "surface_fuel_depth": (2, "m"),
    "surface_moisture": (2, "%"),
    "surface_savr": (2, "m⁻¹"),
    "topography": (2, "m"),
}

# 1 µm tolerance for transform-coefficient comparisons.
_TOL = 1e-6

# Cap matches v1.
_MAX_CELLS = 50_000_000

_HINT = (
    'Use POST /v2/domains/{domain_id}/grids/resample with alignment.target="grid" '
    "to align this grid to the fire-grid lattice."
)


def _iter_roles(
    request: QuicfireExportRequest,
) -> list[tuple[str, FieldSource]]:
    """Enumerate (role_name, FieldSource) for every role set on the request."""
    return [
        (name, source)
        for name, source in (
            ("canopy_bulk_density", request.canopy_bulk_density),
            ("canopy_moisture", request.canopy_moisture),
            ("canopy_savr", request.canopy_savr),
            ("surface_fuel_load", request.surface_fuel_load),
            ("surface_fuel_depth", request.surface_fuel_depth),
            ("surface_moisture", request.surface_moisture),
            ("surface_savr", request.surface_savr),
            ("topography", request.topography),
        )
        if source is not None
    ]


def _domain_crs_string(domain: dict) -> str | None:
    """Extract the EPSG-style CRS string from the Domain's GeoJSON CRS dict."""
    crs = domain.get("crs")
    if isinstance(crs, dict):
        return crs.get("properties", {}).get("name")
    return crs


async def _load_all_grids(
    request: QuicfireExportRequest,
    owner_id: str,
    domain_id: str,
) -> dict[str, dict]:
    """Load every grid the request will touch into a dict keyed by grid_id.

    Includes every role's grid plus the alignment target's grid (when
    `target="grid"`). Delegates 404 (missing/unowned) and 422 (non-completed)
    behavior to `get_document_async`.
    """
    grid_ids: set[str] = {src.grid_id for _, src in _iter_roles(request)}
    if not isinstance(request.alignment, QUICFireExportAlignmentDomainTarget):
        grid_ids.add(request.alignment.grid_id)

    cache: dict[str, dict] = {}
    for grid_id in grid_ids:
        _, snapshot = await get_document_async(
            GRIDS_COLLECTION,
            grid_id,
            owner_id=owner_id,
            domain_id=domain_id,
            document_status="completed",
        )
        cache[grid_id] = snapshot.to_dict()
    return cache


def _check_role_contract(grid_data: dict, src: FieldSource, role_name: str) -> None:
    """Per role: band present, band has expected unit, grid has expected rank."""
    rank, unit = _ROLE_CONTRACT[role_name]
    validate_grid_has_band(grid_data, src.grid_id, src.band)
    validate_band_unit(grid_data, src.grid_id, src.band, unit)
    validate_grid_dimensionality(grid_data, src.grid_id, rank)


def _build_fire_grid(
    request: QuicfireExportRequest,
    domain: dict,
    canopy_geo: dict,
    alignment_grid_doc: dict | None,
) -> dict:
    """Construct the fire-grid spec. Vertical comes from the canopy; the
    horizontal lattice comes from `request.alignment`.

    `alignment_grid_doc` must be supplied when
    `request.alignment.target == "grid"` and ignored otherwise.
    """
    nz = int(canopy_geo["shape"][0])
    dz = float(canopy_geo["z_resolution"])
    z_origin = float(canopy_geo["z_origin"])

    if isinstance(request.alignment, QUICFireExportAlignmentDomainTarget):
        dx = float(request.alignment.dx)
        minx, miny, maxx, maxy = domain["bbox"]
        nx = max(1, ceil((maxx - minx) / dx))
        ny = max(1, ceil((maxy - miny) / dx))
        fire_transform = [dx, 0.0, float(minx), 0.0, -dx, float(miny) + ny * dx]
        fire_crs = _domain_crs_string(domain)
    else:
        assert alignment_grid_doc is not None, (
            "alignment_grid_doc is required when alignment.target='grid'"
        )
        validate_grid_has_georeference(alignment_grid_doc, request.alignment.grid_id)
        ref = alignment_grid_doc["georeference"]
        fire_transform = [float(c) for c in ref["transform"][:6]]
        ny, nx = int(ref["shape"][-2]), int(ref["shape"][-1])
        fire_crs = ref.get("crs")
        dx = abs(fire_transform[0])

    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": dx,
        "dy": dx,
        "dz": dz,
        "transform": fire_transform,
        "z_origin": z_origin,
        "crs": fire_crs,
    }


def _check_role_alignment(
    grid_data: dict, src: FieldSource, role_name: str, fire_grid: dict
) -> None:
    """Per role: CRS, cell size, integer-cell lattice offset, coverage."""
    geo = grid_data["georeference"]
    gtransform = geo["transform"]
    gcrs = geo.get("crs")
    fire_crs = fire_grid["crs"]
    dx = fire_grid["dx"]
    fire_minx = fire_grid["transform"][2]
    fire_maxy = fire_grid["transform"][5]
    fire_maxx = fire_minx + fire_grid["nx"] * dx
    fire_miny = fire_maxy - fire_grid["ny"] * dx

    if gcrs != fire_crs:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Role '{role_name}' grid {src.grid_id}: CRS ({gcrs}) does not "
            f"match fire-grid CRS ({fire_crs}). {_HINT}",
        )

    gdx = abs(float(gtransform[0]))
    gdy = abs(float(gtransform[4]))
    if not isclose(gdx, dx, abs_tol=_TOL) or not isclose(gdy, dx, abs_tol=_TOL):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Role '{role_name}' grid {src.grid_id}: cell size "
            f"({gdx}, {gdy}) does not match fire-grid ({dx}, {dx}). {_HINT}",
        )

    offset_x = (float(gtransform[2]) - fire_minx) / dx
    offset_y = (fire_maxy - float(gtransform[5])) / dx
    if not isclose(offset_x, round(offset_x), abs_tol=_TOL) or not isclose(
        offset_y, round(offset_y), abs_tol=_TOL
    ):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Role '{role_name}' grid {src.grid_id}: origin is not on the "
            f"fire-grid lattice (x offset {offset_x:.6f} cells, y offset "
            f"{offset_y:.6f} cells; must be integers). {_HINT}",
        )

    gh, gw = int(geo["shape"][-2]), int(geo["shape"][-1])
    gminx = float(gtransform[2])
    gmaxy = float(gtransform[5])
    gmaxx = gminx + gw * gdx
    gminy = gmaxy - gh * gdy
    if not (
        gminx <= fire_minx + _TOL
        and gminy <= fire_miny + _TOL
        and gmaxx >= fire_maxx - _TOL
        and gmaxy >= fire_maxy - _TOL
    ):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Role '{role_name}' grid {src.grid_id}: bbox does not cover "
            f"the fire-grid bbox. {_HINT}",
        )


def _check_3d_role_vertical(
    grid_data: dict, src: FieldSource, role_name: str, fire_grid: dict
) -> None:
    """For 3D roles only: `nz`, `dz`, `z_origin` match the fire grid.
    A no-op for 2D roles."""
    geo = grid_data["georeference"]
    if len(geo["shape"]) != 3:
        return
    if (
        int(geo["shape"][0]) != fire_grid["nz"]
        or not isclose(float(geo["z_resolution"]), fire_grid["dz"], abs_tol=_TOL)
        or not isclose(float(geo["z_origin"]), fire_grid["z_origin"], abs_tol=_TOL)
    ):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Role '{role_name}' grid {src.grid_id}: z grid (nz, dz, "
            f"z_origin) does not match fire-grid vertical.",
        )


def _check_cell_count_cap(fire_grid: dict) -> None:
    """Reject fire grids that exceed `_MAX_CELLS`."""
    nx, ny, nz = fire_grid["nx"], fire_grid["ny"], fire_grid["nz"]
    cells = nx * ny * nz
    if cells > _MAX_CELLS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Resolved fire grid has {cells:_} cells "
            f"(nx={nx}, ny={ny}, nz={nz}); exceeds cap of {_MAX_CELLS:_}.",
        )


def _make_source(
    request: QuicfireExportRequest, domain_id: str, fire_grid: dict
) -> QuicfireExportSource:
    """Assemble the persisted `QuicfireExportSource`."""
    return QuicfireExportSource(
        domain_id=domain_id,
        alignment=request.alignment,
        canopy_bulk_density=request.canopy_bulk_density,
        canopy_moisture=request.canopy_moisture,
        canopy_savr=request.canopy_savr,
        surface_fuel_load=request.surface_fuel_load,
        surface_fuel_depth=request.surface_fuel_depth,
        surface_moisture=request.surface_moisture,
        surface_savr=request.surface_savr,
        topography=request.topography,
        rhof_merge=request.rhof_merge,
        moist_merge=request.moist_merge,
        savr_merge=request.savr_merge,
        resolved={"fire_grid": fire_grid},
    )


async def validate_quicfire_request(
    request: QuicfireExportRequest,
    owner_id: str,
    domain: dict,
) -> QuicfireExportSource:
    """Validate a QUIC-Fire export request; return the persisted source."""
    domain_id = domain["id"]
    grid_cache = await _load_all_grids(request, owner_id, domain_id)
    roles = _iter_roles(request)

    for role_name, src in roles:
        _check_role_contract(grid_cache[src.grid_id], src, role_name)

    canopy_geo = grid_cache[request.canopy_bulk_density.grid_id]["georeference"]
    alignment_grid_doc = (
        None
        if isinstance(request.alignment, QUICFireExportAlignmentDomainTarget)
        else grid_cache[request.alignment.grid_id]
    )
    fire_grid = _build_fire_grid(request, domain, canopy_geo, alignment_grid_doc)

    for role_name, src in roles:
        grid_data = grid_cache[src.grid_id]
        _check_role_alignment(grid_data, src, role_name, fire_grid)
        _check_3d_role_vertical(grid_data, src, role_name, fire_grid)

    _check_cell_count_cap(fire_grid)
    return _make_source(request, domain_id, fire_grid)
