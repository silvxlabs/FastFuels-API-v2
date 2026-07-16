"""
api/v2/resources/grids/exports/quicfire/validators.py

Validation for the QUIC-Fire export endpoint.

The orchestrator `validate_quicfire_request` wires together a few pure
helpers, each unit-testable without Firestore or the async runtime:

* `_load_all_grids` — Firestore lookup for every grid the request touches
  (roles + the alignment target when `target="grid"`).
* `_check_role_contract` — per role: band membership, unit, dimensionality.
* `_build_fire_grid` — derives the fire grid's georeference: the horizontal
  lattice and `dz` from the alignment (domain target) or the reference grid +
  canopy (grid target); `nz` / `z_origin` always from the canopy.
* `_check_role_alignment` — per role: CRS, cell size, integer-cell lattice
  offset, horizontal coverage (shared `exports.alignment` checks).
* `_check_3d_role_vertical` — per 3D role: `dz` / `nz` / `z_origin` match
  the fire grid's vertical (the exporter never resamples vertically).
* `_check_cell_count_cap` — total cell count under the v1 cap.
* `_make_source` — assemble the persisted `QuicfireExportSource`.

Every helper raises `HTTPException(422)` on failure.
"""

from math import isclose

from fastapi import HTTPException, status

from api.db.documents import get_document_async
from api.resources.grids.exports.alignment import (
    TOL as _TOL,
)
from api.resources.grids.exports.alignment import (
    build_domain_lattice,
    build_grid_lattice,
    check_role_lattice_alignment,
)
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
)
from lib.config import GRIDS_COLLECTION

# Per-role unit and dimensionality contract.
_ROLE_CONTRACT: dict[str, tuple[int, str]] = {
    "canopy_bulk_density": (3, "kg/m**3"),
    "canopy_moisture": (3, "%"),
    "canopy_savr": (3, "1/m"),
    "surface_fuel_load": (2, "kg/m**2"),
    "surface_fuel_depth": (2, "m"),
    "surface_moisture": (2, "%"),
    "surface_savr": (2, "1/m"),
    "topography": (2, "m"),
}

# Cap matches v1.
_MAX_CELLS = 50_000_000


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
    """Construct the fire-grid spec. `nz` and `z_origin` always come from the
    canopy. The vertical cell size `dz` comes from `request.alignment.dz` for
    the domain target (the exporter later requires the canopy grid's
    `z_resolution` to match it) and from the canopy for the grid target, which
    has no `dz` knob. The horizontal lattice comes from `request.alignment`.

    `alignment_grid_doc` must be supplied when
    `request.alignment.target == "grid"` and ignored otherwise.
    """
    nz = int(canopy_geo["shape"][0])
    z_origin = float(canopy_geo["z_origin"])

    if isinstance(request.alignment, QUICFireExportAlignmentDomainTarget):
        dz = float(request.alignment.dz)
        lattice = build_domain_lattice(domain, float(request.alignment.dx))
    else:
        assert alignment_grid_doc is not None, (
            "alignment_grid_doc is required when alignment.target='grid'"
        )
        dz = float(canopy_geo["z_resolution"])
        lattice = build_grid_lattice(alignment_grid_doc, request.alignment.grid_id)

    return lattice | {"nz": nz, "dz": dz, "z_origin": z_origin}


def _check_role_alignment(
    grid_data: dict, src: FieldSource, role_name: str, fire_grid: dict
) -> None:
    """Per role: CRS, cell size, integer-cell lattice offset, coverage."""
    check_role_lattice_alignment(
        grid_data,
        src.grid_id,
        role_name,
        fire_grid,
        export_label="QUIC-Fire",
        resolution_hint='"alignment": {{"dx": {res}, "dy": {res}}}',
    )


def _check_3d_role_vertical(
    grid_data: dict, src: FieldSource, role_name: str, fire_grid: dict
) -> None:
    """For 3D roles only: `dz`, `nz`, `z_origin` match the fire grid.
    A no-op for 2D roles. The exporter never resamples vertically, so a role
    grid's vertical must already equal the fire grid's."""
    geo = grid_data["georeference"]
    if len(geo["shape"]) != 3:
        return

    gdz = float(geo["z_resolution"])
    dz = fire_grid["dz"]
    if not isclose(gdz, dz, abs_tol=_TOL):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"QUIC-Fire export vertical resolution mismatch: grid '{role_name}' "
            f"({src.grid_id}) has {gdz} m vertical cells (z_resolution), but this "
            f"export builds a {dz} m QUIC-Fire grid vertically. The exporter does "
            f"not resample vertically. Rebuild the tree grid at {dz} m "
            f'("resolution": {{"vertical": {dz}}}), or set '
            f'"alignment": {{"dz": {gdz}}} to match your tree grid.',
        )

    if int(geo["shape"][0]) != fire_grid["nz"] or not isclose(
        float(geo["z_origin"]), fire_grid["z_origin"], abs_tol=_TOL
    ):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Role '{role_name}' grid {src.grid_id}: z grid (nz, z_origin) does "
            f"not match fire-grid vertical.",
        )


def _check_cell_count_cap(fire_grid: dict) -> None:
    """Reject fire grids that exceed `_MAX_CELLS`."""
    nx, ny, nz = fire_grid["nx"], fire_grid["ny"], fire_grid["nz"]
    cells = nx * ny * nz
    if cells > _MAX_CELLS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
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
