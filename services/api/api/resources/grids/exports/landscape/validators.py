"""
api/v2/resources/grids/exports/landscape/validators.py

Validation for the landscape export endpoint.

The orchestrator `validate_landscape_request` wires together a few pure
helpers, each unit-testable without Firestore or the async runtime:

* `_load_all_grids` — Firestore lookup for every grid the request touches
  (roles + the alignment target when `target="grid"`).
* `_check_role_contract` — per role: band membership, unit, 2D rank.
* `_check_fuel_model_declaration` — the declared `fire_behavior_fuel_model`
  is consistent with the fuel model grid's provenance.
* `_build_landscape_grid` — derives the landscape lattice from the alignment
  (shared `exports.alignment` helpers).
* `_check_role_alignment` — per role: CRS, cell size, integer-cell lattice
  offset, coverage (shared `exports.alignment` checks).
* `_check_cell_count_cap` — total cell count under the cap.
* `_make_source` — assemble the persisted `LandscapeExportSource`.

Every helper raises `HTTPException(422)` on failure.
"""

from fastapi import HTTPException, status

from api.db.documents import get_document_async
from api.resources.grids.exports.alignment import (
    build_domain_lattice,
    build_grid_lattice,
    check_role_lattice_alignment,
)
from api.resources.grids.exports.landscape.schema import (
    LandscapeExportAlignmentDomainTarget,
    LandscapeExportRequest,
    LandscapeExportSource,
    LandscapeFieldSource,
)
from api.resources.grids.utils import (
    validate_band_unit,
    validate_grid_dimensionality,
    validate_grid_has_band,
)
from lib.config import GRIDS_COLLECTION

# Per-role unit contract. All landscape roles are 2D. `None` marks the
# categorical fuel model band, which carries no unit.
_ROLE_CONTRACT: dict[str, str | None] = {
    "elevation": "m",
    "slope": "deg",
    "aspect": "deg",
    "fuel_model": None,
    "canopy_cover": "%",
    "canopy_height": "m",
    "canopy_base_height": "m",
    "canopy_bulk_density": "kg/m**3",
}

# Cap matches the QUIC-Fire export.
_MAX_CELLS = 50_000_000


def _iter_roles(
    request: LandscapeExportRequest,
) -> list[tuple[str, LandscapeFieldSource]]:
    """Enumerate (role_name, LandscapeFieldSource) in LANDFIRE band order."""
    return [
        ("elevation", request.elevation),
        ("slope", request.slope),
        ("aspect", request.aspect),
        ("fuel_model", request.fuel_model),
        ("canopy_cover", request.canopy_cover),
        ("canopy_height", request.canopy_height),
        ("canopy_base_height", request.canopy_base_height),
        ("canopy_bulk_density", request.canopy_bulk_density),
    ]


async def _load_all_grids(
    request: LandscapeExportRequest,
    owner_id: str,
    domain_id: str,
) -> dict[str, dict]:
    """Load every grid the request will touch into a dict keyed by grid_id.

    Includes every role's grid plus the alignment target's grid (when
    `target="grid"`). Delegates 404 (missing/unowned) and 422 (non-completed)
    behavior to `get_document_async`.
    """
    grid_ids: set[str] = {src.grid_id for _, src in _iter_roles(request)}
    if not isinstance(request.alignment, LandscapeExportAlignmentDomainTarget):
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


def _check_role_contract(
    grid_data: dict, src: LandscapeFieldSource, role_name: str
) -> None:
    """Per role: band present, band has expected unit, grid is 2D."""
    unit = _ROLE_CONTRACT[role_name]
    validate_grid_has_band(grid_data, src.grid_id, src.band)
    if unit is not None:
        validate_band_unit(grid_data, src.grid_id, src.band, unit)
    validate_grid_dimensionality(grid_data, src.grid_id, 2)


def _check_fuel_model_declaration(grid_data: dict, grid_id: str, declared: str) -> None:
    """The declared fuel model classification must match the fuel model grid's
    provenance.

    A grid created from the built-in LANDFIRE FBFM40 product carries
    `source.product == "fbfm40"`; declaring `fbfm13` for it is a contradiction.
    Custom / uploaded / derived grids carry other source shapes, and the
    declaration is trusted as the user's interpretation.
    """
    product = (grid_data.get("source") or {}).get("product")
    if product == "fbfm40" and declared != "fbfm40":
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Fuel model grid {grid_id} was created from the built-in LANDFIRE "
            f"FBFM40 product, but the request declares "
            f"fire_behavior_fuel_model={declared!r}. Declare 'fbfm40', or "
            f"provide a grid whose codes are actually Anderson 13 fuel models.",
        )


def _build_landscape_grid(
    request: LandscapeExportRequest,
    domain: dict,
    alignment_grid_doc: dict | None,
) -> dict:
    """Construct the landscape lattice from the request's alignment.

    `alignment_grid_doc` must be supplied when
    `request.alignment.target == "grid"` and ignored otherwise.
    """
    if isinstance(request.alignment, LandscapeExportAlignmentDomainTarget):
        return build_domain_lattice(domain, float(request.alignment.resolution))
    assert alignment_grid_doc is not None, (
        "alignment_grid_doc is required when alignment.target='grid'"
    )
    return build_grid_lattice(alignment_grid_doc, request.alignment.grid_id)


def _check_role_alignment(
    grid_data: dict,
    src: LandscapeFieldSource,
    role_name: str,
    landscape_grid: dict,
) -> None:
    """Per role: CRS, cell size, integer-cell lattice offset, coverage."""
    check_role_lattice_alignment(
        grid_data,
        src.grid_id,
        role_name,
        landscape_grid,
        export_label="landscape",
        resolution_hint='"alignment": {{"resolution": {res}}}',
    )


def _check_cell_count_cap(landscape_grid: dict) -> None:
    """Reject landscape lattices that exceed `_MAX_CELLS`."""
    nx, ny = landscape_grid["nx"], landscape_grid["ny"]
    cells = nx * ny
    if cells > _MAX_CELLS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Resolved landscape grid has {cells:_} cells "
            f"(nx={nx}, ny={ny}); exceeds cap of {_MAX_CELLS:_}.",
        )


def _make_source(
    request: LandscapeExportRequest, domain_id: str, landscape_grid: dict
) -> LandscapeExportSource:
    """Assemble the persisted `LandscapeExportSource`."""
    return LandscapeExportSource(
        domain_id=domain_id,
        alignment=request.alignment,
        fire_behavior_fuel_model=request.fire_behavior_fuel_model,
        elevation=request.elevation,
        slope=request.slope,
        aspect=request.aspect,
        fuel_model=request.fuel_model,
        canopy_cover=request.canopy_cover,
        canopy_height=request.canopy_height,
        canopy_base_height=request.canopy_base_height,
        canopy_bulk_density=request.canopy_bulk_density,
        resolved={"landscape_grid": landscape_grid},
    )


async def validate_landscape_request(
    request: LandscapeExportRequest,
    owner_id: str,
    domain: dict,
) -> LandscapeExportSource:
    """Validate a landscape export request; return the persisted source."""
    domain_id = domain["id"]
    grid_cache = await _load_all_grids(request, owner_id, domain_id)
    roles = _iter_roles(request)

    for role_name, src in roles:
        _check_role_contract(grid_cache[src.grid_id], src, role_name)

    _check_fuel_model_declaration(
        grid_cache[request.fuel_model.grid_id],
        request.fuel_model.grid_id,
        request.fire_behavior_fuel_model,
    )

    alignment_grid_doc = (
        None
        if isinstance(request.alignment, LandscapeExportAlignmentDomainTarget)
        else grid_cache[request.alignment.grid_id]
    )
    landscape_grid = _build_landscape_grid(request, domain, alignment_grid_doc)

    for role_name, src in roles:
        _check_role_alignment(grid_cache[src.grid_id], src, role_name, landscape_grid)

    _check_cell_count_cap(landscape_grid)
    return _make_source(request, domain_id, landscape_grid)
