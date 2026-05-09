"""
api/v2/resources/grids/exports/quicfire/validators.py

Per-role validation orchestration for the QUIC-Fire export endpoint.

These helpers orchestrate the general-purpose grid validators in
`api.resources.grids.utils` to enforce QUIC-Fire's role-specific
requirements: 3D vs 2D dimensionality, expected band units, alignment to
the canopy grid's 2D footprint, and spatial coverage of the domain bbox.
"""

from fastapi import HTTPException, status

from api.db.documents import get_document_async
from api.resources.grids.exports.quicfire.schema import (
    FieldSource,
    QuicfireExportRequest,
    QuicfireExportSource,
)
from api.resources.grids.utils import (
    validate_band_unit,
    validate_grid_dimensionality,
    validate_grid_has_band,
    validate_grid_resolution_matches,
)
from lib.config import GRIDS_COLLECTION

# Per-role unit and dimensionality contract. Drives both the per-grid checks
# and the resolved snapshot recorded in Export.source.
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

# Maximum total cell count for the resolved fire grid. Matches the v1 cap.
_MAX_CELLS = 50_000_000


def _iter_roles(
    request: QuicfireExportRequest,
) -> list[tuple[str, FieldSource]]:
    """Return (role_name, FieldSource) for every role that's set on the request."""
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


async def validate_quicfire_request(
    request: QuicfireExportRequest,
    owner_id: str,
    domain: dict,
) -> QuicfireExportSource:
    """Validate a QUIC-Fire export request and produce its persisted source.

    Performs every check described in the design plan:
      1. Per-grid existence + ownership + domain + status='completed'
      2. Per-role band membership
      3. Per-role unit
      4. Per-role dimensionality
      5. Cell-size match between each grid and the canopy grid (CRS and bbox
         are already shared by domain invariant)
      6. Cell-count cap on the resolved fire grid

    Returns the QuicfireExportSource document (with `resolved` snapshot) ready
    to persist into Export.source.

    Raises HTTPException(422) on any validation failure.
    """
    domain_id = domain["id"]
    roles = _iter_roles(request)

    # 1. Load every distinct grid (existence + ownership + domain + status).
    grid_cache: dict[str, dict] = {}
    for _, source in roles:
        if source.grid_id not in grid_cache:
            _, snapshot = await get_document_async(
                GRIDS_COLLECTION,
                source.grid_id,
                owner_id=owner_id,
                domain_id=domain_id,
                document_status="completed",
            )
            grid_cache[source.grid_id] = snapshot.to_dict()

    # 2-4. Per-role checks: band membership, unit, dimensionality.
    for role_name, source in roles:
        grid_data = grid_cache[source.grid_id]
        expected_rank, expected_unit = _ROLE_CONTRACT[role_name]
        validate_grid_dimensionality(grid_data, source.grid_id, expected_rank)
        validate_grid_has_band(grid_data, source.grid_id, source.band)
        validate_band_unit(grid_data, source.grid_id, source.band, expected_unit)

    # 5. Cell-size match against the canopy grid (which sets dx, dy, dz, nz).
    canopy_grid_id = request.canopy_bulk_density.grid_id
    canopy_data = grid_cache[canopy_grid_id]
    for role_name, source in roles:
        if source.grid_id == canopy_grid_id:
            continue
        validate_grid_resolution_matches(
            grid_cache[source.grid_id],
            source.grid_id,
            canopy_data,
            canopy_grid_id,
        )

    # 6. Cell-count cap on the resolved fire grid.
    canopy_geo = canopy_data["georeference"]
    nz, ny, nx = canopy_geo["shape"]
    if nx * ny * nz > _MAX_CELLS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Resolved fire grid has {nx * ny * nz:_} cells "
                f"(nx={nx}, ny={ny}, nz={nz}); exceeds cap of "
                f"{_MAX_CELLS:_}. Use a coarser canopy grid."
            ),
        )

    # Build the resolved snapshot for full reproducibility.
    resolved: dict = {
        "domain": {
            "crs": domain.get("crs"),
            "bbox": domain.get("bbox"),
        },
        "fire_grid": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "transform": list(canopy_geo["transform"]),
            "z_origin": canopy_geo.get("z_origin"),
            "z_resolution": canopy_geo.get("z_resolution"),
            "crs": canopy_geo.get("crs"),
        },
        "roles": {},
    }
    for role_name, source in roles:
        grid_data = grid_cache[source.grid_id]
        georeference = grid_data["georeference"]
        rank, unit = _ROLE_CONTRACT[role_name]
        resolved["roles"][role_name] = {
            "grid_id": source.grid_id,
            "band": source.band,
            "unit": unit,
            "dimensionality": rank,
            "shape": list(georeference["shape"]),
            "transform": list(georeference["transform"]),
            "crs": georeference.get("crs"),
        }

    return QuicfireExportSource(
        domain_id=domain_id,
        canopy_bulk_density=request.canopy_bulk_density,
        canopy_moisture=request.canopy_moisture,
        canopy_savr=request.canopy_savr,
        surface_fuel_load=request.surface_fuel_load,
        surface_fuel_depth=request.surface_fuel_depth,
        surface_moisture=request.surface_moisture,
        surface_savr=request.surface_savr,
        topography=request.topography,
        resolved=resolved,
    )
