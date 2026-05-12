"""
api/v2/resources/grids/exports/quicfire/validators.py

Validation for the QUIC-Fire export endpoint.

Fire grid: vertical (`nz`, `dz`, `z_origin`) always comes from
`canopy_bulk_density`; horizontal lattice comes from `request.alignment`
(`target="domain"` pads Domain.bbox to (dx, dy); `target="grid"` matches
an existing grid's CRS/transform/shape).

Every role grid is then checked: band/unit/dimensionality, then CRS,
cell size, integer-cell lattice offset, and bbox coverage of the fire
grid. 3D roles additionally match (nz, dz, z_origin). Any failure is
a 422. The exporter never resamples; misalignment is the user's job to
fix upstream via `POST /grids/resample`.
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
    """Validate a QUIC-Fire export request; return the persisted source."""
    domain_id = domain["id"]
    roles = _iter_roles(request)

    # 1. Load every grid we'll touch.
    grids_to_load: set[str] = {src.grid_id for _, src in roles}
    if not isinstance(request.alignment, QUICFireExportAlignmentDomainTarget):
        grids_to_load.add(request.alignment.grid_id)

    grid_cache: dict[str, dict] = {}
    for grid_id in grids_to_load:
        _, snapshot = await get_document_async(
            GRIDS_COLLECTION,
            grid_id,
            owner_id=owner_id,
            domain_id=domain_id,
            document_status="completed",
        )
        grid_cache[grid_id] = snapshot.to_dict()

    # 2. Per-role band, unit, dimensionality.
    for role, src in roles:
        rank, unit = _ROLE_CONTRACT[role]
        grid_data = grid_cache[src.grid_id]
        validate_grid_has_band(grid_data, src.grid_id, src.band)
        validate_band_unit(grid_data, src.grid_id, src.band, unit)
        validate_grid_dimensionality(grid_data, src.grid_id, rank)

    # 3. Build the fire grid. Vertical from canopy; horizontal from alignment.
    canopy_geo = grid_cache[request.canopy_bulk_density.grid_id]["georeference"]
    nz = int(canopy_geo["shape"][0])
    dz = float(canopy_geo["z_resolution"])
    z_origin = float(canopy_geo["z_origin"])

    if isinstance(request.alignment, QUICFireExportAlignmentDomainTarget):
        dx = float(request.alignment.dx)
        minx, miny, maxx, maxy = domain["bbox"]
        nx = max(1, ceil((maxx - minx) / dx))
        ny = max(1, ceil((maxy - miny) / dx))
        fire_transform = [dx, 0.0, float(minx), 0.0, -dx, float(miny) + ny * dx]
        fire_crs = (domain.get("crs") or {}).get("properties", {}).get("name")
    else:
        target = grid_cache[request.alignment.grid_id]
        validate_grid_has_georeference(target, request.alignment.grid_id)
        ref = target["georeference"]
        fire_transform = [float(c) for c in ref["transform"][:6]]
        ny, nx = int(ref["shape"][-2]), int(ref["shape"][-1])
        fire_crs = ref.get("crs")
        dx = abs(fire_transform[0])

    fire_minx = fire_transform[2]
    fire_maxy = fire_transform[5]
    fire_maxx = fire_minx + nx * dx
    fire_miny = fire_maxy - ny * dx

    # 4. Per-role lattice alignment + coverage + (3D) z match.
    for role, src in roles:
        geo = grid_cache[src.grid_id]["georeference"]
        gtransform = geo["transform"]
        gcrs = geo.get("crs")

        if gcrs != fire_crs:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"Role '{role}' grid {src.grid_id}: CRS ({gcrs}) does not "
                f"match fire-grid CRS ({fire_crs}). {_HINT}",
            )

        gdx = abs(float(gtransform[0]))
        gdy = abs(float(gtransform[4]))
        if not isclose(gdx, dx, abs_tol=_TOL) or not isclose(gdy, dx, abs_tol=_TOL):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"Role '{role}' grid {src.grid_id}: cell size "
                f"({gdx}, {gdy}) does not match fire-grid ({dx}, {dx}). "
                f"{_HINT}",
            )

        offset_x = (float(gtransform[2]) - fire_minx) / dx
        offset_y = (fire_maxy - float(gtransform[5])) / dx
        if not isclose(offset_x, round(offset_x), abs_tol=_TOL) or not isclose(
            offset_y, round(offset_y), abs_tol=_TOL
        ):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"Role '{role}' grid {src.grid_id}: origin is not on the "
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
                f"Role '{role}' grid {src.grid_id}: bbox does not cover "
                f"the fire-grid bbox. {_HINT}",
            )

        if len(geo["shape"]) == 3:
            if (
                int(geo["shape"][0]) != nz
                or not isclose(float(geo["z_resolution"]), dz, abs_tol=_TOL)
                or not isclose(float(geo["z_origin"]), z_origin, abs_tol=_TOL)
            ):
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    f"Role '{role}' grid {src.grid_id}: z grid (nz, dz, "
                    f"z_origin) does not match fire-grid vertical.",
                )

    # 5. Cell count cap.
    if nx * ny * nz > _MAX_CELLS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Resolved fire grid has {nx * ny * nz:_} cells "
            f"(nx={nx}, ny={ny}, nz={nz}); exceeds cap of {_MAX_CELLS:_}.",
        )

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
        resolved={
            "fire_grid": {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "dx": dx,
                "dy": dx,
                "dz": dz,
                "transform": fire_transform,
                "z_origin": z_origin,
                "crs": fire_crs,
            },
        },
    )
