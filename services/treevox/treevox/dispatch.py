"""Handler dispatch for treevox.

Routes grid processing requests to the appropriate handler based on the source's
(operation, input, entity) triple.
"""

from __future__ import annotations

from collections.abc import Callable

from treevox.errors import ProcessingError
from treevox.handlers import duet, voxelize

# Every handler returns an object carrying `gcs_path`, `georeference`, and
# `chunk_shape`; main.py needs nothing else. Voxelize's georeference is 3D and
# DUET's is 2D, since DUET reads a canopy and writes a surface.
HandlerResult = voxelize.VoxelizationResult | duet.DuetResult


def dispatch_handler(
    grid: dict,
    domain_gdf,
    progress: Callable[[str, int | None], None],
) -> HandlerResult:
    """Route on the source's (operation, input, entity) triple."""
    source = grid["source"]
    key = (source.get("operation"), source.get("input"), source.get("entity"))
    match key:
        case ("voxelize", "inventory", "tree"):
            return voxelize.voxelize_inventory(grid, domain_gdf, progress)
        case ("duet", "grid", "tree"):
            return duet.duet_grid(grid, domain_gdf, progress)
        case _:
            raise ProcessingError(
                code="UNKNOWN_SOURCE",
                message=f"Unknown tree grid source: {key!r}",
                suggestion=(
                    "Supported sources today: "
                    "(operation='voxelize', input='inventory', entity='tree') and "
                    "(operation='duet', input='grid', entity='tree')."
                ),
            )
