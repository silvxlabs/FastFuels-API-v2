"""Handler dispatch for treevox.

Routes voxelization requests to the appropriate handler based on the source's
(operation, input, entity) triple.
"""

from __future__ import annotations

from collections.abc import Callable

from treevox.errors import ProcessingError
from treevox.handlers import voxelize


def dispatch_handler(
    grid: dict,
    domain_gdf,
    progress: Callable[[str, int | None], None],
) -> voxelize.VoxelizationResult:
    """Route on the source's (operation, input, entity) triple."""
    source = grid["source"]
    key = (source.get("operation"), source.get("input"), source.get("entity"))
    match key:
        case ("voxelize", "inventory", "tree"):
            return voxelize.voxelize_inventory(grid, domain_gdf, progress)
        case _:
            raise ProcessingError(
                code="UNKNOWN_SOURCE",
                message=f"Unknown tree grid source: {key!r}",
                suggestion=(
                    "Supported sources today: "
                    "(operation='voxelize', input='inventory', entity='tree')."
                ),
            )
