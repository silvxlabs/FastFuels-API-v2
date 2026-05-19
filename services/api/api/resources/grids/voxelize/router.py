"""
api/v2/resources/grids/voxelize/router.py

Aggregator router for voxelize endpoints — operations that consume an input
(inventory, layerset, lookup) and produce a 3D voxel grid.

`voxelize/` is a deliberate operation-grouped exception to the
`{product}/{source}` URL convention used by 2D grid endpoints. The unifying
concept across these endpoints is the *operation*, not a single product.
"""

from fastapi import APIRouter

from api.resources.grids.voxelize.inventory.router import router as inventory_router

router = APIRouter()

router.include_router(inventory_router, prefix="/inventory")
