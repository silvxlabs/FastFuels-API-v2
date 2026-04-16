"""
api/v2/resources/grids/tree/router.py

Aggregator router for tree fuel grid creation endpoints.

Mounts source-specific sub-routers (inventory today; future: point-cloud,
downscale, ...) under the /tree prefix so URLs follow the {product}/{source}
pattern established elsewhere in the grids router.
"""

from fastapi import APIRouter

from api.resources.grids.tree.inventory.router import router as inventory_router

router = APIRouter()

router.include_router(
    inventory_router, prefix="/inventory", tags=["Grids - Tree Inventory"]
)
