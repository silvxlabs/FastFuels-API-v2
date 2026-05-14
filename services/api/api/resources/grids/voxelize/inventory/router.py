"""
api/v2/resources/grids/voxelize/inventory/router.py

Aggregator for voxelize-from-inventory endpoints, parameterized by the kind
of inventory being voxelized (tree today; shrub, surface, ... in the future).
"""

from fastapi import APIRouter

from api.resources.grids.voxelize.inventory.tree.router import router as tree_router

router = APIRouter()

router.include_router(tree_router, prefix="/tree", tags=["Grids - Voxelize"])
