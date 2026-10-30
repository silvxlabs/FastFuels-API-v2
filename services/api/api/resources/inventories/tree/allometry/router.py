"""
api/v2/resources/inventories/tree/allometry/router.py

Aggregator router for allometry-based tree inventory creation endpoints.

`allometry` is a category of methods that fill in the missing morphology of an
existing tree inventory: `gdam` predicts dbh, crown ratio and species, and
`crown_base_height` computes crown base heights from those.
"""

from fastapi import APIRouter

from api.resources.inventories.tree.allometry.crown_base_height.router import (
    router as crown_base_height_router,
)
from api.resources.inventories.tree.allometry.gdam.router import router as gdam_router

router = APIRouter()

router.include_router(
    gdam_router, prefix="/gdam", tags=["Inventories - Tree Allometry GDAM"]
)
router.include_router(
    crown_base_height_router,
    prefix="/crown_base_height",
    tags=["Inventories - Tree Allometry Crown Base Height"],
)
