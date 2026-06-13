"""
api/v2/resources/inventories/tree/allometry/router.py

Aggregator router for allometry-based tree inventory creation endpoints.

`allometry` is a category of methods that fill in the missing morphology of an
existing tree inventory. `gdam` is the first method; future methods mount here as
siblings under the /allometry prefix with no URL changes elsewhere.
"""

from fastapi import APIRouter

from api.resources.inventories.tree.allometry.gdam.router import router as gdam_router

router = APIRouter()

router.include_router(
    gdam_router, prefix="/gdam", tags=["Inventories - Tree Allometry GDAM"]
)
