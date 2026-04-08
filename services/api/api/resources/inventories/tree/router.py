"""
api/v2/resources/inventories/tree/router.py

Aggregator router for tree inventory creation endpoints.

Mounts source-specific sub-routers (chm, future: pim, point-cloud, ...) under
the /tree prefix so URLs follow the {product}/{source} pattern matching the
grids/tree structure proposed in #151.
"""

from fastapi import APIRouter

from api.resources.inventories.tree.chm.router import router as chm_router

router = APIRouter()

router.include_router(chm_router, prefix="/chm", tags=["Inventories - Tree CHM"])
