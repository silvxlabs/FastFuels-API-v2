"""
api/v2/resources/inventories/tree/pim/fusion/router.py

Aggregator router for PIM fusion inventory endpoints.

Each ``{others}`` combination fused into a PIM is its own sub-router under this
prefix, so the URL reads ``tree/pim/fusion/{others}`` (``chm`` today). A new
combination is a new sub-router here; a new algorithm for an existing
combination is a new ``method`` member on that sub-router's schema, never a new
path.
"""

from fastapi import APIRouter

from api.resources.inventories.tree.pim.fusion.chm.router import router as chm_router

router = APIRouter()

router.include_router(chm_router, prefix="/chm")
