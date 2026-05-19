"""
api/v2/resources/grids/rasterize/router.py

Aggregate router for ``/rasterize/*`` grid sources. Mounted in
``services/api/api/resources/grids/router.py`` at the ``/rasterize`` prefix.

Each subdirectory under ``rasterize/`` defines a source that produces a
grid by rasterizing some vector input. Currently the only source is
``layerset``; new sources slot in here without re-mounting in the parent.
"""

from fastapi import APIRouter

from api.resources.grids.rasterize.layerset.router import router as layerset_router

router = APIRouter()

router.include_router(layerset_router)
