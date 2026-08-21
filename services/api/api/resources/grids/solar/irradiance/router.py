"""
api/v2/resources/grids/solar/irradiance/router.py

Router for irradiance endpoints -- operations that consume an input grid and
produce an irradiance grid.

"""

from fastapi import APIRouter

from api.resources.grids.solar.irradiance.leaflux.router import router as leaflux_router

router = APIRouter()

router.include_router(leaflux_router, prefix="/leaflux")
