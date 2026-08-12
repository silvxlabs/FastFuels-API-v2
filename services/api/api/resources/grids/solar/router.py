"""
api/v2/resources/grids/solar/router.py

Router for solar endpoints -- operations that consume an input grid and
produce a grid with solar information.

"""

from fastapi import APIRouter

from api.resources.grids.solar.irradiance.router import router as irradiance_router

router = APIRouter()

router.include_router(irradiance_router, prefix="/irradiance")
