"""
api/v2/resources/grids/fuel_moisture/dead/router.py

Router for dead fuel moisture content grids. Mounts each model under its own
source segment, e.g. /fuel-moisture/dead/fosberg.
"""

from fastapi import APIRouter

from api.resources.grids.fuel_moisture.dead.fosberg.router import (
    router as fosberg_router,
)

router = APIRouter()

router.include_router(fosberg_router, prefix="/fosberg")
