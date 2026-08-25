"""
api/v2/resources/grids/fuel_moisture/router.py

Router for fuel-moisture grid products. Splits by category (dead vs live)
before the model source segment, e.g. /fuel-moisture/dead/fosberg.
"""

from fastapi import APIRouter

from api.resources.grids.fuel_moisture.dead.router import router as dead_router

router = APIRouter()

router.include_router(dead_router, prefix="/dead")
