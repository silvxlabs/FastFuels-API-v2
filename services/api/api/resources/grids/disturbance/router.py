"""
api/v2/resources/grids/disturbance/router.py

Router for disturbance grid products. Splits by category (annual, and
possibly others such as historical or fuel disturbance) before the source
segment, e.g. /disturbance/annual/landfire.
"""

from fastapi import APIRouter

from api.resources.grids.disturbance.annual.router import router as annual_router

router = APIRouter()

router.include_router(annual_router, prefix="/annual")
