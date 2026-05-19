"""
api/v2/resources/grids/upload/router.py

Aggregator router for grid upload endpoints. Each format has its own
sub-route with its own request schema and documentation:

- POST /upload/geotiff — GeoTIFF (2D)
- POST /upload/netcdf  — CF-conformant netCDF (2D or 3D)
"""

from fastapi import APIRouter

from api.resources.grids.upload.geotiff.router import router as geotiff_router
from api.resources.grids.upload.netcdf.router import router as netcdf_router

router = APIRouter()

router.include_router(geotiff_router, prefix="/geotiff")
router.include_router(netcdf_router, prefix="/netcdf")
