"""
api/v2/resources/grids/lookup/router.py

Router for lookup grid source endpoints.

This module mounts each product's own sub-router; no product-specific
logic lives here. Product endpoints are defined in their respective
subpackages (fbfm13/, fbfm40/, fccs/, ...).
"""

from fastapi import APIRouter

from api.resources.grids.lookup.fbfm13.router import router as fbfm13_lookup_router
from api.resources.grids.lookup.fbfm40.router import router as fbfm40_lookup_router

router = APIRouter()

# Mounted the same way the top-level grids/router.py mounts product
# sub-routers, so that /domains/{id}/grids/lookup/fbfm40 and /fbfm13 are
# unchanged. Each product gets its own tag here (rather than the shared
# "Grids - Lookup" tag applied at the grids/router.py mount point) so
# they're grouped separately in the OpenAPI docs.
router.include_router(
    fbfm13_lookup_router, prefix="/fbfm13", tags=["Grids - Lookup - FBFM13"]
)
router.include_router(
    fbfm40_lookup_router, prefix="/fbfm40", tags=["Grids - Lookup - FBFM40"]
)
