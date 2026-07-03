from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from api.auth import authenticate_user
from api.resources.applications.router import router as applications_router
from api.resources.domains.router import router as domain_router
from api.resources.exports.router import router as exports_router
from api.resources.features.router import router as features_router
from api.resources.features.router import wildcard_router as features_wildcard_router
from api.resources.grids.router import router as grids_router
from api.resources.grids.router import wildcard_router as grids_wildcard_router
from api.resources.inventories.router import router as inventories_router
from api.resources.inventories.router import (
    wildcard_router as inventories_wildcard_router,
)
from api.resources.keys.router import router as keys_router
from api.resources.point_clouds.router import router as point_clouds_router
from api.resources.point_clouds.router import (
    wildcard_router as point_clouds_wildcard_router,
)
from api.resources.users.router import router as users_router
from lib.config import DEPLOYMENT_ENV

CORS_ORIGINS = {
    "prod": [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://beta-app-fastfuels-silvxlabs.web.app",
    ],
    "dev": [
        "http://localhost:3000",
        "http://localhost:8080",
    ],
    "local": [
        "http://localhost:3000",
        "http://localhost:8080",
    ],
}


# Regex for Firebase preview channels: https://silvx-fastfuels--<suffix>.web.app
preview_origin_regex = r"https://silvx-fastfuels--.*\.web\.app"


API_DESCRIPTION = """
A JSON API for generating high-resolution 3D fuel inputs for physics-based
wildfire simulation models such as **QUIC-Fire**, **FIRETEC**, and **FDS**.

> ⚠️ **Beta — under active development.** v2 is a ground-up redesign of the
> FastFuels API. Endpoints, request/response schemas, and resource semantics
> may change without notice while the API is in beta. Pin to a specific
> deployment and follow the changelog if you are building against it.

## What's here

- **Domains** — geographic areas of interest that anchor all other resources.
- **Features** — vector layers (roads, water, etc.) attached to a domain.
- **Inventories** — tree / fuel inventories sampled within a domain.
- **Grids** — rasterized surface, canopy, and topography fields aligned to a
  domain's grid.
- **Exports** — packaged outputs (Zarr, NetCDF, QUIC-Fire inputs, …) generated
  from a domain's inventories and grids.

## Design

Read the design philosophy and resource guides in the
[FastFuels-API-v2 repository](https://github.com/silvxlabs/FastFuels-API-v2)
before integrating — v2 favors explicit, reproducible primitives over the
convenience helpers that shipped in v1.

## Feedback

Found a bug or have a request? Open an issue on
[GitHub](https://github.com/silvxlabs/FastFuels-API-v2/issues).
"""

OPENAPI_TAGS = [
    {"name": "Index", "description": "Service metadata and welcome endpoint."},
    {
        "name": "Domains",
        "description": "Geographic areas of interest. All other resources hang off a domain.",
    },
    {
        "name": "Features",
        "description": "Vector layers (roads, water, etc.) attached to a domain.",
    },
    {
        "name": "Inventories",
        "description": "Tree and fuel inventories sampled within a domain.",
    },
    {
        "name": "Grids",
        "description": "Rasterized surface, canopy, and topography fields aligned to a domain's grid.",
    },
    {
        "name": "Exports",
        "description": "Packaged outputs generated from a domain's inventories and grids.",
    },
    {
        "name": "Point Clouds",
        "description": "Airborne (ALS) and terrestrial (TLS) laser-scan point clouds attached to a domain.",
    },
    {"name": "Keys", "description": "Manage API keys used to authenticate requests."},
    {
        "name": "Applications",
        "description": "Applications registered against the FastFuels API.",
    },
    {
        "name": "Users",
        "description": "The authenticated owner's identity, tier, quotas, and current usage.",
    },
]


app = FastAPI(
    title="FastFuels API",
    summary="3D fuels for next-generation fire behavior models.",
    description=API_DESCRIPTION,
    version="2.0.0-beta",
    license_info={
        "name": "MIT",
        "url": "https://github.com/silvxlabs/FastFuels-API-v2/blob/main/LICENSE",
    },
    openapi_tags=OPENAPI_TAGS,
    separate_input_output_schemas=False,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS.get(DEPLOYMENT_ENV, CORS_ORIGINS["local"]),
    allow_origin_regex=preview_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Disposition",
        "Transfer-Encoding",
        "X-Data-Shape",
        "X-Data-Dtype",
        "X-Data-Order",
        "X-Data-Format",
        "X-Data-Fill-Value",
        "X-Data-NNZ",
        "X-Data-Index-Dtype",
        "X-Data-Value-Dtype",
        "X-Partition-Index",
        "X-Row-Count",
        "X-Total-Rows",
        "X-Num-Partitions",
    ],
)

api_router = APIRouter()


@app.get(
    "/",
    tags=["Index"],
    summary="Welcome / service metadata",
    description="Returns service metadata, documentation links, and the current "
    "deployment status. Useful as a liveness check and as a starting "
    "point for discovering the API.",
)
async def index():
    return {
        "name": "FastFuels API",
        "version": app.version,
        "status": "beta",
        "message": (
            "Welcome to FastFuels v2 — the next-generation API for 3D fuels. "
            "v2 is in active development; schemas and endpoints may change "
            "without notice."
        ),
        "environment": DEPLOYMENT_ENV,
        "documentation": {
            "swagger": {
                "url": "/docs",
                "description": "Interactive API documentation (Swagger UI).",
            },
            "redoc": {
                "url": "/redoc",
                "description": "Static API documentation (ReDoc).",
            },
            "openapi": {
                "url": "/openapi.json",
                "description": "Raw OpenAPI 3.1 schema.",
            },
            "repository": {
                "url": "https://github.com/silvxlabs/FastFuels-API-v2",
                "description": "Source code, design docs, and changelog.",
            },
        },
        "webApplication": {
            "url": "https://beta-app-fastfuels-silvxlabs.web.app",
            "description": "Beta web application for creating, editing, and visualizing FastFuels v2 data.",
        },
        "feedback": {
            "issues": "https://github.com/silvxlabs/FastFuels-API-v2/issues",
            "email": "support.fastfuels@silvxlabs.com",
        },
    }


# Include resource routers
api_router.include_router(domain_router, prefix="/domains", tags=["Domains"])
api_router.include_router(exports_router, prefix="/exports", tags=["Exports"])
api_router.include_router(
    grids_wildcard_router, prefix="/domains/-/grids", tags=["Grids"]
)
api_router.include_router(
    grids_router, prefix="/domains/{domain_id}/grids", tags=["Grids"]
)
api_router.include_router(
    inventories_wildcard_router,
    prefix="/domains/-/inventories",
    tags=["Inventories"],
)
api_router.include_router(
    inventories_router,
    prefix="/domains/{domain_id}/inventories",
    tags=["Inventories"],
)
api_router.include_router(
    point_clouds_wildcard_router,
    prefix="/domains/-/pointclouds",
    tags=["Point Clouds"],
)
api_router.include_router(
    point_clouds_router,
    prefix="/domains/{domain_id}/pointclouds",
    tags=["Point Clouds"],
)
api_router.include_router(keys_router, prefix="/keys", tags=["Keys"])
api_router.include_router(
    applications_router, prefix="/applications", tags=["Applications"]
)
api_router.include_router(users_router, prefix="/users", tags=["Users"])

# Features
api_router.include_router(
    features_wildcard_router, prefix="/domains/-/features", tags=["Features"]
)
api_router.include_router(
    features_router, prefix="/domains/{domain_id}/features", tags=["Features"]
)

# Include router with authentication middleware
app.include_router(api_router, dependencies=[Depends(authenticate_user)])


# Simplify operation IDs
for route in app.routes:
    if isinstance(route, APIRoute):
        route.operation_id = route.name
