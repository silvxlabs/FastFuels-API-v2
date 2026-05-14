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


app = FastAPI(
    title="FastFuels API",
    description="A JSON API for creating, editing, and retrieving 3D fuels data for next generation fire behavior models.",
    version="2.0.0",
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


@app.get("/", tags=["Index"])
async def index():
    return {"message": "FastFuels API"}


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
api_router.include_router(keys_router, prefix="/keys", tags=["Keys"])
api_router.include_router(
    applications_router, prefix="/applications", tags=["Applications"]
)

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
