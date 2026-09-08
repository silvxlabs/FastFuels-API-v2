"""
api/v2/resources/inventories/tree/pim/fusion/chm/router.py

Router for PIM-CHM fusion inventory creation.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    HTTPException,
    Request,
    Response,
    status,
)

from api.db.documents import get_document_async, set_document_async
from api.dependencies import VerifiedDomain
from api.quota import QUOTA_429_RESPONSE, enforce_create_quotas, register_dispatch
from api.resources.grids.utils import (
    validate_band_unit,
    validate_grid_dimensionality,
    validate_grid_has_band,
    validate_grid_has_georeference,
)
from api.resources.inventories.schema import BASE_INVENTORY_COLUMNS, Inventory
from api.resources.inventories.tree.pim.fusion.chm.examples import (
    CREATE_PIM_CHM_FUSION_OPENAPI_EXAMPLES,
)
from api.resources.inventories.tree.pim.fusion.chm.schema import (
    CreatePimChmFusionInventoryRequest,
    PimChmFusionInventorySource,
    ReimputationMethod,
)
from api.resources.inventories.utils import (
    validate_feature_conditions,
    validate_inventory_wide_treatment_area,
)
from api.resources.modifications import stringify_modification_coordinates
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import (
    GRIDS_COLLECTION,
    INVENTORIES_COLLECTION,
    STANDGEN_QUEUE,
    STANDGEN_SERVICE,
)

router = APIRouter()

COLLECTION = INVENTORIES_COLLECTION

# The plot-id band a PIM grid must carry for expansion. Keyed like the base
# tree/pim endpoint; TreeMap is the only PIM product today.
PLOT_ID_BAND = "tm_id"


def _grid_cell_size(grid_data: dict) -> float:
    """Horizontal cell size (meters) from a grid's affine transform.

    ``transform`` is ``(a, b, c, d, e, f)`` where ``a`` is the x pixel size.
    Grids live in metric CRSs, so this is the cell edge in meters.
    """
    return abs(grid_data["georeference"]["transform"][0])


@router.post(
    "",
    response_model=Inventory,
    status_code=status.HTTP_201_CREATED,
    summary="Create an inventory by fusing a PIM with a CHM",
    responses=QUOTA_429_RESPONSE,
)
async def create_pim_chm_fusion_inventory(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreatePimChmFusionInventoryRequest,
        Body(openapi_examples=CREATE_PIM_CHM_FUSION_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create PIM-CHM Fusion Inventory

    Expands a Plot Imputation Map (PIM) into individual tree records, conditioned
    on a Canopy Height Model (CHM). The path names the fused sources; the
    ``method`` object names the algorithm.

    ## Method: `reimputation` (default)

    The v1 fusion algorithm:

    1. Resample the PIM to ``method.resolution``.
    2. Compute the CHM's canopy cover per resampled cell — the fraction of CHM
       cells taller than ``method.min_height``.
    3. Keep a cell's plot only where cover exceeds ``method.cover_threshold``;
       cells at or below become gaps with no trees.
    4. Expand the surviving plots into trees exactly as ``tree/pim`` does.

    The result preserves each plot's species composition and size distributions
    while restricting trees to where the CHM actually shows canopy.

    ## Request Body

    - **source_pim_grid_id**: (required) ID of a completed PIM grid.
    - **source_chm_grid_id**: (required) ID of a completed CHM grid (band `chm`,
      unit meters).
    - **method**: (optional) Fusion algorithm and its parameters. Defaults to
      `reimputation`.
    - **seed**: (optional) Random seed for reproducibility. Random if omitted.
    - **point_process**: (optional) Spatial point process for coordinate
      assignment. Default: ``"inhomogeneous_poisson"``.
    - **modifications** / **treatments**: (optional) Applied after expansion,
      as on ``tree/pim``.
    - **type**, **name**, **description**, **tags**: (optional) Metadata.

    ## Response

    Returns the created Inventory resource with status ``"pending"``. The backend
    (Standgen) processes the fusion asynchronously and sets status to
    ``"completed"`` when ready.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    await validate_feature_conditions(
        [*body.modifications, *body.treatments], owner_id, domain_id
    )
    validate_inventory_wide_treatment_area(domain, body.treatments)

    # Validate the source PIM grid: owned, in this domain, completed.
    _, pim_snapshot = await get_document_async(
        GRIDS_COLLECTION,
        body.source_pim_grid_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    pim_grid_data = pim_snapshot.to_dict()

    # Gate on the band contract, not on provenance: any grid carrying the plot-id
    # band can drive PIM expansion, whoever produced it. Standgen reads this band.
    validate_grid_has_band(pim_grid_data, body.source_pim_grid_id, PLOT_ID_BAND)

    # Validate the source CHM grid: owned, in this domain, completed, with a
    # `chm` band in meters (the reimputation cover step compares heights to a
    # meter threshold without converting units).
    _, chm_snapshot = await get_document_async(
        GRIDS_COLLECTION,
        body.source_chm_grid_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    chm_grid_data = chm_snapshot.to_dict()
    validate_grid_has_band(chm_grid_data, body.source_chm_grid_id, "chm")
    validate_band_unit(chm_grid_data, body.source_chm_grid_id, "chm", "m")
    # A CHM is a surface: the cover step reads a single height per cell, so a
    # volumetric (3-D) grid is rejected rather than silently mis-read.
    validate_grid_dimensionality(chm_grid_data, body.source_chm_grid_id, 2)

    # Reimputation resamples the PIM to `resolution` and reads CHM cover there, so
    # the resolution can be no finer than the CHM cell (no cover detail to gain)
    # and no coarser than the PIM cell (plots would be lost). Read cell sizes from
    # the stored georeferences.
    if isinstance(body.method, ReimputationMethod):
        validate_grid_has_georeference(pim_grid_data, body.source_pim_grid_id)
        validate_grid_has_georeference(chm_grid_data, body.source_chm_grid_id)
        chm_cell = _grid_cell_size(chm_grid_data)
        pim_cell = _grid_cell_size(pim_grid_data)
        tol = 1e-6
        if not (chm_cell <= body.method.resolution + tol):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"Reimputation resolution {body.method.resolution} m is finer "
                    f"than the CHM cell size {chm_cell} m. The resolution must be "
                    f"no finer than the CHM cell."
                ),
            )
        if not (body.method.resolution <= pim_cell + tol):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"Reimputation resolution {body.method.resolution} m is coarser "
                    f"than the PIM cell size {pim_cell} m. The resolution must be "
                    f"no coarser than the PIM cell."
                ),
            )

    inventory_id = uuid.uuid4().hex
    request_time = datetime.now()
    source = PimChmFusionInventorySource(
        source_pim_grid_id=body.source_pim_grid_id,
        source_pim_grid_checksum=pim_grid_data.get("checksum"),
        source_chm_grid_id=body.source_chm_grid_id,
        source_chm_grid_checksum=chm_grid_data.get("checksum"),
        method=body.method,
        point_process=body.point_process,
        seed=body.seed,
    )

    inventory_data = {
        "id": inventory_id,
        "checksum": uuid.uuid4().hex,
        "domain_id": domain_id,
        "type": body.type.value,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "progress": None,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(),
        "modifications": stringify_modification_coordinates(
            [m.model_dump() for m in body.modifications]
        ),
        "treatments": stringify_modification_coordinates(
            [t.model_dump() for t in body.treatments]
        ),
        "columns": [c.model_dump() for c in BASE_INVENTORY_COLUMNS],
        "georeference": None,
        "error": None,
        "tags": body.tags,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, inventory_id, inventory_data)

    # Enqueue task to Standgen for processing
    await create_http_task_async(STANDGEN_QUEUE, STANDGEN_SERVICE, inventory_id)
    register_dispatch(request, response, background_tasks)

    return Inventory(**inventory_data)
