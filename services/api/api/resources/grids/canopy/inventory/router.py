"""
api/v2/resources/grids/canopy/inventory/router.py

Router for deriving 2D canopy fuel grids directly from a tree inventory.

Lives in its own module rather than canopy/router.py because its schema,
examples, and validation are substantial enough to warrant their own files;
like every other canopy endpoint, it dispatches to griddle.
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
from api.resources.grids.canopy.inventory.examples import (
    CREATE_INVENTORY_CANOPY_OPENAPI_EXAMPLES,
)
from api.resources.grids.canopy.inventory.schema import (
    DEFAULT_INVENTORY_CANOPY_RESOLUTION_M,
    AllometryCanopyBiomassSource,
    CanopyFuelcalcCrownClassAdjustment,
    CanopySpeciesInclusion,
    CanopyVerticalDistribution,
    CreateInventoryCanopyRequest,
    InventoryCanopySource,
    build_inventory_canopy_bands,
)
from api.resources.grids.schema import CHUNK_SHAPE, Grid
from api.resources.grids.utils import validate_target_grid_alignment
from api.resources.grids.voxelize.inventory.tree.schema import (
    InventoryColumnMaxCrownRadiusSource,
)
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import (
    GRIDDLE_QUEUE,
    GRIDDLE_SERVICE,
    GRIDS_COLLECTION,
    INVENTORIES_COLLECTION,
)

router = APIRouter()

COLLECTION = GRIDS_COLLECTION

# Columns the allometry endpoint can impute from position + height; position
# and height themselves cannot be imputed.
ALLOMETRY_IMPUTABLE_COLUMNS = frozenset({"dbh", "crown_ratio", "fia_species_code"})


def _required_columns(body: CreateInventoryCanopyRequest) -> set[str]:
    """Columns the selected methods read from the inventory.

    Position and the crown interval are always needed; species, diameter,
    and user-named columns are needed only by the methods that consume them.
    """
    required = {"x", "y", "height", "crown_ratio"}
    if isinstance(body.biomass_source, AllometryCanopyBiomassSource):
        required |= {"dbh", "fia_species_code"}
    else:
        required.add(body.biomass_source.column)
    if body.vertical_distribution is CanopyVerticalDistribution.reinhardt_2006:
        required.add("fia_species_code")
    if body.species_inclusion is CanopySpeciesInclusion.fuelcalc_default:
        required.add("fia_species_code")
    if isinstance(body.crown_class_adjustment, CanopyFuelcalcCrownClassAdjustment):
        required.add("fia_species_code")
    if isinstance(body.max_crown_radius_source, InventoryColumnMaxCrownRadiusSource):
        required.add(body.max_crown_radius_source.column)
    return required


@router.post(
    "",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a canopy fuel grid from a tree inventory",
    responses=QUOTA_429_RESPONSE,
)
async def create_inventory_canopy_grid(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateInventoryCanopyRequest,
        Body(openapi_examples=CREATE_INVENTORY_CANOPY_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Inventory Canopy Grid

    Derives the canopy fuel metrics operational fire models consume — canopy
    bulk density (`cbd`), canopy base height (`cbh`), canopy height (`chm`),
    canopy cover (`cc`), and optionally canopy fuel load (`cfl`) — directly
    from a tree inventory, with no voxelization.

    For each tree, available canopy fuel is estimated from crown biomass,
    distributed vertically over the crown, and attributed to output cells;
    each cell's vertical profile is then reduced to the requested bands. This
    is the FuelCalc-style profile method computed per cell from real stem
    positions instead of per plot from expanded tree records. Bands share
    keys and units with the LANDFIRE canopy source, so the result drops into
    anything that accepts one — including the landscape export.

    Only live trees contribute canopy fuel, matching FuelCalc's exclusion of
    dead trees.

    ## Request Body

    - **source_inventory_id**: (required) ID of a completed tree inventory in
      this domain. Required columns depend on the selected methods; the
      defaults need `x`, `y`, `height`, `crown_ratio`, `dbh`, and
      `fia_species_code`.
    - **alignment**: (optional) Output lattice. Against the domain (the
      default) `resolution` defaults to 30 m — an inventory has no native
      cell size to inherit. Against another grid, omitting `resolution`
      matches that grid's lattice exactly. `target: "native"` is not
      supported.
    - **bands**: (optional) Defaults to `["cbd", "cbh", "chm", "cc"]` — the
      four landscape-file canopy roles. Add `cfl` for canopy fuel load.
    - **biomass_source**: (optional) `allometry` with `nsvb` (default),
      `jenkins`, or `brown_1978` equations, or `inventory_column` carrying
      precomputed per-tree available canopy fuel.
    - **available_fuel**: (optional) Foliage fraction plus the fine-branchwood
      size partition and fraction. Resolved to `null` with an
      `inventory_column` biomass source.
    - **species_inclusion**, **crown_class_adjustment**, **min_tree_height**:
      (optional) Which trees contribute, and how crown weight is adjusted for
      canopy position.
    - **vertical_distribution**, **layer_depth**: (optional) How each tree's
      fuel stacks over its crown, and the profile layer depth (default
      0.3048 m, FuelCalc's 1 ft).
    - **horizontal_distribution**: (optional) `crown_projected` (default)
      splits each tree's fuel over the cells its crown covers;
      `stem` assigns it to the stem cell.
    - **max_crown_radius_source**: (optional) Allometric crown radii
      (default) or a per-tree inventory column (e.g. from LiDAR).
    - **cbd**, **cbh**, **chm**, **cc**: (optional) Per-band reduction
      methods. Each may only be supplied when its band is requested;
      requested bands default to the FuelCalc-style methods.
    - **name**, **description**, **tags**: (optional) Standard metadata.

    The stored grid `source` records every resolved choice, including
    defaults, so the grid is exactly reproducible from the resource alone.

    ## Response

    Returns the created Grid with status `"pending"` and
    `georeference: null`. Griddle computes the canopy metrics asynchronously
    and updates the grid to `"completed"` with a 2D `Georeference` when done.
    """
    owner_id = request.state.id
    domain_id = domain["id"]

    await enforce_create_quotas(COLLECTION, request)

    # Validate source inventory: exists, owned, in this domain, and completed.
    _, inventory_snapshot = await get_document_async(
        INVENTORIES_COLLECTION,
        body.source_inventory_id,
        owner_id=owner_id,
        domain_id=domain_id,
        document_status="completed",
    )
    inventory_data = inventory_snapshot.to_dict()

    # Must be a tree inventory — canopy fuel profiles are built from tree crowns.
    if inventory_data.get("type") != "tree":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Inventory '{body.source_inventory_id}' has type "
                f"'{inventory_data.get('type')}'. This endpoint requires "
                f"a tree inventory."
            ),
        )

    # Must carry the columns the selected methods read. Reject early with
    # tailored guidance rather than dispatching a job that fails on an opaque
    # read: morphology columns can be imputed with the allometry endpoint;
    # position, height, and user-named columns cannot.
    have_columns = {
        c["key"] if isinstance(c, dict) else c
        for c in inventory_data.get("columns", [])
    }
    missing_columns = _required_columns(body) - have_columns
    if missing_columns:
        imputable_missing = sorted(missing_columns & ALLOMETRY_IMPUTABLE_COLUMNS)
        source_only_missing = sorted(missing_columns - ALLOMETRY_IMPUTABLE_COLUMNS)
        guidance = []
        if imputable_missing:
            guidance.append(
                f"Impute {imputable_missing} with the allometry endpoint (POST "
                f"/domains/{domain_id}/inventories/tree/allometry/gdam with "
                f"source_tree_inventory_id='{body.source_inventory_id}'), then "
                f"derive canopy metrics from the resulting inventory."
            )
        if source_only_missing:
            guidance.append(
                f"Column(s) {source_only_missing} cannot be imputed and must be "
                f"present in the inventory's source data."
            )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Inventory '{body.source_inventory_id}' is missing column(s) "
                f"{sorted(missing_columns)} required by the selected canopy "
                f"methods. " + " ".join(guidance)
            ),
        )

    # An inventory has no source raster whose pixel anchor could be preserved.
    if body.alignment.target == "native":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "alignment.target 'native' is not supported for an inventory "
                "source: there is no source raster whose pixel anchor could "
                "be preserved. Use 'domain' or 'grid'."
            ),
        )

    await validate_target_grid_alignment(body.alignment, owner_id, domain_id)

    # Against a grid, `resolution: null` already means something — match that
    # grid's lattice cell-for-cell — so it is left alone. Against the domain
    # there is nothing for null to defer to, so resolve the 30 m default here
    # and store it, making the grid a record of exactly what it was built at.
    alignment = body.alignment
    if alignment.target != "grid" and alignment.resolution is None:
        alignment = alignment.model_copy(
            update={"resolution": DEFAULT_INVENTORY_CANOPY_RESOLUTION_M}
        )

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()

    source = InventoryCanopySource(
        source_inventory_id=body.source_inventory_id,
        source_inventory_checksum=inventory_data.get("checksum"),
        alignment=alignment,
        bands=body.bands,
        biomass_source=body.biomass_source,
        available_fuel=body.available_fuel,
        species_inclusion=body.species_inclusion,
        crown_class_adjustment=body.crown_class_adjustment,
        min_tree_height=body.min_tree_height,
        vertical_distribution=body.vertical_distribution,
        layer_depth=body.layer_depth,
        horizontal_distribution=body.horizontal_distribution,
        max_crown_radius_source=body.max_crown_radius_source,
        cbd=body.cbd,
        cbh=body.cbh,
        chm=body.chm,
        cc=body.cc,
    )
    bands = build_inventory_canopy_bands(body.bands)

    grid_data = {
        "id": grid_id,
        "checksum": uuid.uuid4().hex,
        "domain_id": domain_id,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "created_on": request_time,
        "modified_on": request_time,
        # No exclude_none: explicit nulls are load-bearing choices here — a
        # null relative_threshold_fraction is a flat threshold, a null cbd
        # window is an unsmoothed profile — and dropping them would resurrect
        # the defaults when the stored source is re-parsed.
        "source": source.model_dump(mode="json"),
        # Treevox-backed grids do not support modifications — always empty.
        "modifications": [],
        "bands": [b.model_dump() for b in bands],
        "georeference": None,
        "tags": body.tags,
        "chunks": {"shape": CHUNK_SHAPE, "count": None, "count_by_axis": None},
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)

    # Enqueue task to griddle for canopy metric computation.
    await create_http_task_async(GRIDDLE_QUEUE, GRIDDLE_SERVICE, grid_id)
    register_dispatch(request, response, background_tasks)

    return Grid(**grid_data)
