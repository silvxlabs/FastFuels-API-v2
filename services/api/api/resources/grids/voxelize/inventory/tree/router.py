"""
api/v2/resources/grids/voxelize/inventory/tree/router.py

Router for voxelizing a tree inventory into a 3D tree fuel grid.

Takes a tree inventory resource and voxelizes it into a 3D grid of canopy
fuel properties using crown profile and biomass models from fastfuels-core.
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
from api.resources.features.schema import FeatureType
from api.resources.grids.schema import Grid
from api.resources.grids.utils import incomplete_inventory_columns
from api.resources.grids.voxelize.inventory.tree.examples import (
    CREATE_TREE_INVENTORY_OPENAPI_EXAMPLES,
)
from api.resources.grids.voxelize.inventory.tree.schema import (
    VOXELIZE_REQUIRED_COLUMNS,
    BiomassComponent,
    CreateTreeInventoryRequest,
    CrownFeatureCrownGeometrySource,
    InventoryColumnMaxCrownRadiusSource,
    TreeInventoryVoxelizationSource,
    build_tree_bands,
)
from api.resources.inventories.schema import TREE_ID_COLUMN
from api.schema import JobStatus
from api.tasks import create_http_task_async
from lib.config import (
    FEATURES_COLLECTION,
    GRIDS_COLLECTION,
    INVENTORIES_COLLECTION,
    TREEVOX_QUEUE,
    TREEVOX_SERVICE,
)
from lib.inventory_io import IMPUTABLE_COLUMNS, allometry_request

router = APIRouter()

COLLECTION = GRIDS_COLLECTION


@router.post(
    "",
    response_model=Grid,
    status_code=status.HTTP_201_CREATED,
    summary="Create a 3D tree fuel grid from a tree inventory",
    responses=QUOTA_429_RESPONSE,
)
async def create_tree_inventory_grid(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    domain: VerifiedDomain,
    body: Annotated[
        CreateTreeInventoryRequest,
        Body(openapi_examples=CREATE_TREE_INVENTORY_OPENAPI_EXAMPLES),
    ],
):
    """
    # Create Tree Inventory Grid

    Voxelizes a tree inventory into a 3D canopy fuel grid. Each tree's crown
    is discretized onto the voxel grid using a species-specific crown profile
    model, and per-voxel fuel properties (bulk density, moisture, SAV) are
    computed from biomass and moisture models.

    This is a 3D grid product — resampling and modifications are not
    supported. Apply modifications to the source inventory before voxelizing.

    ## Request Body

    - **source_inventory_id**: (required) ID of a completed tree inventory.
      Its `x`, `y`, `height`, `dbh`, `crown_base_height` and `fia_species_code`
      (and any biomass column read) must have no null values; a null
      `fia_status_code` counts as live.
    - **resolution**: (optional) Voxel resolution in meters. Defaults to
      `{"horizontal": 2.0, "vertical": 1.0}`. All components must be positive.
    - **bands**: (optional) Which output bands to produce. Defaults to
      `["bulk_density.foliage.live"]`. Must be non-empty and contain no
      duplicates. Branchwood and fine bands are accepted by the API, but
      Treevox currently fails those jobs with a not-implemented processing
      error.
    - **crown_profile_model**: (optional) Crown shape. Each tree's crown runs
      from its `crown_base_height` to its height.
      - `purves` (default): Purves et al. (2007), species-specific; widest at
        the crown base.
      - `beta`: species-group beta distribution; widest within the crown, per
        species group.
      - `cone`: right circular cone; widest at the crown base.
      - `cylinder`: right circular cylinder; uniform radius.
      - `single_ellipsoid`: one half-ellipsoid of revolution; widest at the
        crown base.
      - `dual_ellipsoid`: two half-ellipsoids joined at the crown midpoint;
        widest there.
      - `single_paraboloid`: one paraboloid of revolution; widest at the crown
        base.
      - `dual_paraboloid`: two paraboloids joined at the crown midpoint;
        widest there (the LANL Trees crown envelope).
    - **biomass_source**: (optional) Biomass source and requested components. The
      default uses NSVB allometry for foliage. Inventory-column sources must
      provide per-tree kg values for each requested direct component.
    - **max_crown_radius_source**: (optional) Source of each tree's maximum
      crown radius. The default, `{"type": "allometry"}`, uses the Purves et
      al. (2007) maximum crown radius for `purves` and every geometric shape
      (`cone`, `cylinder`, ellipsoids, paraboloids), so a tree's crown is
      equally wide under each of them and only the shape changes; `beta` uses
      its own allometric radius. Pass
      `{"type": "inventory_column", "column": <name>}` to read a per-tree
      maximum radius (m) from an inventory column (e.g. derived from LiDAR).
      The crown profile model still controls the crown shape — only the peak
      radius is rescaled. A tree with a null in that column uses its
      allometric radius.
    - **crown_geometry_source**: (optional) Where crown shapes come from.
      `{"type": "parametric"}` (default) draws every crown from
      `crown_profile_model`. `{"type": "crown_feature", "feature_id": <id>}`
      gives each tree with a crown in that crown Feature a measured crown,
      matched by `tree_id`:
      - Outline: the crown's CHM cells.
      - Top: each cell's CHM height.
      - Bottom: `crown_profile_model`, going up from the crown base. Profiles
        widest at the crown base give a flat bottom there; `beta` and the
        dual profiles give a bottom that rises toward the crown's edge. Each
        cell's bottom is at least one voxel below its top, but never below
        the crown base.

      `max_crown_radius_source` does not apply to measured crowns, and every
      tree keeps the mass `biomass_source` gives it. Trees with no crown in
      the Feature, trees whose crown does not contain their stem, and trees
      with no CHM height above their crown base, are drawn parametrically.
      Crowns of trees not in the inventory are ignored. The inventory must
      have a `tree_id` column, the Feature must be a completed crown Feature
      in this domain, and the CHM grid it was built from must still exist.
    - **max_bulk_density**: (optional) Upper limit on bulk density, in
      kg/m**3. In any voxel where a biomass component's bulk density, live
      and dead combined, exceeds the limit, that component's live and dead
      bulk density are scaled down to it, keeping their live/dead split, and
      `leaf_area_density` is scaled with foliage. Fuel above the limit is
      removed from the grid. Omit for no limit.
    - **moisture_model**: (optional) Live/dead fuel moisture configuration.
      Required shape: `{"live": {"method": "uniform", "value": <percent>}}`
      and/or `{"dead": {"method": "uniform", "value": <percent>}}`.
      Applied only when matching `fuel_moisture.*` bands are requested. Live
      defaults to 100%; dead defaults to 10%.
    - **name**, **description**, **tags**: (optional) Standard metadata.

    ## Response

    Returns the created Grid resource with status `"pending"` and
    `georeference: null`. The Treevox backend performs voxelization
    asynchronously and updates the grid to `"completed"` with a
    `Georeference3D` when done.

    With a `crown_feature` source, `source.source_chm_grid_id` and
    `source.source_chm_grid_checksum` record the crown Feature's CHM grid and
    its `checksum` at creation, and the completed grid's
    `source.crown_geometry_counts` gives the number of trees drawn with a
    measured crown (`measured`) and drawn parametrically because they had no
    crown in the Feature (`parametric_no_crown`), their crown did not contain
    their stem (`parametric_stem_outside_crown`), or they had no CHM height
    above their crown base (`parametric_no_volume`). Trees that cannot be
    drawn at all are not counted.
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

    # Must be a tree inventory — other entity types can't be voxelized this way.
    if inventory_data.get("type") != "tree":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Inventory '{body.source_inventory_id}' has type "
                f"'{inventory_data.get('type')}'. This endpoint requires "
                f"a tree inventory."
            ),
        )

    # Must carry complete values in the columns voxelization needs. An inventory
    # can lack them for more than one reason — a CHM-derived inventory (position
    # + height only) or an upload that omitted or left gaps in the optional
    # morphology columns — so tailor the guidance to which columns are missing
    # rather than assuming a source. Reject early rather than dispatching a job
    # that fails. A radius column must be listed (uploads store an all-null
    # column for unmapped optional columns), but a null radius is allowed: it
    # falls back to allometry.
    required_columns = set(VOXELIZE_REQUIRED_COLUMNS)
    foliage = getattr(body.biomass_source, "columns", {}).get(BiomassComponent.foliage)
    if foliage is not None:
        required_columns.add(foliage.column)
    if isinstance(body.crown_geometry_source, CrownFeatureCrownGeometrySource):
        required_columns.add(TREE_ID_COLUMN.key)
    nullable = set()
    if isinstance(body.max_crown_radius_source, InventoryColumnMaxCrownRadiusSource):
        required_columns.add(body.max_crown_radius_source.column)
        nullable.add(body.max_crown_radius_source.column)
    missing_columns = incomplete_inventory_columns(
        inventory_data, required_columns, nullable
    )
    if missing_columns:
        # Columns the source carries with no nulls: the crown base model's
        # inputs must be imputed with it when the source lacks them.
        source_keys = {
            c["key"] if isinstance(c, dict) else c
            for c in inventory_data.get("columns", [])
        }
        complete = source_keys - incomplete_inventory_columns(
            inventory_data, source_keys
        )
        request = allometry_request(
            missing_columns, complete, domain_id, body.source_inventory_id
        )
        source_only_missing = sorted(missing_columns - IMPUTABLE_COLUMNS)
        guidance = [request] if request else []
        if source_only_missing:
            guidance.append(
                f"Column(s) {source_only_missing} cannot be imputed and must be "
                f"present in the inventory's source data."
            )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Inventory '{body.source_inventory_id}' is missing, or has null "
                f"values in, column(s) {sorted(missing_columns)} required for "
                f"voxelization. " + " ".join(guidance)
            ),
        )

    # A crown_feature source needs a completed crown Feature in this domain and
    # the CHM grid it was built from, whose tops treevox reads.
    source_chm_grid_id = source_chm_grid_checksum = None
    if isinstance(body.crown_geometry_source, CrownFeatureCrownGeometrySource):
        feature_id = body.crown_geometry_source.feature_id
        _, feature_snapshot = await get_document_async(
            FEATURES_COLLECTION,
            feature_id,
            owner_id=owner_id,
            domain_id=domain_id,
            document_status="completed",
        )
        feature = feature_snapshot.to_dict()
        if feature.get("type") != FeatureType.crown.value:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"Feature '{feature_id}' has type '{feature.get('type')}'. "
                    f"crown_geometry_source requires a crown Feature."
                ),
            )
        source_chm_grid_id = feature["source"]["source_chm_grid_id"]
        try:
            _, chm_snapshot = await get_document_async(
                GRIDS_COLLECTION,
                source_chm_grid_id,
                owner_id=owner_id,
                domain_id=domain_id,
                document_status="completed",
            )
        except HTTPException as e:
            # The request names the Feature, not its CHM grid, so a deleted
            # CHM is a problem with the Feature rather than a missing resource.
            if e.status_code != status.HTTP_404_NOT_FOUND:
                raise
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"CHM grid '{source_chm_grid_id}', which crown Feature "
                    f"'{feature_id}' was built from, no longer exists. Build the "
                    f"crown Feature again from a current CHM grid."
                ),
            ) from e
        source_chm_grid_checksum = chm_snapshot.to_dict().get("checksum")

    grid_id = uuid.uuid4().hex
    request_time = datetime.now()

    source = TreeInventoryVoxelizationSource(
        source_inventory_id=body.source_inventory_id,
        source_inventory_checksum=inventory_data.get("checksum"),
        resolution=body.resolution,
        bands=body.bands,
        crown_profile_model=body.crown_profile_model,
        biomass_source=body.biomass_source,
        max_crown_radius_source=body.max_crown_radius_source,
        crown_geometry_source=body.crown_geometry_source,
        max_bulk_density=body.max_bulk_density,
        source_chm_grid_id=source_chm_grid_id,
        source_chm_grid_checksum=source_chm_grid_checksum,
        moisture_model=body.moisture_model,
        seed=body.seed,
    )
    bands = build_tree_bands(body.bands)

    grid_data = {
        "id": grid_id,
        "checksum": uuid.uuid4().hex,
        "domain_id": domain_id,
        "name": body.name,
        "description": body.description,
        "status": JobStatus.pending.value,
        "created_on": request_time,
        "modified_on": request_time,
        "source": source.model_dump(mode="json", exclude_none=True),
        # 3D grids do not support modifications — always empty.
        "modifications": [],
        "bands": [b.model_dump() for b in bands],
        "georeference": None,
        "tags": body.tags,
        # chunks is computed by Treevox once the 3D grid dimensions
        # are known from the domain bbox + resolution.
        "chunks": None,
        "owner_id": owner_id,
    }

    await set_document_async(COLLECTION, grid_id, grid_data)

    # Enqueue task to Treevox for voxelization.
    await create_http_task_async(TREEVOX_QUEUE, TREEVOX_SERVICE, grid_id)
    register_dispatch(request, response, background_tasks)

    return Grid(**grid_data)
