"""
api/v2/resources/point_clouds/schema.py

Core schema models for the Point Cloud resource.

A point cloud is a collection of 3D points captured by a laser scanner, used as
a primitive input to the fuel-modeling pipeline (e.g. an airborne scan becomes a
canopy height model, which becomes a tree inventory). This module defines the
resource as stored and returned by the API. Source-specific creation requests
(uploading a file, fetching from USGS 3DEP) are added by their own routers and
are not part of this module.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from api.schema import JobError, JobProgress, JobStatus, PaginatedResponse


class PointCloudType(StrEnum):
    """How a point cloud was acquired.

    The acquisition platform determines the cloud's geometry and which downstream
    products it can feed, so it is recorded as a first-class, filterable field.

    - ``als`` — **Airborne Laser Scanning.** Captured from an aircraft or drone
      looking down. Covers large areas from above and is the basis for canopy
      height models and individual-tree detection. Available from an upload or
      from USGS 3DEP.
    - ``tls`` — **Terrestrial Laser Scanning.** Captured from a tripod-mounted
      scanner on the ground looking out and up. Resolves fine sub-canopy and
      stem structure over a small plot. Available from an upload only (3DEP is
      airborne and cannot produce terrestrial scans).
    """

    als = "als"
    tls = "tls"


class PointCloudSortField(StrEnum):
    """Fields available for sorting point cloud list results."""

    created_on = "created_on"
    modified_on = "modified_on"
    name = "name"


class PointCloudGeoreference(BaseModel):
    """The coordinate reference system and 3D extent of a point cloud.

    Populated by the backend after the cloud is ingested and inspected; it is
    ``null`` while the point cloud is still ``pending`` or ``running``.
    """

    crs: str = Field(
        ...,
        description=(
            "Coordinate reference system the points are stored in, as an "
            "authority code (e.g. `EPSG:32613`). Horizontal and vertical "
            "coordinates share this CRS."
        ),
        examples=["EPSG:32613", "EPSG:5070"],
    )
    bounds: tuple[float, float, float, float, float, float] = Field(
        ...,
        description=(
            "Axis-aligned 3D bounding box of every point, given as "
            "`[min_x, min_y, min_z, max_x, max_y, max_z]` in the units of "
            "`crs`. Point clouds are three-dimensional, so the box includes a "
            "vertical (z) extent. Use it to check coverage against a domain "
            "before deriving products from the cloud."
        ),
        examples=[[500000.0, 4300000.0, 1800.0, 501000.0, 4301000.0, 1950.0]],
    )


class PointCloudSummary(BaseModel):
    """Summary statistics describing the contents of a point cloud.

    Populated by the backend after the cloud is ingested and inspected; it is
    ``null`` while the point cloud is still ``pending`` or ``running``. Use it to
    gauge a cloud's size, density, and composition without downloading it.
    """

    point_count: int = Field(
        ...,
        description="Total number of points in the cloud.",
        examples=[12873402],
    )
    point_classes: list[int] = Field(
        ...,
        description=(
            "ASPRS standard classification codes present in the cloud, sorted "
            "ascending. Common codes include `1` (unclassified), `2` (ground), "
            "and `3`, `4`, `5` (low, medium, high vegetation)."
        ),
        examples=[[1, 2, 3, 4, 5]],
    )
    density: float = Field(
        ...,
        description=(
            "Average point density over the cloud's horizontal extent, in points "
            "per square meter."
        ),
        examples=[18.7],
    )


class PointCloud(BaseModel):
    """A laser-scanned point cloud scoped to a single domain.

    Point clouds are created asynchronously: a creation request returns
    immediately with ``status="pending"`` and the file is ingested in the
    background. While ``status`` is ``"pending"`` or ``"running"`` the
    derived fields (`georeference`) are ``null``; the backend fills them in once
    ingestion succeeds and flips ``status`` to ``"completed"``. If ingestion
    fails, ``status`` becomes ``"failed"`` and `error` explains why.

    A completed point cloud is an input you compose with other resources — most
    directly, an ALS cloud feeds a canopy-height-model grid, which feeds a tree
    inventory.
    """

    id: str = Field(
        ...,
        description="Unique 32-character hex identifier for this point cloud.",
    )
    domain_id: str = Field(
        ...,
        description="Identifier of the domain this point cloud belongs to.",
    )
    type: PointCloudType = Field(
        ...,
        description=(
            "How the cloud was acquired: `als` (airborne) or `tls` "
            "(terrestrial). Set when the point cloud is created and never "
            "changes. Filter a list by this field to separate airborne scans "
            "from terrestrial ones."
        ),
    )
    name: str = Field(
        "",
        description="Human-readable name for the point cloud.",
    )
    description: str = Field(
        "",
        description="Longer free-text description of the point cloud.",
    )
    status: JobStatus = Field(
        ...,
        description=(
            "Processing state: `pending` (queued), `running` (ingesting), "
            "`completed` (ready to use), or `failed`. Derived fields are "
            "populated only once `completed`."
        ),
    )
    progress: JobProgress | None = Field(
        default=None,
        description="Progress info while `status` is `running`. Null otherwise.",
    )
    created_on: datetime | None = Field(
        default=None,
        description="When the point cloud was created.",
    )
    modified_on: datetime | None = Field(
        default=None,
        description="When the point cloud was last modified.",
    )
    checksum: str | None = Field(
        default=None,
        description=(
            "Version marker for this point cloud's content. It changes each "
            "time the point cloud is rebuilt and is unaffected by metadata-only "
            "edits (name, description, tags). A resource derived from this point "
            "cloud stores the checksum it was built from; comparing that stored "
            "value against this field reveals whether this point cloud has "
            "changed since. May be null for point clouds created before "
            "checksums were introduced."
        ),
    )
    source: dict = Field(
        ...,
        description=(
            "Provenance of the point cloud — where its points came from. Always "
            "carries a `name` discriminator (`upload` for a user-supplied file, "
            "`3dep` for a USGS 3DEP fetch) alongside source-specific parameters. "
            "Stored verbatim so the cloud can be reproduced from its origin."
        ),
        examples=[{"name": "3dep"}, {"name": "upload"}],
    )
    georeference: PointCloudGeoreference | None = Field(
        default=None,
        description=(
            "Coordinate reference system and 3D extent of the points. Null "
            "until the backend finishes ingesting the cloud."
        ),
    )
    summary: PointCloudSummary | None = Field(
        default=None,
        description=(
            "Summary statistics — point count, classification codes present, and "
            "density — describing the cloud's contents. Null until the backend "
            "finishes ingesting the cloud."
        ),
    )
    error: JobError | None = Field(
        default=None,
        description=(
            "Details when `status` is `failed`. The traceback is stored but not "
            "exposed in API responses."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description="User-assigned tags for organizing and filtering point clouds.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                # A completed airborne cloud fetched from 3DEP: georeference and
                # checksum populated, ready to feed a canopy grid.
                {
                    "id": "9f1c2a7b4e0d4c8a9b2e1f3a5c6d7e8f",
                    "domain_id": "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d",
                    "type": "als",
                    "name": "Bridger ALS 2020",
                    "description": "3DEP airborne lidar over the Bridger study area.",
                    "status": "completed",
                    "progress": None,
                    "created_on": "2026-06-01T17:42:10Z",
                    "modified_on": "2026-06-01T17:48:55Z",
                    "checksum": "c0ffee00c0ffee00c0ffee00c0ffee00",
                    "source": {"name": "3dep"},
                    "georeference": {
                        "crs": "EPSG:32612",
                        "bounds": [
                            500000.0,
                            5060000.0,
                            1800.0,
                            501000.0,
                            5061000.0,
                            1980.0,
                        ],
                    },
                    "summary": {
                        "point_count": 12873402,
                        "point_classes": [1, 2, 3, 4, 5],
                        "density": 18.7,
                    },
                    "error": None,
                    "tags": ["bridger", "als"],
                },
                # A terrestrial scan still ingesting from an upload: derived
                # fields null until processing completes.
                {
                    "id": "0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a",
                    "domain_id": "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d",
                    "type": "tls",
                    "name": "Plot 14 TLS",
                    "description": "",
                    "status": "pending",
                    "progress": None,
                    "created_on": "2026-06-09T12:00:00Z",
                    "modified_on": "2026-06-09T12:00:00Z",
                    "checksum": "feedface00feedface00feedface0011",
                    "source": {"name": "upload"},
                    "georeference": None,
                    "summary": None,
                    "error": None,
                    "tags": ["plot-14"],
                },
            ]
        }
    )


class UpdatePointCloudRequestBody(BaseModel):
    """Request body for updating point cloud metadata.

    Only metadata is mutable. The point cloud's content, source, and derived
    fields cannot be changed through this endpoint, so updates never alter the
    `checksum`.
    """

    name: str | None = Field(
        None,
        max_length=255,
        description="New name. Omit to leave unchanged.",
    )
    description: str | None = Field(
        None,
        max_length=2000,
        description="New description. Omit to leave unchanged.",
    )
    tags: list[str] | None = Field(
        None,
        max_length=50,
        description="New tags (replaces the existing list). Omit to leave unchanged.",
    )


class DuplicatePointCloudRequest(BaseModel):
    """Optional metadata overrides for a duplicated point cloud.

    Every field is optional. Any field omitted is carried over verbatim from the
    source point cloud.
    """

    name: str | None = Field(
        None,
        max_length=255,
        description="Name for the copy. Omit to reuse the source point cloud's name.",
    )
    description: str | None = Field(
        None,
        max_length=2000,
        description=(
            "Description for the copy. Omit to reuse the source point cloud's "
            "description."
        ),
    )
    tags: list[str] | None = Field(
        None,
        max_length=50,
        description="Tags for the copy. Omit to reuse the source point cloud's tags.",
    )


class ListPointCloudsResponse(PaginatedResponse):
    """Paginated response for listing point clouds."""

    point_clouds: list[PointCloud]
