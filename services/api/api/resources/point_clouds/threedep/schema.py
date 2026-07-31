"""
api/v2/resources/point_clouds/threedep/schema.py

Schema models for creating a point cloud from USGS 3DEP.

3DEP is the USGS 3D Elevation Program, which publishes the nation's public
airborne lidar. Because 3DEP is airborne by definition, this source always
produces an `als` point cloud — there is no acquisition type to choose.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CreateThreeDepPointCloudRequest(BaseModel):
    """Request body for fetching a point cloud from USGS 3DEP."""

    name: str = Field(
        "",
        max_length=255,
        description="Human-readable name for the point cloud.",
    )
    description: str = Field(
        "",
        max_length=2000,
        description="Longer free-text description of the point cloud.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for organizing and filtering point clouds.",
    )
    datasets: list[str] | None = Field(
        None,
        max_length=20,
        description=(
            "Names of the 3DEP acquisitions to read, in priority order. Omit "
            "this to let the backend choose, which it does by preferring a "
            "single acquisition that covers the whole domain and otherwise "
            "combining the fewest acquisitions that fill it. Set it to pin the "
            "result to specific acquisitions — for example to force a "
            "higher-density or more recent survey where several overlap. Where "
            "two listed acquisitions overlap, the one listed first is used. "
            "Names come from the coverage endpoint; every name must exist and "
            "overlap the domain."
        ),
        examples=[["CO_CentralEasternPlains_1_2020"]],
    )
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "Bridger ALS",
                    "description": "3DEP airborne lidar over the study area.",
                    "tags": ["bridger"],
                }
            ]
        }
    )


class ThreeDepDatasetCoverage(BaseModel):
    """One 3DEP acquisition available over a domain, and what it would supply."""

    name: str = Field(
        ...,
        description=(
            "USGS acquisition name. Pass it in `datasets` on a create request "
            "to pin the fetch to this acquisition."
        ),
        examples=["WY_Southwest_1_2020"],
    )
    url: str = Field(
        ...,
        description="Location of the acquisition's Entwine Point Tile index.",
    )
    contribution_fraction: float = Field(
        ...,
        description=(
            "Fraction of the domain this acquisition would supply, from `0.0` "
            "to `1.0`. Acquisitions overlap each other freely, so this is the "
            "share left over after the acquisitions listed before it have "
            "taken their part — not the raw overlap. The values are therefore "
            "disjoint and sum to `coverage_fraction`."
        ),
        examples=[0.81],
    )
    estimated_density: float = Field(
        ...,
        description=(
            "Average point density of the acquisition, in points per square "
            "metre, computed over its full published extent."
        ),
        examples=[6.81],
    )
    estimated_points: int = Field(
        ...,
        description=(
            "Approximate number of points this acquisition would contribute. "
            "Derived from its density and the area it covers, so treat it as "
            "an order-of-magnitude figure rather than an exact count."
        ),
        examples=[1702335],
    )


class ThreeDepCoverageResponse(BaseModel):
    """Response model for the 3DEP point cloud coverage pre-flight check."""

    available: bool = Field(
        ...,
        description=(
            "Whether any 3DEP lidar covers this domain. When false, a create "
            "request for this domain is rejected."
        ),
    )
    coverage_fraction: float = Field(
        ...,
        description=(
            "Fraction of the domain covered by 3DEP lidar, from `0.0` to "
            "`1.0`. This is the union of every available acquisition, so it "
            "never exceeds 1.0 no matter how much the acquisitions overlap."
        ),
        examples=[1.0],
    )
    datasets: list[ThreeDepDatasetCoverage] = Field(
        ...,
        description=(
            "Acquisitions that would be read, in the order they would be used. "
            "Empty when no lidar covers the domain."
        ),
    )
    estimated_point_count: int = Field(
        ...,
        description=(
            "Approximate total number of points a fetch would return, summed "
            "across acquisitions."
        ),
        examples=[1702335],
    )
    point_budget: int = Field(
        ...,
        description=(
            "Maximum number of points a single fetch may return. Shrink the "
            "domain if the estimate exceeds it."
        ),
        examples=[200000000],
    )
    exceeds_point_budget: bool = Field(
        ...,
        description=(
            "Whether the estimate is over `point_budget`. When true, a create "
            "request for this domain is rejected, so check this before "
            "committing to a fetch."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "available": True,
                    "coverage_fraction": 1.0,
                    "datasets": [
                        {
                            "name": "WY_Southwest_1_2020",
                            "url": (
                                "https://s3-us-west-2.amazonaws.com/"
                                "usgs-lidar-public/WY_Southwest_1_2020/ept.json"
                            ),
                            "contribution_fraction": 1.0,
                            "estimated_density": 6.81,
                            "estimated_points": 1702335,
                        }
                    ],
                    "estimated_point_count": 1702335,
                    "point_budget": 200000000,
                    "exceeds_point_budget": False,
                }
            ]
        }
    )


class ThreeDepPointCloudSource(BaseModel):
    """Provenance recorded on a point cloud fetched from USGS 3DEP."""

    name: Literal["3dep"] = "3dep"
    datasets: list[str] = Field(
        default_factory=list,
        description=(
            "Acquisitions the cloud was built from, in the order they were "
            "used. Recorded so the fetch can be reproduced exactly."
        ),
    )
    requested_datasets: list[str] | None = Field(
        None,
        description=("Acquisitions the request pinned, or null if the backend chose."),
    )
    coverage_fraction: float = Field(
        ...,
        description=(
            "Fraction of the domain the cloud covers. Below 1.0 the cloud has "
            "a gap, which summary statistics alone will not reveal: density is "
            "measured over the points that exist, so a partial cloud still "
            "reports a healthy density."
        ),
    )
    catalog_fetched_on: str | None = Field(
        None,
        description=(
            "When the 3DEP acquisition catalog was read. The catalog changes "
            "as USGS publishes new surveys, so an identical request can select "
            "differently at a later date."
        ),
    )
