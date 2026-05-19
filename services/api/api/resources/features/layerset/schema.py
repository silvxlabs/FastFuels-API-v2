"""
api/v2/resources/features/layerset/schema.py

Schemas for the Layerset feature type.

The upload payload is a standard, flat GeoJSON FeatureCollection where every
Feature carries one fuelbed's attributes on ``properties`` and a polygon on
``geometry``. The ``properties`` block mirrors the per-row column contract
of ``fastfuels_core.rasterize_layerset`` directly so the downstream worker
can build the input GeoDataFrame with a single ``gpd.read_file`` call.

See ``fastfuels_core.layersets`` for the rasterizer's column documentation.
"""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from api.resources.features.schema import CreateFeatureRequestBase, FeatureType


class LayersetSource(BaseModel):
    """Source tracking for user-uploaded layersets."""

    product: str = "Upload"
    description: str = "User-uploaded layerset"


# --- Per-feature properties contract -----------------------------------------


class Distribution(StrEnum):
    """Per-cell spatial-distribution mode for a fuelbed.

    Mirrors ``fastfuels_core.rasterize_layerset``'s ``distribution`` column.
    """

    homogeneous = "homogeneous"
    uniform_random = "uniform_random"
    random_clusters = "random_clusters"


class LayersetProperties(BaseModel):
    """Per-feature properties — one row of input to ``rasterize_layerset``.

    Required fields match the rasterizer's required input columns. Optional
    fields map to the rasterizer's optional bands; omitting them leaves the
    corresponding output band as NaN.
    """

    # Optional traceability identifier carried by the team's example payloads
    # (e.g. ``"Shrub1_52"``); not consumed by the rasterizer itself.
    strata_fb: str | None = None

    # Required per fastfuels_core.rasterize_layerset
    fuel_type: str
    fuel_loading: float
    fuel_height: float
    percent_cover: float = Field(ge=0, le=100)
    distribution: Distribution

    # Required only when ``distribution == "random_clusters"``; validator below.
    patch_size: float | None = None

    # Optional rasterizer inputs — produce NaN output bands when omitted.
    live_fuel_moisture: float | None = None
    dead_fuel_moisture: float | None = None
    heat_of_combustion: float | None = None
    patch_std_dev: float | None = None

    @model_validator(mode="after")
    def patch_size_required_for_random_clusters(self) -> "LayersetProperties":
        if (
            self.distribution is Distribution.random_clusters
            and self.patch_size is None
        ):
            raise ValueError(
                "patch_size is required when distribution == 'random_clusters'"
            )
        return self


# --- GeoJSON geometry + feature wrappers -------------------------------------


class LayersetMultiPolygon(BaseModel):
    """MultiPolygon geometry attached to each Feature.

    Coordinates are intentionally typed as ``list[Any]`` to accept the
    deeply nested ring arrays without locking the schema to a specific
    coordinate dimensionality.
    """

    type: Literal["MultiPolygon"] = "MultiPolygon"
    coordinates: list[Any] = Field(default_factory=list)


class LayersetFeature(BaseModel):
    """One Feature in the layerset FeatureCollection."""

    type: Literal["Feature"] = "Feature"
    properties: LayersetProperties
    geometry: LayersetMultiPolygon


class LayersetCrs(BaseModel):
    """Optional GeoJSON crs block.

    Per RFC 7946, ``crs`` is deprecated at the GeoJSON level, but the team's
    pipeline emits it and downstream consumers (geopandas, this server) read
    it to anchor the bounds extracted in the upload router.
    """

    type: str = "name"
    properties: dict[str, Any] = Field(default_factory=dict)


class LayersetFeatureCollection(BaseModel):
    """Standard GeoJSON FeatureCollection of fuelbed polygons.

    ``name`` and ``crs`` are optional and pass through to the stored GeoJSON
    without server-side reinterpretation.
    """

    type: Literal["FeatureCollection"] = "FeatureCollection"
    name: str | None = None
    crs: LayersetCrs | None = None
    features: list[LayersetFeature] = Field(default_factory=list)


# --- Request Body ------------------------------------------------------------


class CreateLayersetRequestBody(CreateFeatureRequestBase):
    """Request body for uploading a flat GeoJSON layerset."""

    type: Literal[FeatureType.layerset] = FeatureType.layerset
    geojson: LayersetFeatureCollection = Field(
        ...,
        description="A flat GeoJSON FeatureCollection of fuelbed polygons.",
    )
