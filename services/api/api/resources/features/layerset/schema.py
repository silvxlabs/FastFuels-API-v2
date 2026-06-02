"""
api/v2/resources/features/layerset/schema.py

Schemas for the Layerset feature type.

The upload payload is a standard, flat GeoJSON FeatureCollection where every
Feature carries one fuelbed's attributes on ``properties`` and a polygon on
``geometry``. The ``properties`` block mirrors the per-row column contract
of ``fastfuels_core.rasterize_layerset`` directly so the downstream worker
can build the input GeoDataFrame with a single ``gpd.read_file`` call.

Geometry and feature wrappers come from ``geojson_pydantic`` (the same
package the Domains schema uses) so we get RFC 7946 coordinate validation
for free. We extend ``FeatureCollection`` to carry the optional GeoJSON
``crs``/``name`` block that ``fastfuels_core`` reads to anchor units and
projection.

See ``fastfuels_core.layersets`` for the rasterizer's column documentation.
"""

from enum import StrEnum
from typing import Any

from geojson_pydantic import Feature, FeatureCollection
from geojson_pydantic.geometries import MultiPolygon, Polygon
from pydantic import BaseModel, Field, model_validator


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


# --- GeoJSON wrappers (via geojson_pydantic) ---------------------------------


class LayersetFeature(Feature[Polygon | MultiPolygon, LayersetProperties]):
    """One Feature in the layerset FeatureCollection.

    Inherits coordinate validation from ``geojson_pydantic``. Both
    ``Polygon`` and ``MultiPolygon`` are accepted because standard tooling
    (QGIS, GDAL, geopandas) emits ``Polygon`` for single-ring features and
    ``MultiPolygon`` for multi-ring ones. ``properties`` is narrowed to
    non-Optional because every fuelbed row must carry the rasterizer's
    required columns.
    """

    properties: LayersetProperties


class LayersetCrs(BaseModel):
    """Optional GeoJSON crs block.

    Per RFC 7946, ``crs`` is deprecated at the GeoJSON level (and
    ``geojson_pydantic`` therefore omits it), but the team's pipeline emits
    it and downstream consumers (geopandas, this server) read it to anchor
    bounds and the projected-CRS check in the upload router.
    """

    type: str = "name"
    properties: dict[str, Any] = Field(default_factory=dict)


class LayersetFeatureCollection(FeatureCollection[LayersetFeature]):
    """GeoJSON FeatureCollection of fuelbed polygons.

    Extends ``geojson_pydantic.FeatureCollection`` with the optional
    GeoJSON ``crs`` and ``name`` members (RFC 7946 deprecates ``crs``, but
    we need it to anchor the projected-CRS check in the upload router).
    At least one Feature is required — ``fastfuels_core.rasterize_layerset``
    rejects empty inputs, so we surface the error at upload time rather
    than at rasterize time.
    """

    name: str | None = None
    crs: LayersetCrs | None = None

    @model_validator(mode="after")
    def at_least_one_feature(self) -> "LayersetFeatureCollection":
        if not self.features:
            raise ValueError(
                "Layerset must contain at least one Feature. "
                "An empty FeatureCollection cannot be rasterized."
            )
        return self


# --- Request Body ------------------------------------------------------------


class CreateLayersetRequestBody(LayersetFeatureCollection):
    """Request body for uploading a flat GeoJSON layerset.

    The body **is** the GeoJSON FeatureCollection (matching ``POST /domains``,
    whose body is a ``FeatureCollection`` directly), extended with the
    resource-metadata fields. No ``type`` discriminator: the URL
    ``/features/layerset/geojson`` already discriminates layersets from
    road/water uploads.

    ``name`` overrides the optional GeoJSON ``name`` member inherited from
    ``LayersetFeatureCollection`` — the FeatureCollection's name doubles as the
    resource name, exactly as ``CreateDomainRequestBody`` treats it.
    """

    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=50)
