"""
api/v2/resources/grids/rasterize/layerset/schema.py

Schema models for the layerset rasterization grid product.

A layerset is a flat GeoJSON FeatureCollection of fuelbed polygons
uploaded via ``POST /domains/{id}/features/layerset``. This product
rasterizes that GeoJSON into a Zarr grid aligned to a domain or target
grid. The rasterization math lives in
``fastfuels_core.layersets.rasterize_layerset``.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from api.resources.grids.schema import (
    Band,
    CreateSourceGridRequestBase,
)


class OverlapMethod(StrEnum):
    """Per-cell reduction when multiple polygons of the same ``fuel_type``
    overlap a cell. Applies to ``height`` and the optional bands only —
    ``loading`` is always summed by ``fastfuels_core.rasterize_layerset``
    regardless of this setting.
    """

    mean = "mean"
    max = "max"
    min = "min"


class LayersetSource(BaseModel):
    """Source record for a rasterized layerset grid.

    Persisted to Firestore in the Grid document's ``source`` field. The
    ``name`` field is the dispatch discriminator read by
    ``services/griddle/griddle/dispatch.py``.
    """

    name: Literal["layerset"] = "layerset"
    product: Literal["layerset"] = "layerset"
    layerset_id: str
    overlap_method: OverlapMethod
    extent_buffer_cells: int
    alignment: dict
    description: Literal["Rasterized fuelbed layerset"] = "Rasterized fuelbed layerset"


class CreateLayersetRasterizeRequest(CreateSourceGridRequestBase):
    """Request to create a grid by rasterizing a previously-uploaded layerset.

    The referenced layerset must be an existing Feature owned by the caller,
    uploaded via ``POST /domains/{id}/features/layerset``. The worker fetches
    the GeoJSON from GCS at job time; a fresh upload produces a new
    ``feature_id``, so the reference is effectively immutable.
    """

    layerset_id: str = Field(
        ...,
        description="Feature ID of an existing layerset uploaded for this domain.",
    )
    overlap_method: OverlapMethod = Field(
        default=OverlapMethod.mean,
        description="Per-cell reduction when polygons overlap a cell.",
    )
    extent_buffer_cells: int = Field(
        0,
        ge=0,
        le=10,
        description=(
            "Buffer in result-grid cells around the domain extent. Cells "
            "inside the buffered extent that fall outside polygon coverage "
            "are populated with the rasterizer's fill value. Default 0 adds "
            "no buffer. Maximum: 10 cells."
        ),
    )


def build_layerset_bands() -> list[Band]:
    """Build the band list for a layerset-rasterized grid.

    Returns an empty list. The real band layout depends on the unique
    ``fuel_type`` values in the uploaded layerset GeoJSON — the API has no
    visibility into that at create time. The griddle worker reads the
    GeoJSON, runs ``fastfuels_core.rasterize_layerset``, and writes the
    full band list (one entry per ``fuel_type × {loading, height,
    live_fuel_moisture, dead_fuel_moisture, heat_of_combustion}``) back
    to the Firestore grid doc once rasterization completes.
    """
    return []
