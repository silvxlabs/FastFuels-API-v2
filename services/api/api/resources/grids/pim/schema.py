"""
api/v2/resources/grids/pim/schema.py

Schema models for the TreeMap grid product.

TreeMap is a 30m raster where each pixel contains a plot ID (TM_ID) that
maps to FIA tree records. Optionally includes PLT_CN (FIA plot condition
number) as a second band via a lookup from the tree table.
"""

from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator

from api.resources.grids.providers.pim import PimSource
from api.resources.grids.schema import (
    Band,
    BandType,
    CreateSourceGridRequestBase,
    validate_no_duplicates,
)


class TreeMapVersion(StrEnum):
    """Available TreeMap data versions."""

    v2014 = "2014"
    v2016 = "2016"
    v2020 = "2020"
    v2022 = "2022"


class TreeMapBand(StrEnum):
    """Available bands for TreeMap data."""

    tm_id = "tm_id"
    plt_cn = "plt_cn"


TREEMAP_BAND_DEFS = {
    TreeMapBand.tm_id: {
        "key": "tm_id",
        "type": BandType.categorical,
        "unit": None,
    },
    TreeMapBand.plt_cn: {
        "key": "plt_cn",
        "type": BandType.categorical,
        "unit": None,
    },
}


class TreeMapSource(PimSource):
    """Source for TreeMap plot imputation data.

    Returns categorical plot ID rasters at 30m resolution. Each pixel
    maps to FIA tree records via TM_ID or PLT_CN.
    """

    product: Literal["treemap"] = "treemap"
    version: TreeMapVersion
    bands: list[TreeMapBand]
    description: Literal["TreeMap plot imputation raster (FIA plot IDs at 30m)"] = (
        "TreeMap plot imputation raster (FIA plot IDs at 30m)"
    )


class CreateTreeMapRequest(CreateSourceGridRequestBase):
    """Request to create a grid from TreeMap.

    Returns a grid with one or two categorical bands:
    - tm_id: TreeMap raster pixel values (always available)
    - plt_cn: FIA plot condition number (optional, derived from tree table)
    """

    version: TreeMapVersion = TreeMapVersion.v2022
    bands: list[TreeMapBand] = Field(
        default=[TreeMapBand.tm_id],
        min_length=1,
    )

    @field_validator("bands")
    @classmethod
    def no_duplicate_bands(cls, v: list[TreeMapBand]) -> list[TreeMapBand]:
        return validate_no_duplicates(v)


def build_treemap_bands(requested: list[TreeMapBand]) -> list[Band]:
    """Build Band objects for requested TreeMap bands with correct indices."""
    return [
        Band(index=i, **TREEMAP_BAND_DEFS[band]) for i, band in enumerate(requested)
    ]
