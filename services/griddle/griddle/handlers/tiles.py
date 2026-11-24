"""
Shared TileMetadata type for tile-based grid handlers.
"""

from typing import NotRequired, TypedDict


class TileMetadata(TypedDict):
    """Metadata about fetched tiles, written back to the source in Firestore."""

    tiles: list[str]
    tile_source: str | None
    tile_count: int
    native_crs: str | None
    acquisition_dates: list[str] | None
    # NAIP-CHM only; parallel to ``tiles``.
    tile_years: NotRequired[list[int]]
    tile_source_gsds: NotRequired[list[float]]
    tile_models: NotRequired[list[str]]
