"""
Shared TileMetadata type for tile-based grid handlers.
"""

from typing import TypedDict


class TileMetadata(TypedDict):
    """Metadata about fetched tiles, written back to the source in Firestore."""

    tiles: list[str]
    tile_source: str | None
    tile_count: int
    native_crs: str | None
    acquisition_dates: list[str] | None
