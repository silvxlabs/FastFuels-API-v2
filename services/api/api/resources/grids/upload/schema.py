"""Shared schema models for grid upload endpoints (format-agnostic)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from api.resources.grids.schema import Grid


class GridUploadSpec(BaseModel):
    method: Literal["PUT"] = "PUT"
    url: str
    headers: dict[str, str]
    content_type: str
    expires_at: datetime
    max_size_bytes: int


class GridUploadCreatedResponse(BaseModel):
    grid: Grid
    upload: GridUploadSpec
