"""Schema models for the netCDF grid upload endpoint."""

from pydantic import BaseModel, Field


class CreateNetcdfUploadRequest(BaseModel):
    name: str = Field("", max_length=255)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list)
    num_buffer_cells: int = Field(
        0,
        ge=0,
        description=(
            "Number of extra native-resolution cells to keep around the domain "
            "extent in the stored grid. The uploaded netCDF must cover the "
            "domain bbox expanded by num_buffer_cells * native_pixel_size on "
            "each side; pixels beyond that expanded extent are clipped away."
        ),
    )
