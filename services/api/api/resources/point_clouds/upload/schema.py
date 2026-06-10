"""
api/v2/resources/point_clouds/upload/schema.py

Schema models for creating a point cloud from a direct file upload.
"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from api.resources.point_clouds.schema import PointCloud, PointCloudType


class PointCloudUploadFormat(StrEnum):
    """File format of the point cloud being uploaded."""

    las = "las"
    laz = "laz"
    copc = "copc"


class CreatePointCloudUploadRequest(BaseModel):
    """Request body for creating a point cloud from a direct file upload."""

    type: PointCloudType = Field(
        ...,
        description=(
            "How the cloud was acquired: `als` for airborne (aircraft or drone) "
            "or `tls` for terrestrial (tripod) scans."
        ),
    )
    format: PointCloudUploadFormat = Field(
        ...,
        description="Format of the file you will upload: `las`, `laz`, or `copc`.",
    )
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


class PointCloudUploadSpec(BaseModel):
    """Where and how to upload the source file.

    PUT the file to `url` with a `Content-Type` header equal to `content_type`.
    The upload must complete before `expires_at` and must not exceed
    `max_size_bytes`.
    """

    method: Literal["PUT"] = "PUT"
    url: str = Field(..., description="Signed URL to upload the source file to.")
    content_type: str = Field(
        ...,
        description="Value the `Content-Type` header must use when uploading.",
    )
    expires_at: datetime = Field(..., description="When the signed URL expires.")
    max_size_bytes: int = Field(
        ...,
        description="Maximum allowed size of the uploaded file, in bytes.",
    )


class PointCloudUploadCreatedResponse(BaseModel):
    """Response returned when a point cloud upload is created."""

    point_cloud: PointCloud = Field(
        ...,
        description="The created point cloud, with `status` set to `pending`.",
    )
    upload: PointCloudUploadSpec = Field(
        ...,
        description="Where and how to upload the source file.",
    )
