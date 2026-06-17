"""
api/v2/resources/point_clouds/upload/schema.py

Schema models for creating a point cloud from a direct file upload.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from api.resources.point_clouds.schema import PointCloud, PointCloudType


class CreatePointCloudUploadRequest(BaseModel):
    """Request body for creating a point cloud from a direct file upload."""

    type: PointCloudType = Field(
        ...,
        description=(
            "How the cloud was acquired: `als` for airborne (aircraft or drone) "
            "or `tls` for terrestrial (tripod) scans."
        ),
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

    PUT the file to `url`, sending every header in `headers` exactly as given.
    The upload must complete before `expires_at` and must not exceed
    `max_size_bytes`.
    """

    method: Literal["PUT"] = "PUT"
    url: str = Field(..., description="Signed URL to upload the source file to.")
    headers: dict[str, str] = Field(
        ...,
        description=(
            "HTTP headers that must be sent with the PUT request, exactly as "
            "given. The signed URL commits to these headers; the upload is "
            "rejected if any is missing or altered."
        ),
    )
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
