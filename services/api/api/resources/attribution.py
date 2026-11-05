"""
api/resources/attribution.py

Shared license and citation model for resources built from third-party data.
"""

from pydantic import BaseModel, ConfigDict, Field


class Attribution(BaseModel):
    """License and citation for one third-party dataset used by a resource."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "dataset": "OpenStreetMap",
                    "license_name": "ODbL-1.0",
                    "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
                    "citation": "© OpenStreetMap contributors",
                    "access_url": "https://www.openstreetmap.org/copyright",
                    "accessed_on": "2026-11-05",
                }
            ]
        }
    )

    dataset: str = Field(..., description="Name of the source dataset.")
    license_name: str = Field(
        ..., description="SPDX identifier of the dataset's license."
    )
    license_url: str = Field(..., description="Link to the full license text.")
    citation: str = Field(
        ...,
        description="Attribution text to reproduce wherever the data is shown or shared.",
    )
    access_url: str = Field(
        ..., description="Where to find the dataset and its terms of use."
    )
    accessed_on: str = Field(
        ..., description="Date the data was retrieved, as YYYY-MM-DD."
    )
