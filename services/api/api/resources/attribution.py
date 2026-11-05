"""
api/resources/attribution.py

Shared license and citation model for resources built from third-party data,
and the helpers that pass parents' attributions down to derived resources.
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


def merge_attributions(*attribution_lists: list[dict] | None) -> list[dict]:
    """Merge parents' attribution lists into one list for a child resource.

    Entries are de-duplicated by ``(dataset, license_name)``; for a duplicate
    the entry with the earliest ``accessed_on`` is kept. The result is sorted
    by dataset, then license name. ``None`` lists are skipped.
    """
    merged: dict[tuple[str, str], dict] = {}
    for attributions in attribution_lists:
        for entry in attributions or []:
            key = (entry["dataset"], entry["license_name"])
            kept = merged.get(key)
            if kept is None or entry["accessed_on"] < kept["accessed_on"]:
                merged[key] = dict(entry)
    return [merged[key] for key in sorted(merged)]


def inherit_attributions(*parents: dict | None) -> list[dict]:
    """Merge the ``attributions`` of parent resource documents.

    ``parents`` are Firestore documents (features, grids, inventories) a child
    resource is built from. ``None`` parents (an optional parent not given) are
    skipped; a missing or ``None`` ``attributions`` key counts as empty.
    """
    return merge_attributions(
        *(parent.get("attributions") for parent in parents if parent is not None)
    )
