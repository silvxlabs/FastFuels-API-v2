"""
lib/attributions.py

License and citation values for third-party datasets. Pure Python so both the
API (no GDAL) and the workers can import it.
"""

OPENSTREETMAP = "OpenStreetMap"

DATASET_ATTRIBUTIONS = {
    OPENSTREETMAP: {
        "dataset": OPENSTREETMAP,
        "license_name": "ODbL-1.0",
        "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
        "citation": "© OpenStreetMap contributors",
        "access_url": "https://www.openstreetmap.org/copyright",
    },
}


def make_attribution(dataset: str, accessed_on: str) -> dict:
    """Return the attribution dict for ``dataset`` accessed on ``accessed_on``.

    ``accessed_on`` is an ISO date string (``YYYY-MM-DD``). Raises ``KeyError``
    for an unknown dataset.
    """
    return {**DATASET_ATTRIBUTIONS[dataset], "accessed_on": accessed_on}
