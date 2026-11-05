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
