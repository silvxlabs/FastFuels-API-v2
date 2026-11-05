"""
Attribution metadata for export files.

Every format renders the export document's ``attributions`` with these
helpers, so each file uses the same wording.
"""

import json

ATTRIBUTION_FILENAME = "ATTRIBUTION.txt"


def attribution_notice(attributions: list[dict]) -> str:
    """Human-readable notice listing each dataset's citation, license, source
    and access date."""
    lines = [
        "This export contains data adapted from the third-party datasets below.",
        "Reproduce each citation wherever you use or share the data.",
    ]
    for entry in attributions:
        lines += [
            "",
            entry["dataset"],
            f"  Citation: {entry['citation']}",
            f"  License: {entry['license_name']} ({entry['license_url']})",
            f"  Source: {entry['access_url']}",
            f"  Accessed: {entry['accessed_on']}",
        ]
    return "\n".join(lines)


def attribution_metadata(attributions: list[dict]) -> dict[str, str]:
    """String key-value metadata for formats with an attribute channel.

    ``attributions`` holds the JSON list and ``license`` the notice from
    `attribution_notice`. Empty when there are no attributions.
    """
    if not attributions:
        return {}
    return {
        "attributions": json.dumps(attributions),
        "license": attribution_notice(attributions),
    }
