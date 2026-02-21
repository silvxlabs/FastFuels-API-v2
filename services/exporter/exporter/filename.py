"""Filename sanitization for export files."""

import re


def sanitize_filename(name: str, extension: str) -> str:
    """Sanitize a user-provided name into a safe filename.

    Args:
        name: Raw export name from the user.
        extension: File extension including the dot (e.g. ".tif").

    Returns:
        A sanitized filename with the given extension.
    """
    result = name.strip()
    result = re.sub(r"[^a-zA-Z0-9\-]", "_", result)
    result = re.sub(r"_+", "_", result)
    result = result.strip("_")
    result = result[:200]
    if not result:
        result = "export"
    return result + extension
