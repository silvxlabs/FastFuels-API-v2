"""Scott-Burgan 40 (FBFM40) fuel-model label <-> code mapping.

Grid ``fbfm`` bands store the integer Scott-Burgan code (e.g. ``101``), but
users prefer the human-readable label (e.g. ``"GR1"``). This module resolves
labels to codes at the API write boundary so stored conditions/actions are
always integer codes and the processing services stay integer-only.

Pure-Python (no GDAL), so the API may import it.
"""

from __future__ import annotations

# Code groups per the Scott-Burgan 40 classification, as ``prefix -> (offset,
# codes)``. The label for a code is ``f"{prefix}{code - offset}"`` (e.g. 101 ->
# "GR1", 91 -> "NB1"). Codes match the griddle SB40 lookup table
# (services/griddle/griddle/data/sb40_fbfm40.csv).
_FBFM40_GROUPS: dict[str, tuple[int, list[int]]] = {
    "NB": (90, [91, 92, 93, 98, 99]),
    "GR": (100, [101, 102, 103, 104, 105, 106, 107, 108, 109]),
    "GS": (120, [121, 122, 123, 124]),
    "SH": (140, [141, 142, 143, 144, 145, 146, 147, 148, 149]),
    "TU": (160, [161, 162, 163, 164, 165]),
    "TL": (180, [181, 182, 183, 184, 185, 186, 187, 188, 189]),
    "SB": (200, [201, 202, 203, 204]),
}

FBFM40_CODE_TO_LABEL: dict[int, str] = {
    code: f"{prefix}{code - offset}"
    for prefix, (offset, codes) in _FBFM40_GROUPS.items()
    for code in codes
}
FBFM40_LABEL_TO_CODE: dict[str, int] = {
    label: code for code, label in FBFM40_CODE_TO_LABEL.items()
}


class UnknownFuelModelError(ValueError):
    """Raised when a string is not a recognized FBFM40 fuel-model label."""

    def __init__(self, label: str):
        self.label = label
        super().__init__(
            f"{label!r} is not a recognized Scott-Burgan FBFM40 fuel model. "
            "Use a label such as 'GR1' or the numeric code (e.g. 101)."
        )


def resolve_fuel_model_value(value):
    """Resolve FBFM40 string labels to integer codes; pass numbers through.

    Accepts a scalar (``int``/``float``/``str``) or a ``list`` thereof and
    returns the same shape with any string label replaced by its integer code.
    Matching is case-insensitive and ignores surrounding whitespace. Raises
    :class:`UnknownFuelModelError` for an unrecognized label.
    """
    if isinstance(value, list):
        return [resolve_fuel_model_value(item) for item in value]
    if isinstance(value, str):
        code = FBFM40_LABEL_TO_CODE.get(value.strip().upper())
        if code is None:
            raise UnknownFuelModelError(value)
        return code
    return value
