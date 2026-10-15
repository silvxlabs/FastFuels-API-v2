"""What a band holds, read from its key.

A band key is ``<quantity>`` or ``<quantity>.<label>``: ``fbfm``,
``fuel_load.1hr``, ``fuel_moisture.live.shrub``. The quantity fixes the band's
type and unit; the label (a timelag class, a DUET component, a layerset fuel
type) is free. So any ``fuel_load.*`` band is a fuel load in kg/m**2, whichever
producer or upload it came from.
"""

import re
from typing import NamedTuple


class BandQuantity(NamedTuple):
    type: str  # "continuous" or "categorical"
    unit: str | None


_CATEGORICAL = BandQuantity("categorical", None)

BAND_QUANTITIES: dict[str, BandQuantity] = {
    "annual_disturbance": _CATEGORICAL,
    "aspect": BandQuantity("continuous", "deg"),
    "bulk_density": BandQuantity("continuous", "kg/m**3"),
    "cbd": BandQuantity("continuous", "kg/m**3"),
    "cbh": BandQuantity("continuous", "m"),
    "cc": BandQuantity("continuous", "%"),
    "cfl": BandQuantity("continuous", "kg/m**2"),
    "chm": BandQuantity("continuous", "m"),
    "curing": BandQuantity("continuous", "%"),
    "duff_depth": BandQuantity("continuous", "m"),
    "elevation": BandQuantity("continuous", "m"),
    "fbfm": _CATEGORICAL,
    "fbfm13": _CATEGORICAL,
    "fccs": _CATEGORICAL,
    "fuel_depth": BandQuantity("continuous", "m"),
    "fuel_load": BandQuantity("continuous", "kg/m**2"),
    "fuel_moisture": BandQuantity("continuous", "%"),
    "heat_content": BandQuantity("continuous", "kJ/kg"),
    "irradiance": BandQuantity("continuous", None),
    "leaf_area_density": BandQuantity("continuous", "1/m"),
    "plt_cn": _CATEGORICAL,
    "savr": BandQuantity("continuous", "1/m"),
    "slope": BandQuantity("continuous", "deg"),
    "spcd": _CATEGORICAL,
    "tm_id": _CATEGORICAL,
    "tree_id": _CATEGORICAL,
    "volume_fraction": BandQuantity("continuous", None),
}

BAND_KEY_PATTERN = r"^(" + "|".join(BAND_QUANTITIES) + r")(\.[a-z0-9_]+)*$"


def band_quantity(key: str) -> str:
    """The quantity a band key names: the part before the first dot."""
    return key.partition(".")[0]


def check_band_key(key: str) -> BandQuantity:
    """The quantity ``key`` holds.

    Raises:
        ValueError: If the key is malformed or its quantity is unknown.
    """
    if re.fullmatch(BAND_KEY_PATTERN, key) is None:
        raise ValueError(
            f"{key!r} is not a band key. Use <quantity> or <quantity>.<label>, "
            f"with a quantity from {sorted(BAND_QUANTITIES)} and a label of "
            f"lowercase letters, digits and underscores."
        )
    return BAND_QUANTITIES[band_quantity(key)]


def band_mismatch(key: str, type_: str, unit: str | None) -> str | None:
    """Why a band with this type and unit can't carry ``key``, or None if it can."""
    expected = BAND_QUANTITIES[band_quantity(key)]
    # "" is the canonical spelling of dimensionless; it means no unit.
    if type_ != expected.type or (unit or None) != expected.unit:
        unit_text = expected.unit or "no unit"
        return (
            f"{band_quantity(key)!r} bands are {expected.type} with {unit_text}, "
            f"not {type_} with {unit or 'no unit'}"
        )
    return None
