"""Band keys for layerset-rasterized grids.

``fastfuels_core.rasterize_layerset`` emits five properties per fuel type
(``loading``, ``height``, ``live_fuel_moisture``, ``dead_fuel_moisture``,
``heat_of_combustion``). Each is stored as its own band keyed
``<quantity>.<fuel type token>``, quantity first like every other band key
(e.g. ``fuel_load.shrub``), so a layerset band counts as the quantity it holds.

Used by griddle, which writes the bands. Pure Python.
"""

import re

# Rasterizer property -> band key prefix, unit, and display metadata. Units
# match the canonical keys of the same quantity (fuel_load.* kg/m**2,
# fuel_depth.* m, fuel_moisture.* %).
LAYERSET_PROPERTIES: dict[str, dict] = {
    "loading": {
        "prefix": "fuel_load",
        "unit": "kg/m**2",
        "name": "Fuel Load",
        "description": "Oven-dry fuel mass per unit area.",
    },
    "height": {
        "prefix": "fuel_depth",
        "unit": "m",
        "name": "Fuel Depth",
        "description": "Height of the fuel bed above the surface.",
    },
    "live_fuel_moisture": {
        "prefix": "fuel_moisture.live",
        "unit": "%",
        "name": "Live Fuel Moisture",
        "description": "Moisture content of live fuels (% of oven-dry weight).",
    },
    "dead_fuel_moisture": {
        "prefix": "fuel_moisture.dead",
        "unit": "%",
        "name": "Dead Fuel Moisture",
        "description": "Moisture content of dead fuels (% of oven-dry weight).",
    },
    "heat_of_combustion": {
        "prefix": "heat_content",
        "unit": "kJ/kg",
        "name": "Heat Content",
        "description": "Energy released per unit mass of fuel consumed.",
    },
}


def fuel_type_token(fuel_type: str) -> str:
    """A key-safe token for a layerset ``fuel_type``.

    Lowercased; every run of characters outside ``[a-z0-9_]`` (spaces, dots,
    hyphens, non-ASCII) becomes one underscore; leading/trailing underscores
    are stripped. ``"Shrub Layer"`` -> ``"shrub_layer"``, ``"tl.3"`` ->
    ``"tl_3"``.

    Raises:
        ValueError: If nothing key-safe remains.
    """
    token = re.sub(r"[^a-z0-9_]+", "_", str(fuel_type).lower()).strip("_")
    if not token:
        raise ValueError(f"Fuel type {fuel_type!r} has no key-safe characters.")
    return token


def layerset_band_key(property_name: str, fuel_type: str) -> str:
    """The band key for one rasterizer property of one fuel type."""
    prefix = LAYERSET_PROPERTIES[property_name]["prefix"]
    return f"{prefix}.{fuel_type_token(fuel_type)}"


def split_legacy_layerset_key(key: str) -> tuple[str, str]:
    """Split a pre-#672 layerset key ``<fuel_type>.<property>`` into its
    Dataset variable (the fuel type) and ``band`` coord (the property).

    Grids rasterized before #672 store one variable per fuel type with a
    ``band`` coord. Properties never contain a dot but fuel types could, so
    the split is on the last dot.
    """
    var, _, band = key.rpartition(".")
    return var, band
