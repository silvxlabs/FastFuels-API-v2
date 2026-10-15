"""Every band a producer emits must follow lib.band_quantities.

Producer bands are found by scanning the schema modules under
api.resources.grids for module-level Band objects and band-definition dicts, so a new producer is checked without
being registered anywhere.
"""

import importlib
import pkgutil

import api.resources.grids as grids_pkg
import pytest
from api.resources.grids.schema import Band

from lib.band_quantities import band_mismatch, check_band_key


def _band_defs(value) -> list[tuple[str, str, str | None]]:
    if isinstance(value, Band):
        return [(value.key, str(value.type), value.unit)]
    if isinstance(value, dict) and {"key", "type"} <= value.keys():
        return [(value["key"], str(value["type"]), value.get("unit"))]
    return []


def _producer_bands() -> dict[str, tuple[str, str, str | None]]:
    found = {}
    for info in pkgutil.walk_packages(grids_pkg.__path__, f"{grids_pkg.__name__}."):
        if not info.name.endswith(".schema"):
            continue
        module = importlib.import_module(info.name)
        for name, value in vars(module).items():
            defs = _band_defs(value)
            if isinstance(value, dict):
                for key, item in value.items():
                    defs += _band_defs(item)
                    # Lookup metadata tables are keyed by the band-key enum.
                    if name.endswith("_LOOKUP_BAND_METADATA"):
                        defs.append((key.value, str(item["type"]), item.get("unit")))
            for key, type_, unit in defs:
                found[f"{module.__name__}.{name}:{key}"] = (key, type_, unit)
    return found


PRODUCER_BANDS = _producer_bands()


def test_scan_finds_producers():
    keys = {key for key, _, _ in PRODUCER_BANDS.values()}
    assert {"fbfm", "chm", "fuel_load.1hr", "cfl", "fuel_moisture.1hr"} <= keys


@pytest.mark.parametrize("where", sorted(PRODUCER_BANDS))
def test_producer_band_matches_its_quantity(where):
    key, type_, unit = PRODUCER_BANDS[where]
    check_band_key(key)
    assert band_mismatch(key, type_, unit) is None
