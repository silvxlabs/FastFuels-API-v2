"""Tests for lib.band_quantities."""

import pytest

from lib.band_quantities import BAND_QUANTITIES, band_mismatch, check_band_key
from lib.units import validate_unit


@pytest.mark.parametrize(
    "key",
    ["fbfm", "fuel_load.1hr", "fuel_load.mass_1_2m", "fuel_moisture.live.shrub"],
)
def test_accepts_known_quantity_with_any_label(key):
    check_band_key(key)


@pytest.mark.parametrize(
    "key", ["moisture", "custom.x", "fuel_load.", "fuel_load.Shrub", "Fuel_load.1hr"]
)
def test_rejects_unknown_quantity_or_malformed_label(key):
    with pytest.raises(ValueError):
        check_band_key(key)


def test_mismatch():
    assert band_mismatch("fuel_load.1hr", "continuous", "kg/m**2") is None
    assert band_mismatch("fbfm", "categorical", "") is None
    assert band_mismatch("fbfm", "continuous", "m") is not None
    assert band_mismatch("chm", "continuous", "ft") is not None


def test_units_are_canonical():
    for quantity in BAND_QUANTITIES.values():
        validate_unit(quantity.unit)
