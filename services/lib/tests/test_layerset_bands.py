"""Tests for lib.layerset_bands key construction."""

from __future__ import annotations

import pytest

from lib.layerset_bands import (
    LAYERSET_PROPERTIES,
    fuel_type_token,
    layerset_band_key,
    split_legacy_layerset_key,
)
from lib.units import validate_unit


class TestFuelTypeToken:
    @pytest.mark.parametrize(
        "fuel_type,token",
        [
            ("shrub", "shrub"),
            ("Shrub Layer", "shrub_layer"),
            ("TL.3", "tl_3"),
            ("grass--tall", "grass_tall"),
            ("  herb  ", "herb"),
            ("é-litter", "litter"),
            ("1hr", "1hr"),
            ("live_herb", "live_herb"),
        ],
    )
    def test_token(self, fuel_type, token):
        assert fuel_type_token(fuel_type) == token

    @pytest.mark.parametrize("fuel_type", ["", "---", "..."])
    def test_no_key_safe_characters_raises(self, fuel_type):
        with pytest.raises(ValueError):
            fuel_type_token(fuel_type)


class TestBandKeys:
    @pytest.mark.parametrize(
        "prop,key",
        [
            ("loading", "fuel_load.shrub_layer"),
            ("height", "fuel_depth.shrub_layer"),
            ("live_fuel_moisture", "fuel_moisture.live.shrub_layer"),
            ("dead_fuel_moisture", "fuel_moisture.dead.shrub_layer"),
            ("heat_of_combustion", "heat_content.shrub_layer"),
        ],
    )
    def test_key_is_quantity_first(self, prop, key):
        assert layerset_band_key(prop, "Shrub Layer") == key

    def test_units_are_canonical(self):
        for prop in LAYERSET_PROPERTIES.values():
            validate_unit(prop["unit"])


class TestLegacySplit:
    def test_splits_on_last_dot(self):
        assert split_legacy_layerset_key("shrub.loading") == ("shrub", "loading")
        assert split_legacy_layerset_key("tl.3.loading") == ("tl.3", "loading")
