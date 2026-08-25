"""Tests for lib.landfire,config version registry and fuel-model constants."""

from __future__ import annotations

import pytest

from lib.landfire.config import (
    LANDFIRE_VERSIONS,
    NB_CODE_MAP,
    UnknownLandfireVersionError,
    validate_landfire_version,
)


class TestLandfireVersionsTable:
    def test_expected_products(self):
        assert set(LANDFIRE_VERSIONS) == {"fbfm13", "fbfm40", "fccs"}

    def test_default_is_always_in_available(self):
        for product, info in LANDFIRE_VERSIONS.items():
            assert info["default"] in info["available"], product

    def test_fbfm13_versions(self):
        assert LANDFIRE_VERSIONS["fbfm13"]["available"] == ["2023", "2024"]
        assert LANDFIRE_VERSIONS["fbfm13"]["lfps_available"] == ["2025"]
        assert LANDFIRE_VERSIONS["fbfm13"]["default"] == "2024"

    def test_fbfm40_versions(self):
        assert LANDFIRE_VERSIONS["fbfm40"]["available"] == [
            "2019",
            "2020",
            "2022",
            "2023",
            "2024",
        ]
        assert LANDFIRE_VERSIONS["fbfm40"]["lfps_available"] == ["2025"]
        assert LANDFIRE_VERSIONS["fbfm40"]["default"] == "2024"

    def test_fccs_versions(self):
        assert LANDFIRE_VERSIONS["fccs"]["available"] == ["2023"]
        assert LANDFIRE_VERSIONS["fccs"]["lfps_available"] == ["2025"]
        assert LANDFIRE_VERSIONS["fccs"]["default"] == "2023"


class TestValidateLandfireVersion:
    def test_valid_version_returns_unchanged(self):
        assert validate_landfire_version("fbfm13", "2023") == "2023"

    def test_unknown_version_raises(self):
        with pytest.raises(UnknownLandfireVersionError):
            validate_landfire_version("fbfm13", "1999")

    def test_unknown_version_for_fccs_raises(self):
        """fccs only ever had one version -- confirm any other year is rejected."""
        with pytest.raises(UnknownLandfireVersionError):
            validate_landfire_version("fccs", "2024")

    def test_error_message_lists_available_versions(self):
        with pytest.raises(UnknownLandfireVersionError) as exc_info:
            validate_landfire_version("fccs", "2024")
        message = str(exc_info.value)
        assert "2024" in message
        assert "2023" in message

    def test_error_carries_product_and_version(self):
        with pytest.raises(UnknownLandfireVersionError) as exc_info:
            validate_landfire_version("fbfm40", "2021")
        assert exc_info.value.product == "fbfm40"
        assert exc_info.value.version == "2021"


class TestNbCodeMap:
    def test_all_codes_present(self):
        assert set(NB_CODE_MAP) == {"NB1", "NB2", "NB3", "NB8", "NB9"}

    def test_values(self):
        assert NB_CODE_MAP == {
            "NB1": 91,
            "NB2": 92,
            "NB3": 93,
            "NB8": 98,
            "NB9": 99,
        }
