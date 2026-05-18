"""Tests for lib.units canonical-form enforcement."""

from __future__ import annotations

import pytest

from lib.units import canonicalize_unit, validate_unit


class TestValidateUnit:
    """validate_unit accepts canonical strings, rejects everything else."""

    @pytest.mark.parametrize(
        "s",
        [
            None,
            "kg/m**3",
            "kg/m**2",
            "1/m",
            "m",
            "cm",
            "kg",
            "%",
            "deg",
            "in",
            "ft",
            "kJ/kg",
        ],
    )
    def test_canonical_passes(self, s: str | None) -> None:
        validate_unit(s)

    @pytest.mark.parametrize(
        ("legacy", "expected_canonical"),
        [
            ("kg/m³", "kg/m**3"),
            ("kg/m²", "kg/m**2"),
            ("m⁻¹", "1/m"),
            ("kg/m^3", "kg/m**3"),
            ("degrees", "deg"),
        ],
    )
    def test_legacy_form_rejected(self, legacy: str, expected_canonical: str) -> None:
        with pytest.raises(ValueError):
            validate_unit(legacy)
        assert canonicalize_unit(legacy) == expected_canonical

    def test_bare_numeric_form_rejected(self) -> None:
        """kg/m3 has no canonical mapping — pint cannot parse `m3`."""
        with pytest.raises(ValueError):
            validate_unit("kg/m3")
        with pytest.raises(ValueError):
            canonicalize_unit("kg/m3")

    @pytest.mark.parametrize(
        "garbage",
        ["not_a_unit", "kg/widget", "asdf"],
    )
    def test_unrecognized_unit_rejected(self, garbage: str) -> None:
        with pytest.raises(ValueError):
            validate_unit(garbage)


class TestCanonicalizeUnit:
    """canonicalize_unit normalizes legacy strings to the canonical form."""

    @pytest.mark.parametrize(
        ("inp", "out"),
        [
            ("kg/m³", "kg/m**3"),
            ("kg/m²", "kg/m**2"),
            ("m⁻¹", "1/m"),
            ("kg/m^3", "kg/m**3"),
            ("degrees", "deg"),
            ("BTU/lb", "Btu/lb"),
            ("short_ton/acre", "ton/acre"),
            ("kg/m**3", "kg/m**3"),
            ("1/m", "1/m"),
        ],
    )
    def test_normalizes(self, inp: str, out: str) -> None:
        assert canonicalize_unit(inp) == out

    def test_round_trip(self) -> None:
        for s in ["kg/m**3", "1/m", "m", "cm", "kg", "%", "deg"]:
            assert canonicalize_unit(canonicalize_unit(s)) == s
