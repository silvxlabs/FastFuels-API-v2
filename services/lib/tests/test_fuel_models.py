"""Tests for lib.fuel_models FBFM40 label <-> code resolution."""

from __future__ import annotations

import pytest

from lib.fuel_models import (
    FBFM40_CODE_TO_LABEL,
    FBFM40_LABEL_TO_CODE,
    UnknownFuelModelError,
    resolve_fuel_model_value,
)


class TestFbfm40Tables:
    def test_expected_size_and_known_pairs(self):
        # 45 Scott-Burgan 40 codes (5 NB + 9 GR + 4 GS + 9 SH + 5 TU + 9 TL + 4 SB).
        assert len(FBFM40_LABEL_TO_CODE) == 45
        assert FBFM40_LABEL_TO_CODE["GR1"] == 101
        assert FBFM40_LABEL_TO_CODE["GR9"] == 109
        assert FBFM40_LABEL_TO_CODE["NB1"] == 91
        assert FBFM40_LABEL_TO_CODE["NB9"] == 99
        assert FBFM40_LABEL_TO_CODE["SB4"] == 204

    def test_round_trip(self):
        for code, label in FBFM40_CODE_TO_LABEL.items():
            assert FBFM40_LABEL_TO_CODE[label] == code


class TestResolveFuelModelValue:
    def test_resolves_label_to_code(self):
        assert resolve_fuel_model_value("GR1") == 101

    def test_is_case_and_whitespace_insensitive(self):
        assert resolve_fuel_model_value(" gr2 ") == 102

    def test_passes_numbers_through(self):
        assert resolve_fuel_model_value(101) == 101
        assert resolve_fuel_model_value(0.5) == 0.5

    def test_resolves_list_with_mixed_labels_and_codes(self):
        assert resolve_fuel_model_value(["GR1", 102, "SH3"]) == [101, 102, 143]

    def test_unknown_label_raises(self):
        with pytest.raises(UnknownFuelModelError):
            resolve_fuel_model_value("GR99")

    def test_unknown_label_in_list_raises(self):
        with pytest.raises(UnknownFuelModelError):
            resolve_fuel_model_value(["GR1", "NOPE"])
