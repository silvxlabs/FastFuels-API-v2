"""
Unit tests for api/resources/grids/solar/irradiance/leaflux/schema.py.

Pure schema tests with no external dependencies.
"""

import pytest
from api.resources.grids.schema import BandType
from api.resources.grids.solar.irradiance.leaflux.examples import (
    ALL_LEAFLUX_IRRADIANCE_EXAMPLE_VALUES,
)
from api.resources.grids.solar.irradiance.leaflux.schema import (
    LEAFLUX_BAND_DEFS,
    CreateLeafluxIrradianceRequest,
    IrradianceLeafluxSource,
    LeafluxBand,
    build_leaflux_bands,
)
from pydantic import ValidationError

from lib.units import validate_unit

DATE_TIME = "2025-07-01T19:00:00Z"
CANOPY = "irradiance.canopy.relative"
SURFACE = "irradiance.surface.relative"


class TestLeafluxBand:
    """LeafluxBand enum values match the spec."""

    def test_all_expected_bands_present(self):
        assert {b.value for b in LeafluxBand} == {CANOPY, SURFACE}

    def test_band_defs_cover_every_band(self):
        assert set(LEAFLUX_BAND_DEFS.keys()) == set(LeafluxBand)

    @pytest.mark.parametrize(
        "band,expected_type,expected_unit",
        [
            (LeafluxBand.irradiance_canopy_relative, BandType.continuous, None),
            (LeafluxBand.irradiance_surface_relative, BandType.continuous, None),
        ],
    )
    def test_band_definitions(self, band, expected_type, expected_unit):
        definition = LEAFLUX_BAND_DEFS[band]
        assert definition["key"] == band.value
        assert definition["type"] == expected_type
        assert definition["unit"] == expected_unit

    def test_all_units_are_canonical(self):
        for entry in LEAFLUX_BAND_DEFS.values():
            validate_unit(entry.get("unit"))


class TestBuildLeafluxBands:
    """build_leaflux_bands assigns indices in request order."""

    def test_single_band(self):
        bands = build_leaflux_bands([LeafluxBand.irradiance_surface_relative])
        assert len(bands) == 1
        assert bands[0].key == SURFACE
        assert bands[0].name
        assert bands[0].description
        assert bands[0].type == BandType.continuous
        assert bands[0].unit is None
        assert bands[0].index == 0

    def test_all_bands_have_name_and_description(self):
        bands = build_leaflux_bands(list(LeafluxBand))
        for band in bands:
            assert band.name
            assert band.description

    def test_indices_match_request_order(self):
        requested = [
            LeafluxBand.irradiance_canopy_relative,
            LeafluxBand.irradiance_surface_relative,
        ]
        bands = build_leaflux_bands(requested)
        assert [b.index for b in bands] == [0, 1]
        assert [b.key for b in bands] == [CANOPY, SURFACE]

    def test_order_is_preserved_when_reversed(self):
        bands = build_leaflux_bands(
            [
                LeafluxBand.irradiance_surface_relative,
                LeafluxBand.irradiance_canopy_relative,
            ]
        )
        assert [b.key for b in bands] == [SURFACE, CANOPY]

    def test_all_bands(self):
        all_bands = list(LeafluxBand)
        bands = build_leaflux_bands(all_bands)
        assert len(bands) == len(all_bands)
        assert [b.index for b in bands] == list(range(len(all_bands)))


class TestCreateLeafluxIrradianceRequest:
    """Validation rules for the request body."""

    def _minimal(self, **overrides) -> dict:
        body = {"source_grid_id": "abc123", "date_time": DATE_TIME}
        body.update(overrides)
        return body

    def test_minimal_valid_request(self):
        req = CreateLeafluxIrradianceRequest(**self._minimal())
        assert req.source_grid_id == "abc123"
        assert req.source_terrain_grid_id is None
        assert req.bands == [LeafluxBand.irradiance_surface_relative]
        assert req.extinction_coefficient == 0.5
        assert req.name == ""
        assert req.description == ""
        assert req.tags == []

    def test_source_grid_id_is_required(self):
        with pytest.raises(ValidationError):
            CreateLeafluxIrradianceRequest(date_time=DATE_TIME)

    def test_date_time_is_required(self):
        with pytest.raises(ValidationError):
            CreateLeafluxIrradianceRequest(source_grid_id="abc")

    def test_date_time_parses_iso_z(self):
        req = CreateLeafluxIrradianceRequest(**self._minimal())
        assert req.date_time.year == 2025
        assert req.date_time.hour == 19

    def test_bands_defaults_to_surface(self):
        req = CreateLeafluxIrradianceRequest(source_grid_id="abc", date_time=DATE_TIME)
        assert req.bands == [LeafluxBand.irradiance_surface_relative]

    def test_bands_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            CreateLeafluxIrradianceRequest(**self._minimal(bands=[]))

    def test_invalid_band_rejected(self):
        with pytest.raises(ValidationError):
            CreateLeafluxIrradianceRequest(**self._minimal(bands=["not_a_band"]))

    def test_both_bands_accepted(self):
        req = CreateLeafluxIrradianceRequest(**self._minimal(bands=[CANOPY, SURFACE]))
        assert req.bands == [
            LeafluxBand.irradiance_canopy_relative,
            LeafluxBand.irradiance_surface_relative,
        ]

    def test_extinction_defaults_to_half(self):
        req = CreateLeafluxIrradianceRequest(**self._minimal())
        assert req.extinction_coefficient == 0.5

    @pytest.mark.parametrize("extinction", [0.0, -0.5])
    def test_non_positive_extinction_rejected(self, extinction):
        with pytest.raises(ValidationError):
            CreateLeafluxIrradianceRequest(
                **self._minimal(extinction_coefficient=extinction)
            )

    def test_terrain_grid_id_optional_and_accepted(self):
        req = CreateLeafluxIrradianceRequest(
            **self._minimal(source_terrain_grid_id="terrain-1")
        )
        assert req.source_terrain_grid_id == "terrain-1"

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            CreateLeafluxIrradianceRequest(**self._minimal(resolution=10))

    def test_metadata_round_trips(self):
        req = CreateLeafluxIrradianceRequest(
            **self._minimal(
                name="Named grid",
                description="A grid with metadata",
                tags=["solar", "test"],
            )
        )
        assert req.name == "Named grid"
        assert req.description == "A grid with metadata"
        assert req.tags == ["solar", "test"]

    def test_name_too_long_rejected(self):
        with pytest.raises(ValidationError):
            CreateLeafluxIrradianceRequest(**self._minimal(name="x" * 256))

    def test_full_request(self):
        req = CreateLeafluxIrradianceRequest(
            name="Midday irradiance",
            description="Canopy and surface.",
            tags=["solar"],
            source_grid_id="grid-1",
            source_terrain_grid_id="terrain-1",
            bands=[CANOPY, SURFACE],
            date_time=DATE_TIME,
            extinction_coefficient=0.4,
        )
        assert req.source_terrain_grid_id == "terrain-1"
        assert req.bands == [
            LeafluxBand.irradiance_canopy_relative,
            LeafluxBand.irradiance_surface_relative,
        ]
        assert req.extinction_coefficient == 0.4

    @pytest.mark.parametrize(
        "example_name,example_value", ALL_LEAFLUX_IRRADIANCE_EXAMPLE_VALUES
    )
    def test_documented_examples_are_schema_valid(self, example_name, example_value):
        req = CreateLeafluxIrradianceRequest(**example_value)
        assert req.source_grid_id, example_name


class TestIrradianceLeafluxSource:
    """The persisted-source schema stored on the Grid document."""

    def _minimal(self, **overrides) -> dict:
        body = {
            "source_grid_id": "grid-1",
            "bands": [LeafluxBand.irradiance_surface_relative],
            "date_time": DATE_TIME,
            "extinction_coefficient": 0.5,
        }
        body.update(overrides)
        return body

    def test_discriminators_fixed(self):
        source = IrradianceLeafluxSource(**self._minimal())
        assert source.operation == "irradiance"
        assert source.input == "grid"
        assert source.entity == "solar"

    def test_source_grid_checksum_defaults_to_none(self):
        source = IrradianceLeafluxSource(**self._minimal())
        assert source.source_grid_checksum is None

    def test_source_grid_checksum_round_trips(self):
        source = IrradianceLeafluxSource(**self._minimal(source_grid_checksum="sum123"))
        assert source.source_grid_checksum == "sum123"
        assert source.model_dump(mode="json")["source_grid_checksum"] == "sum123"

    @pytest.mark.parametrize(
        "field", ["source_grid_id", "bands", "date_time", "extinction_coefficient"]
    )
    def test_required_fields(self, field):
        """Every resolved model choice is persisted for reproducibility."""
        body = self._minimal()
        del body[field]
        with pytest.raises(ValidationError):
            IrradianceLeafluxSource(**body)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            IrradianceLeafluxSource(**self._minimal(resolution=10))

    def test_none_fields_dropped_on_dump(self):
        source = IrradianceLeafluxSource(**self._minimal())
        data = source.model_dump(mode="json", exclude_none=True)
        assert "source_terrain_grid_id" not in data
        assert "source_grid_checksum" not in data

    def test_model_dump_includes_resolved_fields(self):
        source = IrradianceLeafluxSource(
            **self._minimal(
                source_terrain_grid_id="terrain-1",
                bands=[
                    LeafluxBand.irradiance_canopy_relative,
                    LeafluxBand.irradiance_surface_relative,
                ],
                extinction_coefficient=0.4,
            )
        )
        data = source.model_dump(mode="json", exclude_none=True)
        assert data["operation"] == "irradiance"
        assert data["input"] == "grid"
        assert data["entity"] == "solar"
        assert data["source_grid_id"] == "grid-1"
        assert data["source_terrain_grid_id"] == "terrain-1"
        assert data["bands"] == [CANOPY, SURFACE]
        assert data["extinction_coefficient"] == 0.4
        assert data["date_time"].startswith("2025-07-01")
