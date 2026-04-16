"""
Unit tests for api/resources/grids/tree/schema.py
and api/resources/grids/tree/inventory/schema.py.

Pure schema tests with no external dependencies.
"""

import pytest
from api.resources.grids.schema import BandType
from api.resources.grids.tree.inventory.schema import (
    CreateTreeInventoryRequest,
    TreeInventorySource,
)
from api.resources.grids.tree.schema import (
    TREE_BAND_DEFS,
    BiomassModel,
    CrownProfileModel,
    TreeBand,
    UniformMoistureModel,
    build_tree_bands,
)
from pydantic import ValidationError


class TestTreeBand:
    """TreeBand enum values match the spec."""

    def test_all_expected_bands_present(self):
        expected = {
            "bulk_density.foliage",
            "fuel_moisture.live",
            "savr.foliage",
            "spcd",
            "tree_id",
            "volume_fraction",
        }
        assert {b.value for b in TreeBand} == expected

    def test_band_defs_cover_every_band(self):
        assert set(TREE_BAND_DEFS.keys()) == set(TreeBand)

    @pytest.mark.parametrize(
        "band,expected_type,expected_unit",
        [
            (TreeBand.bulk_density_foliage, BandType.continuous, "kg/m³"),
            (TreeBand.fuel_moisture_live, BandType.continuous, "%"),
            (TreeBand.savr_foliage, BandType.continuous, "m⁻¹"),
            (TreeBand.spcd, BandType.categorical, None),
            (TreeBand.tree_id, BandType.categorical, None),
            (TreeBand.volume_fraction, BandType.continuous, None),
        ],
    )
    def test_band_definitions(self, band, expected_type, expected_unit):
        definition = TREE_BAND_DEFS[band]
        assert definition["key"] == band.value
        assert definition["type"] == expected_type
        assert definition["unit"] == expected_unit


class TestBuildTreeBands:
    """build_tree_bands assigns indices in request order."""

    def test_single_band(self):
        bands = build_tree_bands([TreeBand.bulk_density_foliage])
        assert len(bands) == 1
        assert bands[0].key == "bulk_density.foliage"
        assert bands[0].type == BandType.continuous
        assert bands[0].unit == "kg/m³"
        assert bands[0].index == 0

    def test_indices_match_request_order(self):
        requested = [
            TreeBand.savr_foliage,
            TreeBand.spcd,
            TreeBand.bulk_density_foliage,
        ]
        bands = build_tree_bands(requested)
        assert [b.index for b in bands] == [0, 1, 2]
        assert [b.key for b in bands] == [
            "savr.foliage",
            "spcd",
            "bulk_density.foliage",
        ]

    def test_all_bands(self):
        all_bands = list(TreeBand)
        bands = build_tree_bands(all_bands)
        assert len(bands) == len(all_bands)
        assert [b.index for b in bands] == list(range(len(all_bands)))


class TestUniformMoistureModel:
    def test_default_live_value(self):
        m = UniformMoistureModel()
        assert m.method == "uniform"
        assert m.live == 100.0

    def test_override_live_value(self):
        m = UniformMoistureModel(live=97.0)
        assert m.live == 97.0

    def test_method_cannot_be_overridden(self):
        with pytest.raises(ValidationError):
            UniformMoistureModel(method="fosberg")


class TestCreateTreeInventoryRequest:
    """Validation rules for the request body."""

    def _minimal(self, **overrides) -> dict:
        body = {
            "source_inventory_id": "abc123",
            "resolution": [2.0, 2.0, 1.0],
            "bands": ["bulk_density.foliage"],
        }
        body.update(overrides)
        return body

    def test_minimal_valid_request(self):
        req = CreateTreeInventoryRequest(**self._minimal())
        assert req.source_inventory_id == "abc123"
        assert req.resolution == (2.0, 2.0, 1.0)
        assert req.bands == [TreeBand.bulk_density_foliage]
        assert req.crown_profile_model == CrownProfileModel.purves
        assert req.biomass_model == BiomassModel.nsvb
        assert req.biomass_column is None
        assert req.moisture_model is None
        assert req.name == ""
        assert req.description == ""
        assert req.tags == []

    def test_source_inventory_id_is_required(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(
                resolution=[2.0, 2.0, 1.0],
                bands=["bulk_density.foliage"],
            )

    def test_resolution_is_required(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(
                source_inventory_id="abc",
                bands=["bulk_density.foliage"],
            )

    def test_bands_is_required(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(
                source_inventory_id="abc",
                resolution=[2.0, 2.0, 1.0],
            )

    def test_bands_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(**self._minimal(bands=[]))

    def test_duplicate_bands_rejected(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(
                **self._minimal(bands=["bulk_density.foliage", "bulk_density.foliage"])
            )

    def test_invalid_band_rejected(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(**self._minimal(bands=["not_a_band"]))

    @pytest.mark.parametrize(
        "resolution",
        [
            [0.0, 2.0, 1.0],
            [2.0, -1.0, 1.0],
            [2.0, 2.0, 0.0],
        ],
    )
    def test_non_positive_resolution_rejected(self, resolution):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(**self._minimal(resolution=resolution))

    def test_invalid_crown_profile_model_rejected(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(**self._minimal(crown_profile_model="watershed"))

    def test_invalid_biomass_model_rejected(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(**self._minimal(biomass_model="allometric"))

    def test_fuel_moisture_live_auto_populates_default_moisture_model(self):
        req = CreateTreeInventoryRequest(
            **self._minimal(bands=["bulk_density.foliage", "fuel_moisture.live"])
        )
        assert req.moisture_model is not None
        assert req.moisture_model.method == "uniform"
        assert req.moisture_model.live == 100.0

    def test_fuel_moisture_live_preserves_explicit_moisture_model(self):
        req = CreateTreeInventoryRequest(
            **self._minimal(
                bands=["bulk_density.foliage", "fuel_moisture.live"],
                moisture_model={"method": "uniform", "live": 75.0},
            )
        )
        assert req.moisture_model.method == "uniform"
        assert req.moisture_model.live == 75.0

    def test_moisture_model_stripped_when_fuel_moisture_band_absent(self):
        """moisture_model is dropped if fuel_moisture.live is not requested."""
        req = CreateTreeInventoryRequest(
            **self._minimal(
                moisture_model={"method": "uniform", "live": 50.0},
            )
        )
        assert req.moisture_model is None

    def test_moisture_model_invalid_method_rejected(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(
                **self._minimal(
                    bands=["bulk_density.foliage", "fuel_moisture.live"],
                    moisture_model={"method": "fosberg", "live": 100.0},
                )
            )

    def test_biomass_column_preserved_when_biomass_is_inventory(self):
        req = CreateTreeInventoryRequest(
            **self._minimal(
                biomass_model="inventory",
                biomass_column="my_fuel_load_col",
            )
        )
        assert req.biomass_column == "my_fuel_load_col"

    def test_biomass_column_scrubbed_for_non_inventory_biomass(self):
        req = CreateTreeInventoryRequest(
            **self._minimal(
                biomass_model="nsvb",
                biomass_column="ignored_column",
            )
        )
        assert req.biomass_column is None

    def test_full_request(self):
        req = CreateTreeInventoryRequest(
            name="Tree voxelization",
            description="FDS high-res",
            tags=["fds", "high-res"],
            source_inventory_id="inv123",
            resolution=[1.0, 1.0, 0.5],
            bands=[
                "bulk_density.foliage",
                "savr.foliage",
                "fuel_moisture.live",
                "volume_fraction",
            ],
            crown_profile_model="beta",
            biomass_model="jenkins",
            moisture_model={"method": "uniform", "live": 97.0},
        )
        assert req.name == "Tree voxelization"
        assert req.crown_profile_model == CrownProfileModel.beta
        assert req.biomass_model == BiomassModel.jenkins
        assert req.moisture_model.live == 97.0


class TestTreeInventorySource:
    def test_name_product_description_fixed(self):
        source = TreeInventorySource(
            source_inventory_id="inv123",
            resolution=[2.0, 2.0, 1.0],
            bands=[TreeBand.bulk_density_foliage],
            crown_profile_model=CrownProfileModel.purves,
            biomass_model=BiomassModel.nsvb,
        )
        assert source.name == "inventory"
        assert source.product == "tree"
        assert "tree" in source.description.lower()

    def test_model_dump_includes_resolved_defaults(self):
        source = TreeInventorySource(
            source_inventory_id="inv123",
            resolution=(2.0, 2.0, 1.0),
            bands=[TreeBand.bulk_density_foliage, TreeBand.fuel_moisture_live],
            crown_profile_model=CrownProfileModel.purves,
            biomass_model=BiomassModel.nsvb,
            moisture_model=UniformMoistureModel(live=85.0),
        )
        data = source.model_dump()
        assert data["name"] == "inventory"
        assert data["product"] == "tree"
        assert data["source_inventory_id"] == "inv123"
        assert data["resolution"] == (2.0, 2.0, 1.0)
        assert data["bands"] == ["bulk_density.foliage", "fuel_moisture.live"]
        assert data["crown_profile_model"] == "purves"
        assert data["biomass_model"] == "nsvb"
        assert data["moisture_model"] == {"method": "uniform", "live": 85.0}
