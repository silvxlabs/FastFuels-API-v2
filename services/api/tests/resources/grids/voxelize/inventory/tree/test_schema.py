"""
Unit tests for api/resources/grids/voxelize/inventory/tree/schema.py.

Pure schema tests with no external dependencies.
"""

import pytest
from api.resources.grids.schema import BandType
from api.resources.grids.voxelize.inventory.tree.examples import (
    ALL_TREE_INVENTORY_EXAMPLE_VALUES,
)
from api.resources.grids.voxelize.inventory.tree.schema import (
    TREE_BAND_DEFS,
    AllometryBiomassSource,
    AllometryMaxCrownRadiusSource,
    BiomassComponent,
    BiomassComponentState,
    BiomassEquations,
    CreateTreeInventoryRequest,
    CrownProfileModel,
    FineBiomassConfig,
    InventoryColumnMaxCrownRadiusSource,
    InventoryColumnsBiomassSource,
    MaxCrownRadiusUnit,
    MoistureModel,
    TreeBand,
    TreeInventoryVoxelizationSource,
    UniformMoistureValue,
    build_tree_bands,
)
from pydantic import ValidationError

from lib.units import validate_unit


class TestTreeBand:
    """TreeBand enum values match the spec."""

    def test_all_expected_bands_present(self):
        expected = {
            "bulk_density.foliage.live",
            "bulk_density.foliage.dead",
            "bulk_density.branchwood.live",
            "bulk_density.branchwood.dead",
            "bulk_density.fine.live",
            "bulk_density.fine.dead",
            "leaf_area_density",
            "fuel_moisture.live",
            "fuel_moisture.dead",
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
            (TreeBand.bulk_density_foliage_live, BandType.continuous, "kg/m**3"),
            (TreeBand.bulk_density_foliage_dead, BandType.continuous, "kg/m**3"),
            (TreeBand.bulk_density_branchwood_live, BandType.continuous, "kg/m**3"),
            (TreeBand.bulk_density_branchwood_dead, BandType.continuous, "kg/m**3"),
            (TreeBand.bulk_density_fine_live, BandType.continuous, "kg/m**3"),
            (TreeBand.bulk_density_fine_dead, BandType.continuous, "kg/m**3"),
            (TreeBand.leaf_area_density, BandType.continuous, "1/m"),
            (TreeBand.fuel_moisture_live, BandType.continuous, "%"),
            (TreeBand.fuel_moisture_dead, BandType.continuous, "%"),
            (TreeBand.savr_foliage, BandType.continuous, "1/m"),
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

    def test_all_units_are_canonical(self):
        for entry in TREE_BAND_DEFS.values():
            validate_unit(entry.get("unit"))


class TestBuildTreeBands:
    """build_tree_bands assigns indices in request order."""

    def test_single_band(self):
        bands = build_tree_bands([TreeBand.bulk_density_foliage_live])
        assert len(bands) == 1
        assert bands[0].key == "bulk_density.foliage.live"
        assert bands[0].name == "Live Foliage Bulk Density"
        assert bands[0].description
        assert bands[0].type == BandType.continuous
        assert bands[0].unit == "kg/m**3"
        assert bands[0].index == 0

    def test_all_bands_have_name_and_description(self):
        bands = build_tree_bands(list(TreeBand))
        for band in bands:
            assert band.name
            assert band.description

    def test_indices_match_request_order(self):
        requested = [
            TreeBand.savr_foliage,
            TreeBand.spcd,
            TreeBand.bulk_density_foliage_live,
        ]
        bands = build_tree_bands(requested)
        assert [b.index for b in bands] == [0, 1, 2]
        assert [b.key for b in bands] == [
            "savr.foliage",
            "spcd",
            "bulk_density.foliage.live",
        ]

    def test_all_bands(self):
        all_bands = list(TreeBand)
        bands = build_tree_bands(all_bands)
        assert len(bands) == len(all_bands)
        assert [b.index for b in bands] == list(range(len(all_bands)))


class TestUniformMoistureValue:
    def test_default_value(self):
        m = UniformMoistureValue()
        assert m.method == "uniform"
        assert m.value == 100.0

    def test_override_value(self):
        m = UniformMoistureValue(value=97.0)
        assert m.value == 97.0

    def test_method_cannot_be_overridden(self):
        with pytest.raises(ValidationError):
            UniformMoistureValue(method="fosberg")


class TestMoistureModel:
    def test_live_and_dead_values(self):
        model = MoistureModel(
            live={"method": "uniform", "value": 95.0},
            dead={"method": "uniform", "value": 8.0},
        )

        assert model.live.value == 95.0
        assert model.dead.value == 8.0


class TestBiomassSource:
    def test_default_allometry_foliage_config(self):
        biomass_source = AllometryBiomassSource()

        assert biomass_source.type == "allometry"
        assert biomass_source.equations == BiomassEquations.nsvb
        assert biomass_source.components == [BiomassComponent.foliage]
        assert biomass_source.component_states[BiomassComponent.foliage].live == 1.0
        assert biomass_source.component_states[BiomassComponent.foliage].dead == 0.0
        assert biomass_source.fine is None

    def test_inventory_column_direct_fine_config(self):
        biomass_source = InventoryColumnsBiomassSource(
            columns={
                "fine": {
                    "column": "fine_biomass",
                    "unit": "kg",
                }
            },
            components=["fine"],
        )

        assert biomass_source.type == "inventory_columns"
        assert biomass_source.columns[BiomassComponent.fine].column == "fine_biomass"
        assert biomass_source.components == [BiomassComponent.fine]
        assert biomass_source.component_states[BiomassComponent.fine].live == 1.0
        assert biomass_source.fine is None

    def test_component_state_partition_can_be_configured(self):
        biomass_source = AllometryBiomassSource(
            components=["foliage"],
            component_states={"foliage": {"live": 0.8, "dead": 0.2}},
        )

        assert biomass_source.component_states[BiomassComponent.foliage].live == 0.8
        assert biomass_source.component_states[BiomassComponent.foliage].dead == 0.2

    def test_component_state_partition_must_sum_to_one(self):
        with pytest.raises(ValidationError, match="sum to 1.0"):
            BiomassComponentState(live=0.8, dead=0.3)

    def test_component_state_keys_must_be_requested_components(self):
        with pytest.raises(ValidationError, match="component_states keys"):
            AllometryBiomassSource(
                components=["foliage"],
                component_states={"branchwood": {"live": 1.0, "dead": 0.0}},
            )

    def test_derived_fine_config(self):
        biomass_source = AllometryBiomassSource(
            equations="nsvb",
            components=["fine"],
            fine={
                "recipe": "foliage_plus_branchwood_fraction",
                "branchwood_fraction": 0.1,
            },
        )

        assert isinstance(biomass_source.fine, FineBiomassConfig)
        assert biomass_source.fine.recipe == "foliage_plus_branchwood_fraction"
        assert biomass_source.fine.branchwood_fraction == 0.1

    def test_inventory_units_must_be_kg(self):
        with pytest.raises(ValidationError):
            InventoryColumnsBiomassSource(
                columns={
                    "foliage": {
                        "column": "foliage_biomass",
                        "unit": "kg/m^2",
                    }
                },
                components=["foliage"],
            )

    def test_inventory_direct_component_requires_matching_column(self):
        with pytest.raises(ValidationError, match="missing a 'foliage' column"):
            InventoryColumnsBiomassSource(
                columns={
                    "fine": {
                        "column": "fine_biomass",
                        "unit": "kg",
                    }
                },
                components=["foliage"],
            )

    def test_inventory_derived_fine_requires_foliage_and_branchwood_columns(self):
        with pytest.raises(ValidationError, match="foliage, branchwood"):
            InventoryColumnsBiomassSource(
                columns={
                    "fine": {
                        "column": "fine_biomass",
                        "unit": "kg",
                    }
                },
                components=["fine"],
                fine={
                    "recipe": "foliage_plus_branchwood_fraction",
                    "branchwood_fraction": 0.1,
                },
            )

    def test_fine_config_requires_fine_component(self):
        with pytest.raises(ValidationError, match="requires 'fine' in components"):
            AllometryBiomassSource(
                components=["branchwood"],
                fine={
                    "recipe": "foliage_plus_branchwood_fraction",
                    "branchwood_fraction": 0.1,
                },
            )

    def test_allometry_fine_requires_fine_config(self):
        with pytest.raises(ValidationError, match="requires a fine configuration"):
            AllometryBiomassSource(components=["fine"])

    def test_duplicate_components_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate biomass components"):
            AllometryBiomassSource(components=["foliage", "foliage"])

    def test_fine_config_requires_recipe_and_fraction(self):
        with pytest.raises(ValidationError):
            FineBiomassConfig(
                recipe="foliage_plus_branchwood_fraction",
            )

    def test_branchwood_fraction_must_be_fraction(self):
        with pytest.raises(ValidationError):
            FineBiomassConfig(
                recipe="foliage_plus_branchwood_fraction",
                branchwood_fraction=1.5,
            )


class TestMaxCrownRadiusSource:
    def test_default_allometry_source(self):
        source = AllometryMaxCrownRadiusSource()
        assert source.type == "allometry"

    def test_inventory_column_source(self):
        source = InventoryColumnMaxCrownRadiusSource(column="lidar_max_radius")
        assert source.type == "inventory_column"
        assert source.column == "lidar_max_radius"
        assert source.unit == MaxCrownRadiusUnit.m

    def test_inventory_column_unit_must_be_meters(self):
        with pytest.raises(ValidationError):
            InventoryColumnMaxCrownRadiusSource(column="r", unit="ft")

    def test_inventory_column_requires_column(self):
        with pytest.raises(ValidationError):
            InventoryColumnMaxCrownRadiusSource()

    def test_inventory_column_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            InventoryColumnMaxCrownRadiusSource(column="r", species="something")

    def test_unknown_discriminator_rejected(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(
                source_inventory_id="abc",
                resolution={"horizontal": 2.0, "vertical": 1.0},
                bands=["bulk_density.foliage.live"],
                max_crown_radius_source={"type": "lookup_table", "column": "r"},
            )


class TestCreateTreeInventoryRequest:
    """Validation rules for the request body."""

    def _minimal(self, **overrides) -> dict:
        body = {
            "source_inventory_id": "abc123",
        }
        body.update(overrides)
        return body

    def test_minimal_valid_request(self):
        req = CreateTreeInventoryRequest(**self._minimal())
        assert req.source_inventory_id == "abc123"
        assert req.resolution.horizontal == 2.0
        assert req.resolution.vertical == 1.0
        assert req.bands == [TreeBand.bulk_density_foliage_live]
        assert req.crown_profile_model == CrownProfileModel.purves
        assert req.biomass_source.equations == BiomassEquations.nsvb
        assert req.biomass_source.components == [BiomassComponent.foliage]
        assert req.max_crown_radius_source.type == "allometry"
        assert req.moisture_model is None
        assert req.name == ""
        assert req.description == ""
        assert req.tags == []

    def test_max_crown_radius_inventory_column_accepted(self):
        req = CreateTreeInventoryRequest(
            **self._minimal(
                max_crown_radius_source={
                    "type": "inventory_column",
                    "column": "lidar_max_radius",
                    "unit": "m",
                }
            )
        )
        assert isinstance(
            req.max_crown_radius_source, InventoryColumnMaxCrownRadiusSource
        )
        assert req.max_crown_radius_source.column == "lidar_max_radius"

    def test_source_inventory_id_is_required(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(
                resolution={"horizontal": 2.0, "vertical": 1.0},
                bands=["bulk_density.foliage.live"],
            )

    def test_resolution_defaults_to_two_by_one(self):
        req = CreateTreeInventoryRequest(
            source_inventory_id="abc",
            bands=["bulk_density.foliage.live"],
        )
        assert req.resolution.horizontal == 2.0
        assert req.resolution.vertical == 1.0

    def test_bands_defaults_to_live_foliage_bulk_density(self):
        req = CreateTreeInventoryRequest(
            source_inventory_id="abc",
            resolution={"horizontal": 2.0, "vertical": 1.0},
        )
        assert req.bands == [TreeBand.bulk_density_foliage_live]

    def test_bands_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(**self._minimal(bands=[]))

    def test_duplicate_bands_rejected(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(
                **self._minimal(
                    bands=["bulk_density.foliage.live", "bulk_density.foliage.live"]
                )
            )

    def test_invalid_band_rejected(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(**self._minimal(bands=["not_a_band"]))

    @pytest.mark.parametrize(
        "resolution",
        [
            {"horizontal": 0.0, "vertical": 1.0},
            {"horizontal": -1.0, "vertical": 1.0},
            {"horizontal": 2.0, "vertical": 0.0},
        ],
    )
    def test_non_positive_resolution_rejected(self, resolution):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(**self._minimal(resolution=resolution))

    def test_invalid_crown_profile_model_rejected(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(**self._minimal(crown_profile_model="watershed"))

    def test_invalid_biomass_source_rejected(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(
                **self._minimal(
                    biomass_source={
                        "type": "allometry",
                        "equations": "allometric",
                        "components": ["foliage"],
                    }
                )
            )

    def test_legacy_biomass_model_field_rejected(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(**self._minimal(biomass_model="nsvb"))

    def test_legacy_biomass_field_rejected(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(
                **self._minimal(
                    biomass={
                        "type": "allometry",
                        "equations": "nsvb",
                        "components": ["foliage"],
                    }
                )
            )

    def test_fuel_moisture_live_auto_populates_default_moisture_model(self):
        req = CreateTreeInventoryRequest(
            **self._minimal(bands=["bulk_density.foliage.live", "fuel_moisture.live"])
        )
        assert req.moisture_model is not None
        assert req.moisture_model.live.method == "uniform"
        assert req.moisture_model.live.value == 100.0

    def test_fuel_moisture_dead_auto_populates_default_moisture_model(self):
        req = CreateTreeInventoryRequest(
            **self._minimal(bands=["bulk_density.foliage.dead", "fuel_moisture.dead"])
        )
        assert req.moisture_model is not None
        assert req.moisture_model.dead.method == "uniform"
        assert req.moisture_model.dead.value == 10.0
        assert req.moisture_model.live is None

    def test_fuel_moisture_live_preserves_explicit_moisture_model(self):
        req = CreateTreeInventoryRequest(
            **self._minimal(
                bands=["bulk_density.foliage.live", "fuel_moisture.live"],
                moisture_model={"live": {"method": "uniform", "value": 75.0}},
            )
        )
        assert req.moisture_model.live.method == "uniform"
        assert req.moisture_model.live.value == 75.0

    def test_moisture_model_stripped_when_fuel_moisture_band_absent(self):
        """moisture_model is dropped if fuel_moisture.live is not requested."""
        req = CreateTreeInventoryRequest(
            **self._minimal(
                moisture_model={"live": {"method": "uniform", "value": 50.0}},
            )
        )
        assert req.moisture_model is None

    def test_moisture_model_invalid_method_rejected(self):
        with pytest.raises(ValidationError):
            CreateTreeInventoryRequest(
                **self._minimal(
                    bands=["bulk_density.foliage.live", "fuel_moisture.live"],
                    moisture_model={"live": {"method": "fosberg", "value": 100.0}},
                )
            )

    def test_full_request(self):
        req = CreateTreeInventoryRequest(
            name="Tree voxelization",
            description="FDS high-res",
            tags=["fds", "high-res"],
            source_inventory_id="inv123",
            resolution={"horizontal": 1.0, "vertical": 0.5},
            bands=[
                "bulk_density.foliage.live",
                "savr.foliage",
                "fuel_moisture.live",
                "volume_fraction",
            ],
            crown_profile_model="beta",
            biomass_source={
                "type": "allometry",
                "equations": "jenkins",
                "components": ["foliage"],
            },
            moisture_model={"live": {"method": "uniform", "value": 97.0}},
        )
        assert req.name == "Tree voxelization"
        assert req.crown_profile_model == CrownProfileModel.beta
        assert req.biomass_source.equations == BiomassEquations.jenkins
        assert req.moisture_model.live.value == 97.0

    @pytest.mark.parametrize(
        "example_name,example_value", ALL_TREE_INVENTORY_EXAMPLE_VALUES
    )
    def test_documented_examples_are_schema_valid(self, example_name, example_value):
        body = {**example_value, "source_inventory_id": "abc123"}
        req = CreateTreeInventoryRequest(**body)
        assert req.source_inventory_id == "abc123", example_name

    def test_band_component_not_in_biomass_components_rejected(self):
        """bulk_density bands must reference a configured biomass component."""
        with pytest.raises(ValidationError, match="biomass_source.components"):
            CreateTreeInventoryRequest(
                **self._minimal(
                    bands=["bulk_density.branchwood.live"],
                    biomass_source={
                        "type": "allometry",
                        "equations": "nsvb",
                        "components": ["foliage"],
                    },
                )
            )

    def test_inventory_columns_band_component_mismatch_rejected(self):
        """Same gate applies for inventory_columns biomass sources."""
        with pytest.raises(ValidationError, match="biomass_source.components"):
            CreateTreeInventoryRequest(
                **self._minimal(
                    bands=["bulk_density.foliage.live"],
                    biomass_source={
                        "type": "inventory_columns",
                        "columns": {"branchwood": {"column": "bw_kg", "unit": "kg"}},
                        "components": ["branchwood"],
                    },
                )
            )

    def test_band_component_matching_components_accepted(self):
        """foliage bands with foliage configured pass."""
        req = CreateTreeInventoryRequest(
            **self._minimal(
                bands=["bulk_density.foliage.live", "bulk_density.foliage.dead"],
                biomass_source={
                    "type": "allometry",
                    "equations": "nsvb",
                    "components": ["foliage"],
                    "component_states": {"foliage": {"live": 0.9, "dead": 0.1}},
                },
            )
        )
        assert req.biomass_source.components == [BiomassComponent.foliage]

    def test_non_bulk_density_bands_are_unaffected_by_component_check(self):
        """volume_fraction / spcd / tree_id / savr / fuel_moisture don't gate."""
        req = CreateTreeInventoryRequest(
            **self._minimal(
                bands=["volume_fraction", "spcd", "tree_id", "savr.foliage"],
                biomass_source={
                    "type": "allometry",
                    "equations": "nsvb",
                    "components": ["foliage"],
                },
            )
        )
        assert req.bands == [
            TreeBand.volume_fraction,
            TreeBand.spcd,
            TreeBand.tree_id,
            TreeBand.savr_foliage,
        ]

    def test_band_component_mismatch_lists_every_offending_band(self):
        """Multiple unconfigured bulk_density bands are reported together."""
        with pytest.raises(ValidationError) as exc_info:
            CreateTreeInventoryRequest(
                **self._minimal(
                    bands=[
                        "bulk_density.branchwood.live",
                        "bulk_density.fine.live",
                    ],
                    biomass_source={
                        "type": "allometry",
                        "equations": "nsvb",
                        "components": ["foliage"],
                    },
                )
            )
        message = str(exc_info.value)
        assert "branchwood" in message
        assert "fine" in message


class TestSeedField:
    """The `seed` field makes voxelization reproducible.

    On the request: optional with an auto-generated default so users can
    choose reproducibility or leave it to the API.
    On the persisted source: required — we always record which seed drove
    the output.
    """

    def _minimal_body(self, **overrides):
        body = {
            "source_inventory_id": "inv1",
            "resolution": {"horizontal": 2.0, "vertical": 1.0},
            "bands": ["bulk_density.foliage.live"],
        }
        body.update(overrides)
        return body

    def test_seed_autogenerated_when_absent(self):
        req = CreateTreeInventoryRequest(**self._minimal_body())
        assert isinstance(req.seed, int)
        assert req.seed > 0

    def test_seed_passes_through_when_supplied(self):
        req = CreateTreeInventoryRequest(**self._minimal_body(seed=42))
        assert req.seed == 42

    def test_auto_seeds_differ_across_requests(self):
        """Two omitted-seed requests should get different values.

        Uses the same randint space as pim's seed generator (1..1B); a
        collision is possible but vanishingly unlikely in a single pair.
        """
        a = CreateTreeInventoryRequest(**self._minimal_body())
        b = CreateTreeInventoryRequest(**self._minimal_body())
        assert a.seed != b.seed

    def test_source_requires_seed(self):
        """TreeInventoryVoxelizationSource persists the seed — no sensible default exists."""
        with pytest.raises(ValidationError):
            TreeInventoryVoxelizationSource(
                source_inventory_id="inv1",
                resolution={"horizontal": 2.0, "vertical": 1.0},
                bands=[TreeBand.bulk_density_foliage_live],
                crown_profile_model=CrownProfileModel.purves,
                biomass_source=AllometryBiomassSource(),
            )


class TestTreeInventoryVoxelizationSource:
    def test_discriminators_fixed(self):
        source = TreeInventoryVoxelizationSource(
            source_inventory_id="inv123",
            resolution={"horizontal": 2.0, "vertical": 1.0},
            bands=[TreeBand.bulk_density_foliage_live],
            crown_profile_model=CrownProfileModel.purves,
            biomass_source=AllometryBiomassSource(),
            seed=42,
        )
        assert source.operation == "voxelize"
        assert source.input == "inventory"
        assert source.entity == "tree"

    def test_source_inventory_checksum_defaults_to_none(self):
        source = TreeInventoryVoxelizationSource(
            source_inventory_id="inv123",
            resolution={"horizontal": 2.0, "vertical": 1.0},
            bands=[TreeBand.bulk_density_foliage_live],
            crown_profile_model=CrownProfileModel.purves,
            biomass_source=AllometryBiomassSource(),
            seed=42,
        )
        assert source.source_inventory_checksum is None

    def test_source_inventory_checksum_round_trips(self):
        source = TreeInventoryVoxelizationSource(
            source_inventory_id="inv123",
            source_inventory_checksum="sum123",
            resolution={"horizontal": 2.0, "vertical": 1.0},
            bands=[TreeBand.bulk_density_foliage_live],
            crown_profile_model=CrownProfileModel.purves,
            biomass_source=AllometryBiomassSource(),
            seed=42,
        )
        assert source.source_inventory_checksum == "sum123"
        data = source.model_dump(mode="json")
        assert data["source_inventory_checksum"] == "sum123"

    def test_max_crown_radius_source_persists_inventory_column(self):
        source = TreeInventoryVoxelizationSource(
            source_inventory_id="inv123",
            resolution={"horizontal": 2.0, "vertical": 1.0},
            bands=[TreeBand.bulk_density_foliage_live],
            crown_profile_model=CrownProfileModel.purves,
            biomass_source=AllometryBiomassSource(),
            max_crown_radius_source=InventoryColumnMaxCrownRadiusSource(
                column="lidar_max_radius",
            ),
            seed=42,
        )
        data = source.model_dump(mode="json", exclude_none=True)
        assert data["max_crown_radius_source"] == {
            "type": "inventory_column",
            "column": "lidar_max_radius",
            "unit": "m",
        }

    def test_max_crown_radius_source_defaults_to_allometry_on_persisted_source(self):
        source = TreeInventoryVoxelizationSource(
            source_inventory_id="inv123",
            resolution={"horizontal": 2.0, "vertical": 1.0},
            bands=[TreeBand.bulk_density_foliage_live],
            crown_profile_model=CrownProfileModel.purves,
            biomass_source=AllometryBiomassSource(),
            seed=42,
        )
        assert source.max_crown_radius_source.type == "allometry"

    def test_model_dump_includes_resolved_defaults(self):
        source = TreeInventoryVoxelizationSource(
            source_inventory_id="inv123",
            resolution={"horizontal": 2.0, "vertical": 1.0},
            bands=[TreeBand.bulk_density_foliage_live, TreeBand.fuel_moisture_live],
            crown_profile_model=CrownProfileModel.purves,
            biomass_source=AllometryBiomassSource(),
            moisture_model=MoistureModel(
                live=UniformMoistureValue(value=85.0),
            ),
            seed=42,
        )
        data = source.model_dump(mode="json", exclude_none=True)
        assert data["operation"] == "voxelize"
        assert data["input"] == "inventory"
        assert data["entity"] == "tree"
        assert data["source_inventory_id"] == "inv123"
        assert data["resolution"] == {"horizontal": 2.0, "vertical": 1.0}
        assert data["bands"] == ["bulk_density.foliage.live", "fuel_moisture.live"]
        assert data["crown_profile_model"] == "purves"
        assert data["biomass_source"] == {
            "type": "allometry",
            "equations": "nsvb",
            "components": ["foliage"],
            "component_states": {"foliage": {"live": 1.0, "dead": 0.0}},
        }
        assert data["moisture_model"] == {"live": {"method": "uniform", "value": 85.0}}
        assert data["seed"] == 42

    def test_band_component_mismatch_rejected(self):
        """Source rejects bulk_density bands whose component isn't configured.

        Defense in depth: even if a stored doc somehow held a mismatched
        bands/biomass_source pair, reconstruction fails loudly.
        """
        with pytest.raises(ValidationError, match="biomass_source.components"):
            TreeInventoryVoxelizationSource(
                source_inventory_id="inv1",
                resolution={"horizontal": 2.0, "vertical": 1.0},
                bands=[TreeBand.bulk_density_branchwood_live],
                crown_profile_model=CrownProfileModel.purves,
                biomass_source=AllometryBiomassSource(),  # foliage default
                seed=42,
            )
