"""
Unit tests for api/v2/resources/grids/canopy/inventory/schema.py

Pure schema tests: request defaults and validators, per-band method
resolution, available-fuel resolution, band definitions, the persisted
source round trip, and the documented OpenAPI examples.
"""

import pytest
from api.resources.grids.canopy.inventory.examples import (
    CREATE_INVENTORY_CANOPY_OPENAPI_EXAMPLES,
    INVENTORY_CANOPY_EXAMPLE_VALUES,
)
from api.resources.grids.canopy.inventory.schema import (
    CFL_BAND_DEF,
    AllometryCanopyBiomassSource,
    CanopyAllometryMaxCrownRadiusSource,
    CanopyAvailableFuel,
    CanopyBiomassEquations,
    CanopyBranchwoodSizePartition,
    CanopyCbdLoadOverDepth,
    CanopyCbdRunningMean,
    CanopyCcCrownUnion,
    CanopyCrownWidthEquations,
    CanopyFuelcalcCrownClassAdjustment,
    CanopyHorizontalDistribution,
    CanopyNoCrownClassAdjustment,
    CanopyProfileThreshold,
    CanopyRunningMeanEdge,
    CanopySpeciesInclusion,
    CanopyVerticalDistribution,
    CreateInventoryCanopyRequest,
    InventoryCanopyBand,
    InventoryCanopySource,
    InventoryColumnCanopyBiomassSource,
    build_inventory_canopy_bands,
)
from api.resources.grids.canopy.schema import LANDFIRE_CANOPY_BAND_DEFS
from api.resources.grids.schema import BandType
from api.resources.grids.voxelize.inventory.tree.schema import (
    InventoryColumnMaxCrownRadiusSource,
)
from pydantic import ValidationError

INVENTORY_ID = "9c1f2ab4708d4290a8ab6ecf35f21ab4"


def make_request(**overrides) -> CreateInventoryCanopyRequest:
    return CreateInventoryCanopyRequest(source_inventory_id=INVENTORY_ID, **overrides)


class TestRequestDefaults:
    def test_source_inventory_id_is_required(self):
        with pytest.raises(ValidationError):
            CreateInventoryCanopyRequest()

    def test_minimal_request_resolves_fuelcalc_method_with_nsvb_allometry(self):
        req = make_request()
        assert req.bands == [
            InventoryCanopyBand.cbd,
            InventoryCanopyBand.cbh,
            InventoryCanopyBand.chm,
            InventoryCanopyBand.cc,
        ]
        assert req.alignment.target == "domain"
        assert req.alignment.resolution is None
        assert isinstance(req.biomass_source, AllometryCanopyBiomassSource)
        assert req.biomass_source.equations is CanopyBiomassEquations.nsvb
        assert req.species_inclusion is CanopySpeciesInclusion.all_species
        assert isinstance(req.crown_class_adjustment, CanopyNoCrownClassAdjustment)
        assert req.min_tree_height == 0.0
        assert req.vertical_distribution is CanopyVerticalDistribution.reinhardt_2006
        assert req.layer_depth == pytest.approx(0.3048)
        assert (
            req.horizontal_distribution is CanopyHorizontalDistribution.crown_projected
        )
        assert isinstance(
            req.max_crown_radius_source, CanopyAllometryMaxCrownRadiusSource
        )

    def test_minimal_request_resolves_available_fuel_defaults(self):
        # NSVB defaults to the national `none` partition (prices every
        # species) with the 0.075 total-branchwood fraction.
        req = make_request()
        assert isinstance(req.available_fuel, CanopyAvailableFuel)
        assert req.available_fuel.foliage_fraction == 1.0
        assert req.available_fuel.branchwood.fraction == 0.075
        assert (
            req.available_fuel.branchwood.size_partition
            is CanopyBranchwoodSizePartition.none
        )

    def test_minimal_request_resolves_band_method_defaults(self):
        req = make_request()
        assert isinstance(req.cbd, CanopyCbdRunningMean)
        assert req.cbd.window == 3.0
        assert isinstance(req.cbh, CanopyProfileThreshold)
        assert req.cbh.threshold == 0.012
        assert req.cbh.relative_threshold_fraction == 0.1
        assert req.cbh.smoothing_window is None
        assert isinstance(req.chm, CanopyProfileThreshold)
        assert isinstance(req.cc, CanopyCcCrownUnion)

    def test_metadata_defaults(self):
        req = make_request()
        assert req.name == ""
        assert req.description == ""
        assert req.tags == []

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            make_request(unknown_field=1)

    def test_native_alignment_target_accepted_by_schema_rejected_by_router(self):
        # The schema accepts the shared alignment union; the router rejects
        # `native` with a 422. Document the schema-level behaviour here so a
        # future tightening is a deliberate change.
        req = make_request(alignment={"target": "native"})
        assert req.alignment.target == "native"


class TestBands:
    def test_duplicate_bands_rejected(self):
        with pytest.raises(ValidationError, match="uplicate"):
            make_request(bands=["cbd", "cbd"])

    def test_empty_bands_rejected(self):
        with pytest.raises(ValidationError):
            make_request(bands=[])

    def test_unknown_band_rejected(self):
        with pytest.raises(ValidationError):
            make_request(bands=["ch"])

    def test_cfl_is_opt_in(self):
        req = make_request(bands=["cbd", "cfl"])
        assert InventoryCanopyBand.cfl in req.bands
        assert InventoryCanopyBand.cfl not in make_request().bands

    def test_unrequested_bands_have_null_methods(self):
        req = make_request(bands=["cbd"])
        assert isinstance(req.cbd, CanopyCbdRunningMean)
        assert req.cbh is None
        assert req.chm is None
        assert req.cc is None

    def test_method_for_unrequested_band_rejected(self):
        with pytest.raises(ValidationError, match="'cc' band was not requested"):
            make_request(bands=["cbd"], cc={"method": "crown_union"})

    def test_explicit_null_method_for_unrequested_band_allowed(self):
        req = make_request(bands=["cc"], cbd=None)
        assert req.cbd is None
        assert isinstance(req.cc, CanopyCcCrownUnion)

    def test_explicit_null_method_for_requested_band_resolves_default(self):
        req = make_request(bands=["cbd"], cbd=None)
        assert isinstance(req.cbd, CanopyCbdRunningMean)

    def test_cfl_has_no_method_field(self):
        with pytest.raises(ValidationError):
            make_request(bands=["cfl"], cfl={"method": "anything"})


class TestBuildInventoryCanopyBands:
    def test_landfire_parity_bands_reuse_landfire_definitions(self):
        bands = build_inventory_canopy_bands(
            [InventoryCanopyBand.cbd, InventoryCanopyBand.cc]
        )
        assert [b.index for b in bands] == [0, 1]
        for band, key in zip(bands, ("cbd", "cc")):
            expected = LANDFIRE_CANOPY_BAND_DEFS[key]
            assert band.key == key
            assert band.unit == expected["unit"]
            assert band.name == expected["name"]

    def test_cfl_band_definition(self):
        (band,) = build_inventory_canopy_bands([InventoryCanopyBand.cfl])
        assert band.index == 0
        assert band.key == "cfl"
        assert band.unit == "kg/m**2"
        assert band.type is BandType.continuous
        for key, expected in CFL_BAND_DEF.items():
            assert getattr(band, key) == expected

    def test_indices_follow_request_order(self):
        bands = build_inventory_canopy_bands(
            [InventoryCanopyBand.cfl, InventoryCanopyBand.chm, InventoryCanopyBand.cbh]
        )
        assert [(b.index, b.key) for b in bands] == [(0, "cfl"), (1, "chm"), (2, "cbh")]


class TestBiomassSourceAndAvailableFuel:
    def test_brown_1978_resolves_native_size_partition(self):
        req = make_request(
            biomass_source={"type": "allometry", "equations": "brown_1978"}
        )
        assert (
            req.available_fuel.branchwood.size_partition
            is CanopyBranchwoodSizePartition.equations
        )

    @pytest.mark.parametrize("equations", ["nsvb", "jenkins"])
    def test_total_branchwood_families_resolve_none(self, equations):
        # The national families default to the `none` partition (fraction of
        # total branchwood, 0.075) — the only basis that prices every species.
        req = make_request(biomass_source={"type": "allometry", "equations": equations})
        assert (
            req.available_fuel.branchwood.size_partition
            is CanopyBranchwoodSizePartition.none
        )
        assert req.available_fuel.branchwood.fraction == 0.075

    @pytest.mark.parametrize("equations", ["nsvb", "jenkins"])
    def test_equations_partition_rejected_without_size_classes(self, equations):
        with pytest.raises(ValidationError, match="requires biomass equations"):
            make_request(
                biomass_source={"type": "allometry", "equations": equations},
                available_fuel={"branchwood": {"size_partition": "equations"}},
            )

    @pytest.mark.parametrize("equations", ["nsvb", "jenkins", "brown_1978"])
    def test_none_partition_valid_with_every_family(self, equations):
        req = make_request(
            biomass_source={"type": "allometry", "equations": equations},
            available_fuel={"branchwood": {"size_partition": "none", "fraction": 0.2}},
        )
        assert (
            req.available_fuel.branchwood.size_partition
            is CanopyBranchwoodSizePartition.none
        )
        assert req.available_fuel.branchwood.fraction == 0.2

    def test_explicit_partition_is_preserved(self):
        req = make_request(
            available_fuel={"branchwood": {"size_partition": "brown_proportions"}}
        )
        assert (
            req.available_fuel.branchwood.size_partition
            is CanopyBranchwoodSizePartition.brown_proportions
        )

    @pytest.mark.parametrize("field", ["foliage_fraction"])
    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_foliage_fraction_bounds(self, field, value):
        with pytest.raises(ValidationError):
            make_request(available_fuel={field: value})

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_branchwood_fraction_bounds(self, value):
        with pytest.raises(ValidationError):
            make_request(available_fuel={"branchwood": {"fraction": value}})

    def test_inventory_column_source_nulls_available_fuel(self):
        req = make_request(
            biomass_source={"type": "inventory_column", "column": "canopy_fuel_kg"}
        )
        assert isinstance(req.biomass_source, InventoryColumnCanopyBiomassSource)
        assert req.biomass_source.column == "canopy_fuel_kg"
        assert req.biomass_source.unit == "kg"
        assert req.available_fuel is None

    def test_inventory_column_source_rejects_available_fuel(self):
        with pytest.raises(ValidationError, match="cannot be combined"):
            make_request(
                biomass_source={"type": "inventory_column", "column": "fuel"},
                available_fuel={"foliage_fraction": 0.5},
            )

    def test_inventory_column_source_allows_explicit_null_available_fuel(self):
        req = make_request(
            biomass_source={"type": "inventory_column", "column": "fuel"},
            available_fuel=None,
        )
        assert req.available_fuel is None

    def test_biomass_source_discriminator(self):
        with pytest.raises(ValidationError):
            make_request(biomass_source={"type": "lookup"})

    def test_unknown_equations_rejected(self):
        with pytest.raises(ValidationError):
            make_request(biomass_source={"type": "allometry", "equations": "chojnacky"})


class TestAdjustments:
    def test_species_inclusion_values(self):
        assert (
            make_request(species_inclusion="fuelcalc_default").species_inclusion
            is CanopySpeciesInclusion.fuelcalc_default
        )
        with pytest.raises(ValidationError):
            make_request(species_inclusion="conifers")

    def test_fuelcalc_crown_class_adjustment(self):
        req = make_request(crown_class_adjustment={"method": "fuelcalc_table"})
        assert isinstance(
            req.crown_class_adjustment, CanopyFuelcalcCrownClassAdjustment
        )
        assert req.crown_class_adjustment.missing_crown_class == "other_none"

    def test_crown_class_adjustment_discriminator(self):
        with pytest.raises(ValidationError):
            make_request(crown_class_adjustment={"method": "dominant"})

    @pytest.mark.parametrize("value", [-1.0, 50.1])
    def test_min_tree_height_bounds(self, value):
        with pytest.raises(ValidationError):
            make_request(min_tree_height=value)

    def test_min_tree_height_ffe_value(self):
        assert make_request(min_tree_height=1.83).min_tree_height == 1.83


class TestDistributions:
    def test_vertical_distribution_values(self):
        assert (
            make_request(vertical_distribution="uniform").vertical_distribution
            is CanopyVerticalDistribution.uniform
        )
        with pytest.raises(ValidationError):
            make_request(vertical_distribution="beta")

    def test_horizontal_distribution_values(self):
        assert (
            make_request(horizontal_distribution="stem").horizontal_distribution
            is CanopyHorizontalDistribution.stem
        )
        with pytest.raises(ValidationError):
            make_request(horizontal_distribution="nearest")

    @pytest.mark.parametrize("value", [0.0, 0.009, 5.01])
    def test_layer_depth_bounds(self, value):
        with pytest.raises(ValidationError):
            make_request(layer_depth=value)

    def test_max_crown_radius_from_column(self):
        req = make_request(
            max_crown_radius_source={"type": "inventory_column", "column": "crown_r"}
        )
        assert isinstance(
            req.max_crown_radius_source, InventoryColumnMaxCrownRadiusSource
        )
        assert req.max_crown_radius_source.column == "crown_r"


class TestCbdMethods:
    def test_running_mean_null_window_disables_smoothing(self):
        req = make_request(cbd={"method": "maximum_running_mean", "window": None})
        assert req.cbd.window is None

    @pytest.mark.parametrize("window", [0.0, -1.0, 15.1])
    def test_running_mean_window_bounds(self, window):
        with pytest.raises(ValidationError):
            make_request(cbd={"method": "maximum_running_mean", "window": window})

    def test_running_mean_forbids_extra_fields(self):
        with pytest.raises(ValidationError):
            make_request(cbd={"method": "maximum_running_mean", "threshold": 0.012})

    def test_load_over_depth_default_depth_is_canopy_depth(self):
        req = make_request(cbd={"method": "load_over_depth"})
        assert isinstance(req.cbd, CanopyCbdLoadOverDepth)
        assert req.cbd.depth == "canopy_depth"

    def test_load_over_depth_canopy_depth_requires_threshold_cbh_and_chm(self):
        with pytest.raises(ValidationError, match="cbh, chm"):
            make_request(bands=["cbd"], cbd={"method": "load_over_depth"})

    def test_load_over_depth_canopy_depth_rejects_other_cbh_method(self):
        with pytest.raises(
            ValidationError, match="Missing or using another method: cbh"
        ):
            make_request(
                cbd={"method": "load_over_depth"}, cbh={"method": "mean_crown_base"}
            )

    def test_load_over_depth_canopy_depth_valid_with_default_thresholds(self):
        req = make_request(cbd={"method": "load_over_depth"})
        assert isinstance(req.cbh, CanopyProfileThreshold)
        assert isinstance(req.chm, CanopyProfileThreshold)

    def test_load_over_depth_other_depths_have_no_dependency(self):
        req = make_request(
            bands=["cbd"],
            cbd={"method": "load_over_depth", "depth": "mean_crown_length"},
        )
        assert req.cbd.depth == "mean_crown_length"
        assert req.cbh is None

    def test_cbd_rejects_methods_of_other_bands(self):
        with pytest.raises(ValidationError):
            make_request(cbd={"method": "bulk_density_threshold"})


class TestCbhChmMethods:
    def test_flat_threshold_via_null_relative_fraction(self):
        req = make_request(
            cbh={
                "method": "bulk_density_threshold",
                "threshold": 0.037,
                "relative_threshold_fraction": None,
            }
        )
        assert req.cbh.threshold == 0.037
        assert req.cbh.relative_threshold_fraction is None

    @pytest.mark.parametrize("threshold", [0.0, -0.01, 1.01])
    def test_threshold_bounds(self, threshold):
        with pytest.raises(ValidationError):
            make_request(
                cbh={"method": "bulk_density_threshold", "threshold": threshold}
            )

    @pytest.mark.parametrize("fraction", [0.0, 1.01])
    def test_relative_fraction_bounds(self, fraction):
        with pytest.raises(ValidationError):
            make_request(
                cbh={
                    "method": "bulk_density_threshold",
                    "relative_threshold_fraction": fraction,
                }
            )

    def test_smoothing_window(self):
        req = make_request(
            chm={"method": "bulk_density_threshold", "smoothing_window": 1.524}
        )
        assert req.chm.smoothing_window == 1.524
        with pytest.raises(ValidationError):
            make_request(
                chm={"method": "bulk_density_threshold", "smoothing_window": 0.0}
            )

    def test_cbh_and_chm_thresholds_are_independent_objects(self):
        req = make_request(
            cbh={"method": "bulk_density_threshold", "threshold": 0.05},
        )
        assert req.cbh.threshold == 0.05
        assert req.chm.threshold == 0.012

    def test_mean_crown_base(self):
        req = make_request(cbh={"method": "mean_crown_base"})
        assert req.cbh.method == "mean_crown_base"

    def test_height_percentile(self):
        req = make_request(chm={"method": "height_percentile", "percentile": 95})
        assert req.chm.percentile == 95.0
        with pytest.raises(ValidationError):
            make_request(chm={"method": "height_percentile", "percentile": 0.5})

    def test_cbh_rejects_chm_only_method(self):
        with pytest.raises(ValidationError):
            make_request(cbh={"method": "height_percentile"})

    def test_chm_rejects_cbh_only_method(self):
        with pytest.raises(ValidationError):
            make_request(chm={"method": "mean_crown_base"})


class TestCcMethods:
    def test_crown_overlap(self):
        assert make_request(cc={"method": "crown_overlap"}).cc.method == "crown_overlap"

    def test_cover_fraction_default_height(self):
        req = make_request(cc={"method": "cover_fraction"})
        assert req.cc.height_threshold == 2.0

    @pytest.mark.parametrize("value", [-0.1, 100.1])
    def test_cover_fraction_height_bounds(self, value):
        with pytest.raises(ValidationError):
            make_request(cc={"method": "cover_fraction", "height_threshold": value})

    def test_cc_rejects_profile_methods(self):
        with pytest.raises(ValidationError):
            make_request(cc={"method": "bulk_density_threshold"})


class TestInventoryCanopySource:
    def _source_from_request(self, req: CreateInventoryCanopyRequest):
        return InventoryCanopySource(
            source_inventory_id=req.source_inventory_id,
            source_inventory_checksum="abc123",
            alignment=req.alignment,
            bands=req.bands,
            biomass_source=req.biomass_source,
            available_fuel=req.available_fuel,
            species_inclusion=req.species_inclusion,
            crown_class_adjustment=req.crown_class_adjustment,
            min_tree_height=req.min_tree_height,
            vertical_distribution=req.vertical_distribution,
            layer_depth=req.layer_depth,
            horizontal_distribution=req.horizontal_distribution,
            max_crown_radius_source=req.max_crown_radius_source,
            cbd=req.cbd,
            cbh=req.cbh,
            chm=req.chm,
            cc=req.cc,
        )

    def test_name_product_description_are_fixed(self):
        source = self._source_from_request(make_request())
        assert source.name == "canopy"
        assert source.product == "inventory"
        assert (
            source.description == "Canopy fuel metrics computed from a tree inventory"
        )
        with pytest.raises(ValidationError):
            InventoryCanopySource(
                product="landfire", source_inventory_id=INVENTORY_ID, bands=["cbd"]
            )

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            InventoryCanopySource(bands=["cbd"])
        with pytest.raises(ValidationError):
            InventoryCanopySource(source_inventory_id=INVENTORY_ID)

    def test_checksum_defaults_to_none(self):
        source = InventoryCanopySource(source_inventory_id=INVENTORY_ID, bands=["cbd"])
        assert source.source_inventory_checksum is None

    def test_json_round_trip_preserves_explicit_nulls(self):
        req = make_request(
            bands=["cbd", "cbh"],
            cbd={"method": "maximum_running_mean", "window": None},
            cbh={
                "method": "bulk_density_threshold",
                "threshold": 0.037,
                "relative_threshold_fraction": None,
            },
        )
        dumped = self._source_from_request(req).model_dump(mode="json")
        # Load-bearing nulls survive the dump (no exclude_none).
        assert dumped["cbd"] == {
            "method": "maximum_running_mean",
            "window": None,
            "edge": "ground_clamped",
        }
        assert dumped["cbh"]["relative_threshold_fraction"] is None
        assert dumped["chm"] is None
        assert dumped["cc"] is None
        reloaded = InventoryCanopySource.model_validate(dumped)
        assert reloaded.cbd.window is None
        assert reloaded.cbh.relative_threshold_fraction is None
        assert reloaded == self._source_from_request(req)

    def test_persisted_source_records_resolved_defaults(self):
        dumped = self._source_from_request(make_request()).model_dump(mode="json")
        assert dumped["biomass_source"] == {"type": "allometry", "equations": "nsvb"}
        assert dumped["available_fuel"]["branchwood"] == {
            "size_partition": "none",
            "fraction": 0.075,
        }
        assert dumped["cbd"] == {
            "method": "maximum_running_mean",
            "window": 3.0,
            "edge": "ground_clamped",
        }
        assert dumped["cbh"] == {
            "method": "bulk_density_threshold",
            "threshold": 0.012,
            "relative_threshold_fraction": 0.1,
            "smoothing_window": None,
            "smoothing_edge": "ground_clamped",
        }
        assert dumped["cc"] == {"method": "crown_union"}
        assert dumped["max_crown_radius_source"] == {
            "type": "allometry",
            "equations": "purves",
        }

    def test_source_forbids_extra_fields(self):
        with pytest.raises(ValidationError):
            InventoryCanopySource(
                source_inventory_id=INVENTORY_ID, bands=["cbd"], resolution=30.0
            )


class TestOpenApiExamples:
    @pytest.mark.parametrize(
        "name,value",
        INVENTORY_CANOPY_EXAMPLE_VALUES,
        ids=[n for n, _ in INVENTORY_CANOPY_EXAMPLE_VALUES],
    )
    def test_every_example_validates(self, name, value):
        req = CreateInventoryCanopyRequest.model_validate(value)
        assert req.source_inventory_id

    def test_example_values_match_openapi_entries(self):
        for name, value in INVENTORY_CANOPY_EXAMPLE_VALUES:
            assert name in CREATE_INVENTORY_CANOPY_OPENAPI_EXAMPLES, name
            assert CREATE_INVENTORY_CANOPY_OPENAPI_EXAMPLES[name]["value"] == value
        # Only the placeholder-grid_id example is absent from the router-test
        # list; every other OpenAPI example must be exercised there.
        assert set(CREATE_INVENTORY_CANOPY_OPENAPI_EXAMPLES) - {
            n for n, _ in INVENTORY_CANOPY_EXAMPLE_VALUES
        } == {"aligned_to_landfire_grid"}

    @pytest.mark.parametrize("name", sorted(CREATE_INVENTORY_CANOPY_OPENAPI_EXAMPLES))
    def test_every_openapi_example_validates(self, name):
        CreateInventoryCanopyRequest.model_validate(
            CREATE_INVENTORY_CANOPY_OPENAPI_EXAMPLES[name]["value"]
        )

    def test_openapi_entries_have_summary_and_description(self):
        for name, entry in CREATE_INVENTORY_CANOPY_OPENAPI_EXAMPLES.items():
            assert entry["summary"], name
            assert entry["description"], name

    def test_explicit_defaults_example_equals_minimal_request(self):
        examples = dict(INVENTORY_CANOPY_EXAMPLE_VALUES)
        minimal = CreateInventoryCanopyRequest.model_validate(examples["minimal"])
        explicit = CreateInventoryCanopyRequest.model_validate(
            examples["explicit_defaults"]
        )
        # Metadata differs; every modeling choice must resolve identically.
        exclude = {"name", "description", "tags", "alignment"}
        assert minimal.model_dump(exclude=exclude) == explicit.model_dump(
            exclude=exclude
        )

    def test_fuelcalc_comparison_example_uses_parity_settings(self):
        examples = dict(INVENTORY_CANOPY_EXAMPLE_VALUES)
        req = CreateInventoryCanopyRequest.model_validate(
            examples["fuelcalc_comparison"]
        )
        assert req.biomass_source.equations is CanopyBiomassEquations.brown_1978
        assert (
            req.available_fuel.branchwood.size_partition
            is CanopyBranchwoodSizePartition.equations
        )
        assert req.species_inclusion is CanopySpeciesInclusion.fuelcalc_default
        assert isinstance(
            req.crown_class_adjustment, CanopyFuelcalcCrownClassAdjustment
        )
        assert req.horizontal_distribution is CanopyHorizontalDistribution.stem
        assert req.cc.method == "crown_overlap"
        assert req.max_crown_radius_source.equations is (
            CanopyCrownWidthEquations.crookston_stage
        )
        # FuelCalc reduces CBD and locates both threshold heights over
        # the same 5 ft ground-clamped running mean. Omitting either the
        # window or its edge convention moves a reported height by a
        # layer, so parity needs all four settings, not just the window.
        assert req.cbd.window == pytest.approx(1.524)
        assert req.cbd.edge is CanopyRunningMeanEdge.ground_clamped
        for band in (req.cbh, req.chm):
            assert band.smoothing_window == pytest.approx(1.524)
            assert band.smoothing_edge is CanopyRunningMeanEdge.ground_clamped
