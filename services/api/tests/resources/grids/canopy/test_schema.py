"""
Unit tests for api/v2/resources/grids/canopy/schema.py
and api/v2/resources/grids/providers/canopy.py

Tests the Meta/NAIP canopy schema models, CanopySource base, and band definitions.
These are pure unit tests with no external dependencies.
"""

import pytest
from api.resources.grids.canopy.schema import (
    Attribution,
    ChmMaxAggregation,
    ChmMeanAggregation,
    ChmMedianAggregation,
    ChmPercentileAggregation,
    ChmSpikeFilter,
    CreateMetaChmRequest,
    CreateNaipChmRequest,
    CreatePointCloudChmRequest,
    MetaChmSource,
    NaipChmSource,
    PointCloudChmSource,
    build_chm_bands,
)
from api.resources.grids.providers.canopy import CanopySource
from api.resources.grids.schema import BandType
from pydantic import ValidationError


class TestCanopySource:
    """Tests for CanopySource base model."""

    def test_name_is_always_canopy(self):
        """The name field is always 'canopy'."""
        source = CanopySource(product="meta")
        assert source.name == "canopy"

    def test_name_cannot_be_overridden(self):
        """The name field cannot be set to anything other than 'canopy'."""
        with pytest.raises(ValidationError):
            CanopySource(name="other", product="meta")

    def test_product_is_required(self):
        """The product field is required."""
        with pytest.raises(ValidationError):
            CanopySource()

    def test_description_defaults_to_empty_string(self):
        """The description field defaults to empty string."""
        source = CanopySource(product="meta")
        assert source.description == ""

    def test_description_can_be_set(self):
        """The description field can be set."""
        source = CanopySource(
            product="meta",
            description="Test description",
        )
        assert source.description == "Test description"

    def test_extent_buffer_cells_defaults_to_zero(self):
        source = CanopySource(product="meta")
        assert source.extent_buffer_cells == 0

    def test_extent_buffer_cells_can_be_set(self):
        source = CanopySource(product="meta", extent_buffer_cells=10)
        assert source.extent_buffer_cells == 10

    def test_extent_buffer_cells_rejects_negative(self):
        with pytest.raises(ValidationError):
            CanopySource(product="meta", extent_buffer_cells=-1)


class TestMetaChmSource:
    """Tests for MetaChmSource model."""

    def test_product_is_always_meta(self):
        """The product field is always 'meta'."""
        source = MetaChmSource(version="2")
        assert source.product == "meta"

    def test_product_cannot_be_overridden(self):
        """The product field cannot be set to anything other than 'meta'."""
        with pytest.raises(ValidationError):
            MetaChmSource(product="other", version="2")

    def test_name_is_always_canopy(self):
        """The name field is always 'canopy'."""
        source = MetaChmSource(version="2")
        assert source.name == "canopy"

    def test_description_is_fixed(self):
        """The description has a fixed value."""
        source = MetaChmSource(version="2")
        assert "Meta" in source.description
        assert "canopy height" in source.description

    def test_model_dump(self):
        """Model serializes correctly."""
        source = MetaChmSource(version="2")
        data = source.model_dump()
        assert data["name"] == "canopy"
        assert data["product"] == "meta"
        assert "description" in data

    def test_attribution_defaults_to_none(self):
        """Attribution is None by default."""
        source = MetaChmSource(version="2")
        assert source.attribution is None

    def test_attribution_accepted(self):
        """MetaChmSource accepts an Attribution object."""
        attr = Attribution(
            license_name="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            citation="Test citation",
            access_url="https://example.com",
            accessed_on="2026-02-27",
        )
        source = MetaChmSource(version="2", attribution=attr)
        assert source.attribution.license_name == "CC-BY-4.0"

    def test_attribution_serialized_in_model_dump(self):
        """Attribution is included in model_dump output."""
        attr = Attribution(
            license_name="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            citation="Test citation",
            access_url="https://example.com",
            accessed_on="2026-02-27",
        )
        source = MetaChmSource(version="2", attribution=attr)
        data = source.model_dump()
        assert data["attribution"]["license_name"] == "CC-BY-4.0"
        assert data["attribution"]["accessed_on"] == "2026-02-27"


class TestCreateMetaChmRequest:
    """Tests for CreateMetaChmRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request with no required body fields."""
        request = CreateMetaChmRequest()
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []
        assert request.version == "2"

    def test_full_request_with_all_fields(self):
        """Full request with all optional fields."""
        request = CreateMetaChmRequest(
            name="Test Grid",
            description="A test grid",
            tags=["test", "chm"],
            version="2",
        )
        assert request.name == "Test Grid"
        assert request.description == "A test grid"
        assert request.tags == ["test", "chm"]
        assert request.version == "2"

    def test_version_can_be_set_to_v1(self):
        """Version can be explicitly set to v1."""
        request = CreateMetaChmRequest(version="1")
        assert request.version == "1"

    def test_extent_buffer_cells_defaults_to_zero(self):
        request = CreateMetaChmRequest()
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_accepts_positive(self):
        request = CreateMetaChmRequest(extent_buffer_cells=10)
        assert request.extent_buffer_cells == 10

    def test_extent_buffer_cells_accepts_zero(self):
        request = CreateMetaChmRequest(extent_buffer_cells=0)
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_rejects_negative(self):
        with pytest.raises(ValidationError):
            CreateMetaChmRequest(extent_buffer_cells=-1)

    def test_extent_buffer_cells_rejects_above_maximum(self):
        with pytest.raises(ValidationError):
            CreateMetaChmRequest(extent_buffer_cells=11)


class TestAttribution:
    """Tests for Attribution model."""

    def test_valid_attribution(self):
        """All fields accepted."""
        attr = Attribution(
            license_name="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            citation="Some citation text",
            access_url="https://example.com",
            accessed_on="2026-02-27",
        )
        assert attr.license_name == "CC-BY-4.0"
        assert attr.access_url == "https://example.com"

    def test_missing_field_rejected(self):
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            Attribution(
                license_name="CC-BY-4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
                # missing citation, access_url, accessed_on
            )


class TestChmBands:
    """Tests for build_chm_bands helper and band definitions."""

    def test_single_chm_band(self):
        """build_chm_bands returns a single band."""
        bands = build_chm_bands()
        assert len(bands) == 1

    def test_band_key_is_chm(self):
        """The band key is 'chm'."""
        bands = build_chm_bands()
        assert bands[0].key == "chm"

    def test_band_type_is_continuous(self):
        """The band type is continuous."""
        bands = build_chm_bands()
        assert bands[0].type == BandType.continuous

    def test_band_unit_is_meters(self):
        """The band unit is 'm'."""
        bands = build_chm_bands()
        assert bands[0].unit == "m"

    def test_band_index_is_zero(self):
        """The band index is 0."""
        bands = build_chm_bands()
        assert bands[0].index == 0

    def test_band_has_name_and_description(self):
        """The CHM band carries a human-readable name and description."""
        bands = build_chm_bands()
        assert bands[0].name == "Canopy Height"
        assert bands[0].description


class TestNaipChmSource:
    """Tests for NaipChmSource model."""

    def test_product_is_always_naip(self):
        """The product field is always 'naip'."""
        source = NaipChmSource()
        assert source.product == "naip"

    def test_name_is_always_canopy(self):
        """The name field is always 'canopy'."""
        source = NaipChmSource()
        assert source.name == "canopy"

    def test_description_is_fixed(self):
        """The description has a fixed value."""
        source = NaipChmSource()
        assert "NAIP" in source.description
        assert "0.6m resolution" in source.description

    def test_model_dump(self):
        """Model serializes correctly."""
        source = NaipChmSource()
        data = source.model_dump()
        assert data["product"] == "naip"
        assert "description" in data


class TestCreateNaipChmRequest:
    """Tests for CreateNaipChmRequest model."""

    def test_minimal_valid_request(self):
        """Minimal request with no required body fields."""
        request = CreateNaipChmRequest()
        # Assumes you updated the default to 2023 in the schema!
        assert request.name == ""
        assert request.description == ""
        assert request.tags == []
        assert request.modifications == []

    def test_full_request_with_all_fields(self):
        """Full request with all optional fields."""
        request = CreateNaipChmRequest(
            name="Test NAIP Grid",
            description="A test grid",
            tags=["test", "chm", "naip"],
        )
        assert request.name == "Test NAIP Grid"
        assert request.description == "A test grid"
        assert request.tags == ["test", "chm", "naip"]

    def test_extent_buffer_cells_defaults_to_zero(self):
        request = CreateNaipChmRequest()
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_accepts_zero(self):
        request = CreateNaipChmRequest(extent_buffer_cells=0)
        assert request.extent_buffer_cells == 0

    def test_extent_buffer_cells_accepts_positive(self):
        request = CreateNaipChmRequest(extent_buffer_cells=10)
        assert request.extent_buffer_cells == 10

    def test_extent_buffer_cells_rejects_negative(self):
        with pytest.raises(ValidationError):
            CreateNaipChmRequest(extent_buffer_cells=-1)

    def test_extent_buffer_cells_rejects_above_maximum(self):
        with pytest.raises(ValidationError):
            CreateNaipChmRequest(extent_buffer_cells=11)


class TestChmSpikeFilter:
    """Removing lone spurious returns from a point-cloud CHM."""

    def test_defaults_are_the_measured_ones(self):
        """A request that says nothing gets the shipped behaviour."""
        spike_filter = ChmSpikeFilter()

        assert spike_filter.min_canopy_footprint_m == 3.0
        assert spike_filter.min_prominence_m == 25.0

    def test_values_are_accepted(self):
        spike_filter = ChmSpikeFilter(
            min_canopy_footprint_m=30.0, min_prominence_m=40.0
        )

        assert spike_filter.min_canopy_footprint_m == 30.0
        assert spike_filter.min_prominence_m == 40.0

    @pytest.mark.parametrize("field", ["min_canopy_footprint_m", "min_prominence_m"])
    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_non_positive_distances_rejected(self, field, value):
        with pytest.raises(ValidationError):
            ChmSpikeFilter(**{field: value})


class TestCreatePointCloudChmRequest:
    """The point-cloud CHM create request."""

    def test_minimal_valid_request(self):
        request = CreatePointCloudChmRequest(source_point_cloud_id="cloud-1")

        assert request.source_point_cloud_id == "cloud-1"

    def test_spike_filter_is_on_by_default(self):
        request = CreatePointCloudChmRequest(source_point_cloud_id="cloud-1")

        assert request.spike_filter == ChmSpikeFilter()

    def test_null_spike_filter_turns_it_off(self):
        """The way to opt out, rather than a "none" sentinel inside the object."""
        request = CreatePointCloudChmRequest(
            source_point_cloud_id="cloud-1", spike_filter=None
        )

        assert request.spike_filter is None

    def test_spike_filter_can_be_set(self):
        request = CreatePointCloudChmRequest(
            source_point_cloud_id="cloud-1",
            spike_filter={"min_canopy_footprint_m": 30.0},
        )

        assert request.spike_filter.min_canopy_footprint_m == 30.0
        assert request.spike_filter.min_prominence_m == 25.0


class TestPointCloudChmSource:
    """What a stored point-cloud CHM grid records about how it was built."""

    def test_spike_filter_is_recorded(self):
        """Resolved, like `alignment.resolution`, so the grid is reproducible."""
        source = PointCloudChmSource(
            source_point_cloud_id="cloud-1", spike_filter=ChmSpikeFilter()
        )

        assert source.model_dump()["spike_filter"] == {
            "min_canopy_footprint_m": 3.0,
            "min_prominence_m": 25.0,
        }

    def test_spike_filter_records_being_off(self):
        source = PointCloudChmSource(source_point_cloud_id="cloud-1", spike_filter=None)

        assert source.model_dump()["spike_filter"] is None


class TestChmAggregation:
    """The statistic a point-cloud CHM cell reduces its returns with."""

    def test_default_is_the_maximum(self):
        """Omitting it has to reproduce what every existing grid was built with."""
        request = CreatePointCloudChmRequest(source_point_cloud_id="cloud-1")

        assert request.aggregation == ChmMaxAggregation()

    @pytest.mark.parametrize(
        "method,model",
        [
            ("max", ChmMaxAggregation),
            ("mean", ChmMeanAggregation),
            ("median", ChmMedianAggregation),
        ],
    )
    def test_a_method_without_a_rank_carries_no_parameter(self, method, model):
        request = CreatePointCloudChmRequest(
            source_point_cloud_id="cloud-1", aggregation={"method": method}
        )

        assert isinstance(request.aggregation, model)
        assert request.aggregation.method == method

    def test_percentile_carries_its_rank(self):
        request = CreatePointCloudChmRequest(
            source_point_cloud_id="cloud-1",
            aggregation={"method": "percentile", "percentile": 98},
        )

        assert isinstance(request.aggregation, ChmPercentileAggregation)
        assert request.aggregation.percentile == 98.0

    def test_percentile_without_a_rank_is_rejected(self):
        """The discriminator is what makes `percentile` required, and only there."""
        with pytest.raises(ValidationError):
            CreatePointCloudChmRequest(
                source_point_cloud_id="cloud-1", aggregation={"method": "percentile"}
            )

    @pytest.mark.parametrize("percentile", [0.0, -1.0, 100.1, 101.0])
    def test_a_rank_outside_the_range_is_rejected(self, percentile):
        with pytest.raises(ValidationError):
            CreatePointCloudChmRequest(
                source_point_cloud_id="cloud-1",
                aggregation={"method": "percentile", "percentile": percentile},
            )

    @pytest.mark.parametrize("percentile", [0.5, 50.0, 95.0, 98.0, 100.0])
    def test_a_rank_inside_the_range_is_accepted(self, percentile):
        request = CreatePointCloudChmRequest(
            source_point_cloud_id="cloud-1",
            aggregation={"method": "percentile", "percentile": percentile},
        )

        assert request.aggregation.percentile == percentile

    def test_a_rank_on_another_method_is_rejected(self):
        """`percentile` is meaningless on `mean`; accepting it would hide a typo."""
        with pytest.raises(ValidationError):
            CreatePointCloudChmRequest(
                source_point_cloud_id="cloud-1",
                aggregation={"method": "mean", "percentile": 95},
            )

    def test_an_unknown_method_is_rejected(self):
        with pytest.raises(ValidationError):
            CreatePointCloudChmRequest(
                source_point_cloud_id="cloud-1", aggregation={"method": "p95"}
            )


class TestPointCloudChmSourceAggregation:
    """What a stored point-cloud CHM grid records about its statistic."""

    def test_the_statistic_is_recorded(self):
        source = PointCloudChmSource(
            source_point_cloud_id="cloud-1",
            aggregation=ChmPercentileAggregation(percentile=95),
        )

        assert source.model_dump()["aggregation"] == {
            "method": "percentile",
            "percentile": 95.0,
        }

    def test_a_grid_that_records_nothing_took_the_maximum(self):
        """Every grid built before the control existed took the maximum.

        The default is a true statement about those grids rather than a
        placeholder, so the field is never null and never has to be read as
        "unknown".
        """
        source = PointCloudChmSource(source_point_cloud_id="cloud-1")

        assert source.model_dump()["aggregation"] == {"method": "max"}
