"""
Unit tests for uploader/handlers/inventory.py

Tests parsing and validation logic in isolation using temporary files.
No GCP I/O — all file operations use local /tmp paths.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import MultiPoint, Point
from uploader.handlers.inventory import _parse, _validate

from lib.errors import ProcessingError

# Domain CRS used across tests (UTM zone 10N — matches typical California domains)
DOMAIN_CRS = "EPSG:32610"


def _write_csv(data: dict, path: str) -> None:
    pd.DataFrame(data).to_csv(path, index=False)


def _write_geojson(
    points: list[tuple], attrs: dict, path: str, crs: str = "EPSG:4326"
) -> None:
    """Write a GeoJSON file with Point features."""
    gdf = gpd.GeoDataFrame(
        attrs,
        geometry=[Point(x, y) for x, y in points],
        crs=crs,
    )
    gdf.to_file(path, driver="GeoJSON")


def _write_geopackage(
    points: list[tuple], attrs: dict, path: str, crs: str = DOMAIN_CRS
) -> None:
    """Write a GeoPackage file with Point features."""
    gdf = gpd.GeoDataFrame(
        attrs,
        geometry=[Point(x, y) for x, y in points],
        crs=crs,
    )
    gdf.to_file(path, driver="GPKG")


# Sample coordinates in domain CRS (UTM zone 10N)
SAMPLE_POINTS = [(500000.0, 4200000.0), (500100.0, 4200100.0), (500200.0, 4200200.0)]
SAMPLE_X = [p[0] for p in SAMPLE_POINTS]
SAMPLE_Y = [p[1] for p in SAMPLE_POINTS]
SAMPLE_HEIGHT = [10.0, 15.0, 20.0]


class TestParseCsv:
    def test_v2_column_names_no_mapping(self, tmp_path):
        """CSV already using v2 column names requires no mapping."""
        path = str(tmp_path / "trees.csv")
        _write_csv({"x": SAMPLE_X, "y": SAMPLE_Y, "height": SAMPLE_HEIGHT}, path)

        df = _parse("csv", path, {}, DOMAIN_CRS)
        assert list(df["x"]) == SAMPLE_X
        assert list(df["height"]) == SAMPLE_HEIGHT

    def test_column_mapping_renames_columns(self, tmp_path):
        """Column mapping renames user columns to v2 names."""
        path = str(tmp_path / "trees.csv")
        _write_csv(
            {"easting": SAMPLE_X, "northing": SAMPLE_Y, "HT": SAMPLE_HEIGHT}, path
        )

        col_map = {"x": "easting", "y": "northing", "height": "HT"}
        df = _parse("csv", path, col_map, DOMAIN_CRS)
        assert "x" in df.columns
        assert "y" in df.columns
        assert "height" in df.columns
        assert "easting" not in df.columns

    def test_non_v2_columns_dropped(self, tmp_path):
        """Columns that are neither v2 names nor in the mapping are dropped."""
        path = str(tmp_path / "trees.csv")
        _write_csv(
            {
                "x": SAMPLE_X,
                "y": SAMPLE_Y,
                "height": SAMPLE_HEIGHT,
                "plot_id": [1, 2, 3],
                "surveyor": ["a", "b", "c"],
            },
            path,
        )

        df = _parse("csv", path, {}, DOMAIN_CRS)
        assert "plot_id" not in df.columns
        assert "surveyor" not in df.columns

    def test_optional_columns_preserved_when_present(self, tmp_path):
        """Optional v2 columns present in the file are kept."""
        path = str(tmp_path / "trees.csv")
        _write_csv(
            {
                "x": SAMPLE_X,
                "y": SAMPLE_Y,
                "height": SAMPLE_HEIGHT,
                "fia_species_code": [122, 202, 15],
                "dbh": [10.0, 20.0, 30.0],
            },
            path,
        )

        df = _parse("csv", path, {}, DOMAIN_CRS)
        assert "fia_species_code" in df.columns
        assert "dbh" in df.columns

    def test_alias_mapping_for_optional_columns(self, tmp_path):
        """Mapping works for optional columns like SPCD→fia_species_code."""
        path = str(tmp_path / "trees.csv")
        _write_csv(
            {
                "x": SAMPLE_X,
                "y": SAMPLE_Y,
                "height": SAMPLE_HEIGHT,
                "SPCD": [122, 202, 15],
            },
            path,
        )

        col_map = {"fia_species_code": "SPCD"}
        df = _parse("csv", path, col_map, DOMAIN_CRS)
        assert "fia_species_code" in df.columns
        assert list(df["fia_species_code"]) == [122, 202, 15]


class TestParseGeoJson:
    def test_point_geometry_extracts_xy(self, tmp_path):
        """GeoJSON Point features: x/y extracted from geometry."""
        # GeoJSON coordinates are always lon/lat (EPSG:4326)
        # Use small lon/lat values near California
        lon_lat_points = [(-120.0, 37.0), (-120.1, 37.1)]
        path = str(tmp_path / "trees.geojson")
        _write_geojson(lon_lat_points, {"height": [10.0, 15.0]}, path)

        df = _parse("geojson", path, {}, DOMAIN_CRS)
        assert "x" in df.columns
        assert "y" in df.columns
        assert "height" in df.columns
        assert len(df) == 2

    def test_multipoint_exploded_to_points(self, tmp_path):
        """MultiPoint features are exploded into individual Point rows."""
        gdf = gpd.GeoDataFrame(
            {"height": [10.0]},
            geometry=[MultiPoint([(-120.0, 37.0), (-120.1, 37.1)])],
            crs="EPSG:4326",
        )
        path = str(tmp_path / "trees.geojson")
        gdf.to_file(path, driver="GeoJSON")

        df = _parse("geojson", path, {}, DOMAIN_CRS)
        assert len(df) == 2

    def test_column_mapping_applied_to_attributes(self, tmp_path):
        """Column mapping renames attribute columns."""
        lon_lat_points = [(-120.0, 37.0), (-120.1, 37.1)]
        path = str(tmp_path / "trees.geojson")
        _write_geojson(lon_lat_points, {"HT": [10.0, 15.0]}, path)

        col_map = {"height": "HT"}
        df = _parse("geojson", path, col_map, DOMAIN_CRS)
        assert "height" in df.columns
        assert "HT" not in df.columns

    def test_crs_reprojected_to_domain(self, tmp_path):
        """GeoJSON coordinates are reprojected from EPSG:4326 to domain CRS."""
        lon_lat_points = [(-120.0, 37.0)]
        path = str(tmp_path / "trees.geojson")
        _write_geojson(lon_lat_points, {"height": [10.0]}, path)

        df = _parse("geojson", path, {}, DOMAIN_CRS)
        # After reprojection to UTM zone 10N, x should be ~500k meters
        assert df["x"].iloc[0] > 100_000

    def test_empty_file_raises(self, tmp_path):
        """Empty GeoJSON raises EMPTY_FILE error."""
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        path = str(tmp_path / "empty.geojson")
        gdf.to_file(path, driver="GeoJSON")

        with pytest.raises(ProcessingError) as exc_info:
            _parse("geojson", path, {}, DOMAIN_CRS)
        assert exc_info.value.code == "EMPTY_FILE"

    def test_polygon_geometry_raises(self, tmp_path):
        """Non-point geometry type raises INVALID_GEOMETRY_TYPE error."""
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame(
            {"height": [10.0]},
            geometry=[Polygon([(-120, 37), (-120.1, 37), (-120.1, 37.1), (-120, 37)])],
            crs="EPSG:4326",
        )
        path = str(tmp_path / "polygon.geojson")
        gdf.to_file(path, driver="GeoJSON")

        with pytest.raises(ProcessingError) as exc_info:
            _parse("geojson", path, {}, DOMAIN_CRS)
        assert exc_info.value.code == "INVALID_GEOMETRY_TYPE"


class TestParseGeoPackage:
    def test_geopackage_extracts_xy(self, tmp_path):
        """GeoPackage with domain CRS extracts x/y without reprojection."""
        path = str(tmp_path / "trees.gpkg")
        _write_geopackage(
            SAMPLE_POINTS, {"height": SAMPLE_HEIGHT}, path, crs=DOMAIN_CRS
        )

        df = _parse("geopackage", path, {}, DOMAIN_CRS)
        assert "x" in df.columns
        assert len(df) == len(SAMPLE_POINTS)

    def test_geopackage_different_crs_reprojected(self, tmp_path):
        """GeoPackage with different CRS is reprojected to domain CRS."""
        path = str(tmp_path / "trees.gpkg")
        # Write in geographic CRS
        lon_lat_points = [(-120.0, 37.0), (-120.1, 37.1)]
        _write_geopackage(
            lon_lat_points, {"height": [10.0, 15.0]}, path, crs="EPSG:4326"
        )

        df = _parse("geopackage", path, {}, DOMAIN_CRS)
        # After reprojection to UTM zone 10N, x should be in meters
        assert df["x"].iloc[0] > 100_000


class TestValidate:
    def _make_df(self, **kwargs):
        defaults = {
            "x": SAMPLE_X,
            "y": SAMPLE_Y,
            "height": SAMPLE_HEIGHT,
        }
        defaults.update(kwargs)
        return pd.DataFrame(defaults)

    def test_valid_minimal_df_passes(self):
        """DataFrame with only required columns passes validation."""
        df = self._make_df()
        result = _validate(df)
        assert list(result["x"]) == SAMPLE_X

    def test_missing_height_raises(self):
        """Missing required height column raises SCHEMA_VALIDATION_ERROR."""
        df = pd.DataFrame({"x": SAMPLE_X, "y": SAMPLE_Y})
        with pytest.raises(ProcessingError) as exc_info:
            _validate(df)
        assert exc_info.value.code == "SCHEMA_VALIDATION_ERROR"

    def test_missing_x_raises(self):
        """Missing required x column raises SCHEMA_VALIDATION_ERROR."""
        df = pd.DataFrame({"y": SAMPLE_Y, "height": SAMPLE_HEIGHT})
        with pytest.raises(ProcessingError) as exc_info:
            _validate(df)
        assert exc_info.value.code == "SCHEMA_VALIDATION_ERROR"

    def test_missing_optional_columns_added_as_null(self):
        """Missing optional columns are added as NaN and do not cause errors."""
        df = self._make_df()
        result = _validate(df)
        assert "fia_species_code" in result.columns
        assert result["fia_species_code"].isna().all()

    def test_optional_columns_preserved_when_present(self):
        """Optional columns with valid values pass validation."""
        df = self._make_df(
            fia_species_code=[122, 202, 15],
            fia_status_code=[1, 1, 2],
            dbh=[10.0, 20.0, 30.0],
            crown_ratio=[0.5, 0.6, 0.7],
        )
        result = _validate(df)
        assert list(result["fia_species_code"]) == [122, 202, 15]

    def test_fia_status_code_out_of_range_raises(self):
        """fia_status_code value not in [0,1,2,3] raises SCHEMA_VALIDATION_ERROR."""
        df = self._make_df(fia_status_code=[1, 1, 5])
        with pytest.raises(ProcessingError) as exc_info:
            _validate(df)
        assert exc_info.value.code == "SCHEMA_VALIDATION_ERROR"

    def test_negative_height_raises(self):
        """height < 0 raises SCHEMA_VALIDATION_ERROR."""
        df = self._make_df(height=[-1.0, 10.0, 15.0])
        with pytest.raises(ProcessingError) as exc_info:
            _validate(df)
        assert exc_info.value.code == "SCHEMA_VALIDATION_ERROR"

    def test_height_above_max_raises(self):
        """height > 116 raises SCHEMA_VALIDATION_ERROR."""
        df = self._make_df(height=[10.0, 120.0, 15.0])
        with pytest.raises(ProcessingError) as exc_info:
            _validate(df)
        assert exc_info.value.code == "SCHEMA_VALIDATION_ERROR"

    def test_crown_ratio_out_of_range_raises(self):
        """crown_ratio > 1 raises SCHEMA_VALIDATION_ERROR."""
        df = self._make_df(crown_ratio=[0.5, 1.5, 0.7])
        with pytest.raises(ProcessingError) as exc_info:
            _validate(df)
        assert exc_info.value.code == "SCHEMA_VALIDATION_ERROR"

    def test_string_height_coerced(self):
        """String height values are coerced to float (Config: coerce=True)."""
        df = self._make_df(height=["10.0", "15.0", "20.0"])
        result = _validate(df)
        assert result["height"].dtype == float

    def test_null_optional_values_allowed(self):
        """NaN in optional columns does not raise."""

        df = self._make_df(fia_species_code=[122, None, 15])
        result = _validate(df)
        assert pd.isna(result["fia_species_code"].iloc[1])
