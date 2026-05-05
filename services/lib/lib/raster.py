# External imports
import rasterio
import rioxarray
from geopandas import GeoDataFrame
from rasterio.warp import transform_bounds
from xarray import DataArray

# Reduce HTTP round trips when opening remote Cloud Optimized GeoTIFFs.
# See: https://gdal.org/en/stable/user/configoptions.html
GDAL_COG_CONFIG = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "YES",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "5000000",
    "GDAL_INGESTED_BYTES_AT_OPEN": "32768",
}

REPROJECTION_GUARD_CELLS = 3


def cog_env(**extra: str) -> rasterio.Env:
    """rasterio.Env preconfigured for remote COG access.

    Caller-supplied kwargs override or extend GDAL_COG_CONFIG (e.g.
    AWS_NO_SIGN_REQUEST="YES" for anonymous public S3 buckets).
    """
    return rasterio.Env(**{**GDAL_COG_CONFIG, **extra})


class RasterConnection:
    allowed_connection_types = ["rioxarray"]

    def __init__(self, raster_path: str, connection_type: str = "rioxarray", **kwargs):
        if connection_type not in self.allowed_connection_types:
            raise ValueError(f"Connection type {connection_type} not supported.")
        self.connection_type = connection_type

        # Connect to the raster
        if connection_type == "rioxarray":
            self.raster = rioxarray.open_rasterio(raster_path, **kwargs)
            self.raster_crs = self.raster.rio.crs
            x_resolution, y_resolution = self.raster.rio.resolution()
            self.raster_x_resolution = abs(x_resolution)
            self.raster_y_resolution = abs(y_resolution)
            self.raster_resolution = self.raster_x_resolution
            self.raster_bounds = self.raster.rio.bounds()
            self.raster_dtype = self.raster.dtype

    def roi_within_raster_bounds(self, roi: GeoDataFrame) -> bool:
        """
        Check if the ROI is within the bounds of the raster.
        """
        roi_reprojected = roi.to_crs(self.raster_crs)
        roi_bounds = roi_reprojected.total_bounds

        return (
            roi_bounds[0] >= self.raster_bounds[0]
            and roi_bounds[1] >= self.raster_bounds[1]
            and roi_bounds[2] <= self.raster_bounds[2]
            and roi_bounds[3] <= self.raster_bounds[3]
        )

    def extract_window(
        self,
        roi: GeoDataFrame,
        interpolation_padding_cells: int,
    ) -> DataArray:
        """
        Extract the window of the raster that contains the ROI.

        kwargs passed to the
        """
        if self.connection_type == "rioxarray":
            return self._extract_window_rioxarray(roi, interpolation_padding_cells)

    def _extract_window_rioxarray(
        self,
        roi: GeoDataFrame,
        interpolation_padding_cells: int,
    ) -> DataArray:
        """
        Extract the window of the raster that contains the ROI using rioxarray.
        Only reproject if the ROI CRS differs from the raster CRS.
        """
        window = self.raster.rio.clip_box(
            *self._source_clip_bounds(roi, interpolation_padding_cells)
        )

        if roi.crs != self.raster_crs:
            window_reprojected = window.rio.reproject(roi.crs)
        else:
            window_reprojected = window

        roi_padded = self._target_clip_bounds(roi, interpolation_padding_cells)

        clip = window_reprojected.rio.clip_box(*roi_padded)

        return clip

    def _source_clip_bounds(
        self,
        roi: GeoDataFrame,
        interpolation_padding_cells: int = 0,
    ) -> tuple[float, float, float, float]:
        """Return source raster bounds needed to cover the ROI after reprojection."""
        target_bounds = self._target_clip_bounds(roi, interpolation_padding_cells)
        if roi.crs != self.raster_crs:
            bounds = transform_bounds(roi.crs, self.raster_crs, *target_bounds)
        else:
            bounds = tuple(target_bounds)

        x_guard = REPROJECTION_GUARD_CELLS * self.raster_x_resolution
        y_guard = REPROJECTION_GUARD_CELLS * self.raster_y_resolution

        return (
            bounds[0] - x_guard,
            bounds[1] - y_guard,
            bounds[2] + x_guard,
            bounds[3] + y_guard,
        )

    def _target_clip_bounds(
        self,
        roi: GeoDataFrame,
        interpolation_padding_cells: int,
    ) -> tuple[float, float, float, float]:
        """Return final output clip bounds in the ROI CRS."""
        padding = interpolation_padding_cells * self.raster_resolution
        return (
            roi.total_bounds[0] - padding,
            roi.total_bounds[1] - padding,
            roi.total_bounds[2] + padding,
            roi.total_bounds[3] + padding,
        )
