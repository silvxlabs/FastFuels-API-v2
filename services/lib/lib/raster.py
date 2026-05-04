# External imports
import rasterio
import rioxarray
from geopandas import GeoDataFrame
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
            self.raster_resolution = abs(self.raster.rio.resolution()[0])
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
        projection_padding_meters: float,
        interpolation_padding_cells: int,
    ) -> DataArray:
        """
        Extract the window of the raster that contains the ROI.

        kwargs passed to the
        """
        if self.connection_type == "rioxarray":
            return self._extract_window_rioxarray(
                roi, projection_padding_meters, interpolation_padding_cells
            )

    def _extract_window_rioxarray(
        self,
        roi: GeoDataFrame,
        projection_padding_meters: float,
        interpolation_padding_cells: int,
    ) -> DataArray:
        """
        Extract the window of the raster that contains the ROI using rioxarray.
        Only reproject if the ROI CRS differs from the raster CRS. Apply
        projection padding in all cases.
        """
        if roi.crs != self.raster_crs:
            roi_reprojected = roi.to_crs(self.raster_crs)
            roi_bounds = roi_reprojected.total_bounds
        else:
            roi_bounds = roi.total_bounds

        roi_padded_bounds = [
            roi_bounds[0] - projection_padding_meters,
            roi_bounds[1] - projection_padding_meters,
            roi_bounds[2] + projection_padding_meters,
            roi_bounds[3] + projection_padding_meters,
        ]
        window = self.raster.rio.clip_box(*roi_padded_bounds)

        if roi.crs != self.raster_crs:
            window_reprojected = window.rio.reproject(roi.crs)
        else:
            window_reprojected = window

        # Calculate padded bounds in the ROI's CRS
        roi_padded = [
            roi.total_bounds[0] - interpolation_padding_cells * self.raster_resolution,
            roi.total_bounds[1] - interpolation_padding_cells * self.raster_resolution,
            roi.total_bounds[2] + interpolation_padding_cells * self.raster_resolution,
            roi.total_bounds[3] + interpolation_padding_cells * self.raster_resolution,
        ]

        clip = window_reprojected.rio.clip_box(*roi_padded)

        return clip
