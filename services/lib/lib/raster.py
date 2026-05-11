# External imports
import math

import rasterio
import rioxarray
from affine import Affine
from geopandas import GeoDataFrame
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, transform_bounds
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
        destination_crs: CRS | None = None,
        destination_transform: Affine | None = None,
        destination_shape: tuple[int, int] | None = None,
        destination_resolution: float | None = None,
        resampling: Resampling = Resampling.nearest,
    ) -> DataArray:
        """Extract the raster window covering the ROI plus result-cell padding.

        Destination semantics (single rio.reproject path):

        - ``destination_transform`` and ``destination_shape`` both set:
          reproject directly into that lattice. The destination defines
          the extent, so the trailing clip step is skipped.
        - ``destination_crs`` set without transform/shape: reproject to
          that CRS at ``destination_resolution`` (if given) or the source's
          native cell size, then clip to the ROI extent + padding.
        - All None: today's behavior — reproject to ``roi.crs`` (or skip
          if equal) and clip to the ROI.
        """
        if self.connection_type == "rioxarray":
            return self._extract_window_rioxarray(
                roi,
                interpolation_padding_cells,
                destination_crs=destination_crs,
                destination_transform=destination_transform,
                destination_shape=destination_shape,
                destination_resolution=destination_resolution,
                resampling=resampling,
            )

    def _extract_window_rioxarray(
        self,
        roi: GeoDataFrame,
        interpolation_padding_cells: int,
        destination_crs: CRS | None = None,
        destination_transform: Affine | None = None,
        destination_shape: tuple[int, int] | None = None,
        destination_resolution: float | None = None,
        resampling: Resampling = Resampling.nearest,
    ) -> DataArray:
        """Extract the window of the raster that contains the ROI using
        rioxarray. Performs a single reprojection.

        ``interpolation_padding_cells`` is measured in final result-grid
        cells in the ROI CRS, not in source raster cells. On the
        destination-override path the buffer is already baked into
        ``destination_transform``/``destination_shape`` by the alignment
        helper, so ``interpolation_padding_cells`` is effectively unused
        there — the source clip is sized from the destination bounds
        instead, which may extend beyond the ROI (e.g. ``target='grid'``
        with a buffered target grid).
        """
        # Destination override: reproject directly to the requested lattice.
        # Source clip must cover the full destination bounds — not the ROI —
        # so reproject can fill every destination cell. The destination may
        # extend beyond the ROI when target='grid' aligns to a grid with its
        # own footprint buffer, or when extent_buffer_cells expanded the
        # destination lattice.
        if destination_transform is not None and destination_shape is not None:
            dst_crs = destination_crs if destination_crs is not None else roi.crs
            window = self.raster.rio.clip_box(
                *self._source_clip_bounds_for_destination(
                    destination_transform, destination_shape, dst_crs
                )
            )
            return window.rio.reproject(
                dst_crs,
                transform=destination_transform,
                shape=destination_shape,
                resampling=resampling,
            )

        window = self.raster.rio.clip_box(
            *self._source_clip_bounds(
                roi,
                interpolation_padding_cells,
                destination_resolution=destination_resolution,
            )
        )

        # CRS-only override (e.g. target="native" with a custom resolution):
        # reproject preserving source-pixel anchor, optionally at a new
        # resolution, then clip to the ROI extent + padding.
        if destination_crs is not None:
            if destination_resolution is not None:
                window_reprojected = window.rio.reproject(
                    destination_crs,
                    resolution=destination_resolution,
                    resampling=resampling,
                )
            else:
                window_reprojected = window.rio.reproject(
                    destination_crs, resampling=resampling
                )
            roi_padded = self._target_clip_bounds(
                roi,
                interpolation_padding_cells,
                destination_resolution=destination_resolution,
            )
            return window_reprojected.rio.clip_box(*roi_padded)

        # Default behavior: reproject to ROI CRS (if needed) and clip.
        if roi.crs != self.raster_crs:
            window_reprojected = window.rio.reproject(roi.crs, resampling=resampling)
        else:
            window_reprojected = window

        roi_padded = self._target_clip_bounds(roi, interpolation_padding_cells)
        return window_reprojected.rio.clip_box(*roi_padded)

    def _source_clip_bounds_for_destination(
        self,
        destination_transform: Affine,
        destination_shape: tuple[int, int],
        destination_crs: CRS,
    ) -> tuple[float, float, float, float]:
        """Return source raster bounds covering a destination lattice.

        Used by the destination-override path so the source clip covers
        every cell ``rio.reproject`` will need to fill — including cells
        outside the ROI when the destination lattice extends beyond it
        (e.g. ``target='grid'`` aligned to a grid with its own footprint
        buffer, or ``extent_buffer_cells > 0``). The destination footprint
        is north-up by construction (``alignment.lattice_from_bounds``),
        so ``e < 0`` and ``a > 0`` here.
        """
        h, w = destination_shape
        a = destination_transform.a
        e = destination_transform.e
        minx = destination_transform.c
        maxy = destination_transform.f
        maxx = minx + w * a
        miny = maxy + h * e

        if destination_crs != self.raster_crs:
            bounds = transform_bounds(
                destination_crs, self.raster_crs, minx, miny, maxx, maxy
            )
        else:
            bounds = (minx, miny, maxx, maxy)

        x_guard = REPROJECTION_GUARD_CELLS * self.raster_x_resolution
        y_guard = REPROJECTION_GUARD_CELLS * self.raster_y_resolution
        return (
            bounds[0] - x_guard,
            bounds[1] - y_guard,
            bounds[2] + x_guard,
            bounds[3] + y_guard,
        )

    def _source_clip_bounds(
        self,
        roi: GeoDataFrame,
        interpolation_padding_cells: int = 0,
        destination_resolution: float | None = None,
    ) -> tuple[float, float, float, float]:
        """Return source raster bounds needed to cover the ROI after reprojection."""
        target_bounds = self._target_clip_bounds(
            roi,
            interpolation_padding_cells,
            destination_resolution=destination_resolution,
        )
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
        destination_resolution: float | None = None,
    ) -> tuple[float, float, float, float]:
        """Return final output clip bounds in the ROI CRS.

        Padding is measured in final result-grid cells. When
        ``destination_resolution`` is supplied (alignment paths that change
        the output cell size), the buffer is sized in destination cells.
        Otherwise the source raster's cell size is estimated in the ROI
        CRS, preserving pre-#205 behavior on the default path.
        """
        if interpolation_padding_cells == 0:
            return tuple(roi.total_bounds)

        if destination_resolution is not None:
            x_resolution = y_resolution = destination_resolution
        else:
            x_resolution, y_resolution = self.target_native_resolution(roi)

        x_padding = interpolation_padding_cells * x_resolution
        y_padding = interpolation_padding_cells * y_resolution
        return (
            roi.total_bounds[0] - x_padding,
            roi.total_bounds[1] - y_padding,
            roi.total_bounds[2] + x_padding,
            roi.total_bounds[3] + y_padding,
        )

    def target_native_resolution(self, roi: GeoDataFrame) -> tuple[float, float]:
        """Estimate the source raster's native pixel size in the ROI CRS.

        Used by alignment paths so that ``source_native_resolution`` passed
        to ``resolve_alignment_destination`` is in domain CRS units, not
        source CRS units. For a geographic source (degrees) feeding a UTM
        domain (meters), this returns metres-per-pixel.
        """
        if roi.crs == self.raster_crs:
            return self.raster_x_resolution, self.raster_y_resolution

        source_bounds = transform_bounds(roi.crs, self.raster_crs, *roi.total_bounds)
        source_width = max(
            1,
            math.ceil((source_bounds[2] - source_bounds[0]) / self.raster_x_resolution),
        )
        source_height = max(
            1,
            math.ceil((source_bounds[3] - source_bounds[1]) / self.raster_y_resolution),
        )
        target_transform, _, _ = calculate_default_transform(
            self.raster_crs,
            roi.crs,
            source_width,
            source_height,
            *source_bounds,
        )
        return abs(target_transform.a), abs(target_transform.e)
