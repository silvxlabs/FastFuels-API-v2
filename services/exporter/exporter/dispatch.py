"""
Handler dispatch for exporter.

Routes export requests to the appropriate handler based on source format.
"""

from collections.abc import Callable

from exporter.errors import ProcessingError
from exporter.handlers import grid, inventory, netcdf, quicfire


def dispatch_handler(
    export: dict,
    progress: Callable[[str, int | None], None],
) -> str:
    """Route to appropriate handler based on source format.

    Args:
        export: Export document from Firestore
        progress: Function to report progress (message, percent)

    Returns:
        GCS path to the exported file

    Raises:
        ProcessingError: If source format is unknown or processing fails
    """
    source = export["source"]
    source_name = source["name"]

    match source_name:
        case "geotiff":
            return grid.export_geotiff(export, source, progress)
        case "zarr":
            return grid.export_zarr(export, source, progress)
        case "netcdf":
            return netcdf.export_netcdf(export, source, progress)
        case "parquet":
            return inventory.export_parquet(export, source, progress)
        case "csv":
            return inventory.export_csv(export, source, progress)
        case "geojson":
            return inventory.export_geojson(export, source, progress)
        case "geopackage":
            return inventory.export_geopackage(export, source, progress)
        case "quicfire":
            return quicfire.export_quicfire(export, source, progress)
        case _:
            raise ProcessingError(
                code="UNKNOWN_FORMAT",
                message=f"Unknown export format: {source_name}",
                suggestion="Supported formats: geotiff, zarr, netcdf, parquet, csv, geojson, geopackage, quicfire",
            )
