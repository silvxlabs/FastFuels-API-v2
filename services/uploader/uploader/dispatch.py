"""
Handler dispatch for Uploader.

Routes upload events to the appropriate handler based on resource type.
Object path convention: {resource_type}/{resource_id}/{filename}
"""

from lib.errors import ProcessingError
from uploader.handlers.grid import handle_grid_geotiff, handle_grid_netcdf
from uploader.handlers.inventory import handle_inventory


def dispatch_handler(
    resource_type: str,
    resource_id: str,
    bucket: str,
    object_name: str,
    doc: dict,
) -> None:
    """Route to appropriate handler based on resource type.

    Args:
        resource_type: First path segment (e.g. "inventories", "grids")
        resource_id: Second path segment — the Firestore document ID
        bucket: GCS bucket the object was written to
        object_name: Full GCS object path
        doc: Resource document loaded from Firestore

    Raises:
        ProcessingError: If resource type has no registered handler, or if a
            "grids" doc has an unknown source.format.
    """
    match resource_type:
        case "inventories":
            handle_inventory(resource_id, bucket, object_name, doc)
        case "grids":
            fmt = (doc.get("source") or {}).get("format")
            match fmt:
                case "geotiff":
                    handle_grid_geotiff(resource_id, bucket, object_name, doc)
                case "netcdf":
                    handle_grid_netcdf(resource_id, bucket, object_name, doc)
                case _:
                    raise ProcessingError(
                        code="UNKNOWN_GRID_FORMAT",
                        message=f"No uploader handler for grid format: {fmt}",
                    )
        case _:
            raise ProcessingError(
                code="UNKNOWN_RESOURCE_TYPE",
                message=f"No handler for resource type: {resource_type}",
            )
