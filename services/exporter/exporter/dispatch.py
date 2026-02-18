"""
Handler dispatch for exporter.

Routes export requests to the appropriate handler based on source format.
"""

from collections.abc import Callable

from exporter.errors import ProcessingError
from exporter.handlers import geotiff


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
            return geotiff.export_geotiff(export, source, progress)
        # Future handlers:
        # case "quicfire":
        #     return quicfire.export_quicfire(export, source, progress)
        case _:
            raise ProcessingError(
                code="UNKNOWN_FORMAT",
                message=f"Unknown export format: {source_name}",
                suggestion="Supported formats: geotiff",
            )
