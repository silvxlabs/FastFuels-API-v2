"""
lib.landfire - LANDFIRE data-product config and API client.

Contains the shared version registry and the LANDFIRE Product Service client.
"""

from lib.landfire.config import (
    LANDFIRE_VERSIONS,
    NB_CODE_MAP,
    UnknownLandfireVersionError,
    validate_landfire_version,
)
from lib.landfire.lfps import (
    CoverageStatus,
    LfpsJob,
    LfpsJobFailedError,
    LfpsJobTimeoutError,
    LfpsProduct,
    covers_annual,
    covers_seasonal,
    download,
    list_products,
    poll_status,
    submit_job,
)

__all__ = [
    "LANDFIRE_VERSIONS",
    "NB_CODE_MAP",
    "UnknownLandfireVersionError",
    "validate_landfire_version",
    "LfpsJob",
    "LfpsJobFailedError",
    "LfpsJobTimeoutError",
    "LfpsProduct",
    "list_products",
    "submit_job",
    "poll_status",
    "download",
    "CoverageStatus",
    "covers_annual",
    "covers_seasonal",
]
