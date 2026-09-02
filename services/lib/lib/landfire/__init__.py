"""
lib.landfire - LANDFIRE data-product config and API client.

Contains the shared version registry and the LANDFIRE Product Service client.
"""

from lib.landfire.config import (
    LANDFIRE_VERSIONS,
    NB_CODE_MAP,
    SEASON_CODES,
    UnknownLandfireVersionError,
    validate_landfire_version,
)
from lib.landfire.lfps import (
    CoverageStatus,
    LandfireRelease,
    LfpsJob,
    LfpsJobFailedError,
    LfpsJobTimeoutError,
    LfpsProduct,
    covers_annual,
    covers_seasonal,
    download,
    list_products,
    list_releases,
    poll_status,
    resolve_lf_product,
    submit_job,
)

__all__ = [
    "LANDFIRE_VERSIONS",
    "NB_CODE_MAP",
    "SEASON_CODES",
    "UnknownLandfireVersionError",
    "validate_landfire_version",
    "LfpsJob",
    "LfpsJobFailedError",
    "LfpsJobTimeoutError",
    "LfpsProduct",
    "list_products",
    "list_releases",
    "LandfireRelease",
    "submit_job",
    "poll_status",
    "download",
    "CoverageStatus",
    "covers_annual",
    "covers_seasonal",
    "resolve_lf_product",
]
