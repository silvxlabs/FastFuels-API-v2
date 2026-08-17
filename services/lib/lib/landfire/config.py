"""LANDFIRE config: version registry, fuel-model code map, and LFPS contact email.

Single source of truth for which LANDFIRE data-product versions are
served and which one is the default. `api`'s per-product request schemas
build their version enums from this, and `griddle`'s fetch handlers pull
their default `version` argument from it, so the two services can't drift
apart.
"""

from __future__ import annotations

import os

# Contact email logged for LANDFIRE Product Service
LANDFIRE_USER_EMAIL = os.getenv("LANDFIRE_USER_EMAIL", "lwiard@newmexicoconsortium.org")

LANDFIRE_VERSIONS: dict[str, dict[str, list[str] | str]] = {
    "fbfm13": {
        "available": ["2023", "2024"],
        "default": "2024",
    },
    "fbfm40": {
        "available": ["2019", "2020", "2022", "2023", "2024"],
        "default": "2024",
    },
    "fccs": {
        "available": ["2023"],
        "default": "2023",
    },
}


class UnknownLandfireVersionError(ValueError):
    """Raised when a requested LANDFIRE version isn't available for a product."""

    def __init__(self, product: str, version: str):
        self.product = product
        self.version = version
        available = LANDFIRE_VERSIONS[product]["available"]
        super().__init__(
            f"{version!r} is not an available LANDFIRE {product!r} version. "
            f"Available versions: {', '.join(available)}."
        )


def validate_landfire_version(product: str, version: str) -> str:
    """Validate that `version` is available for `product`; return it unchanged.

    Raises :class:`UnknownLandfireVersionError` for an unavailable version.
    """
    if version not in LANDFIRE_VERSIONS[product]["available"]:
        raise UnknownLandfireVersionError(product, version)
    return version


# Non-burnable LANDFIRE fuel model codes, shared across FBFM40/FBFM13.
NB_CODE_MAP: dict[str, int] = {
    "NB1": 91,  # Urban/developed
    "NB2": 92,  # Snow/ice
    "NB3": 93,  # Agriculture
    "NB8": 98,  # Water
    "NB9": 99,  # Bare ground
}
