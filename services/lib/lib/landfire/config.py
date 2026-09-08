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
        "lfps_available": ["2025"],
        "default": "2024",
    },
    "fbfm40": {
        "available": ["2019", "2020", "2022", "2023", "2024"],
        "lfps_available": ["2025"],
        "default": "2024",
    },
    "fccs": {
        "available": ["2023"],
        "lfps_available": ["2025"],
        "default": "2023",
    },
    "annual_disturbance": {
        "lfps_available": ["2025"],
        "default": "2025",
    },
}

# Most LANDFIRE_VERSIONS registry keys match their LFPS catalog acronym once
# upper-cased (e.g. "fbfm40" -> "FBFM40"). LFPS_ACRONYM_OVERRIDES lists the
# exceptions, where the LFPS acronym isn't derivable from the registry key.
LFPS_ACRONYM_OVERRIDES: dict[str, str] = {
    "annual_disturbance": "LDist",
}


def lfps_acronym(product: str) -> str:
    """The acronym LFPS's live catalog uses for `product`.

    Returns the override from LFPS_ACRONYM_OVERRIDES if `product`
    is listed there; otherwise returns `product.upper()`.
    """
    return LFPS_ACRONYM_OVERRIDES.get(product, product.upper())


# LANDFIRE's Seasonal Fuels product publishes four windows across the
# year, listed here in calendar order: early spring (ES), spring (SP),
# summer (SU), and fall (FA). Ordering code relies on this order.
SEASON_CODES: tuple[str, ...] = ("ES", "SP", "SU", "FA")


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
