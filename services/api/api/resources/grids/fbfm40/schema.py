"""
api/v2/resources/grids/fbfm40/schema.py

Schema models for the FBFM40 grid product.

FBFM40 returns categorical Scott-Burgan 40 fuel model codes at 30m
resolution from LANDFIRE. To convert codes to fuel parameters (fuel loads,
SAV, depth), use the /grids/lookup/fbfm40 endpoint.
"""

from enum import StrEnum
from typing import Literal

from pydantic import field_validator, model_validator

from api.resources.grids.providers.landfire import (
    LandfireSource,
    NonBurnableFuelModel,
    check_no_duplicate_non_burnable,
)
from api.resources.grids.schema import (
    Band,
    BandType,
    CreateSourceGridRequestBase,
)
from lib.landfire import LANDFIRE_VERSIONS, SEASON_CODES

# Build the version enum class from LANDFIRE_VERSIONS, combining both
# "available" (staged annual) and "lfps_available" (on-demand LFPS)
# versions. Both are accepted here; CreateLandfireFbfm40Request's
# model_validator below then requires `season` for an LFPS-only version,
# and forbids it for a staged-annual-only version.
_FBFM40_ALL_VERSIONS = list(
    dict.fromkeys(
        LANDFIRE_VERSIONS["fbfm40"]["available"]
        + LANDFIRE_VERSIONS["fbfm40"]["lfps_available"]
    )
)
LandfireFbfm40Version = StrEnum(
    "LandfireFbfm40Version",
    {f"v{version}": version for version in _FBFM40_ALL_VERSIONS},
)

# LANDFIRE Seasonal Fuels windows: early spring (ES), spring (SP), summer
# (SU), fall (FA). Mirrors LandfireFbfm40Version's dynamic-enum pattern.
LandfireSeason = StrEnum(
    "LandfireSeason",
    {season: season for season in SEASON_CODES},
)


class LandfireFbfm40Source(LandfireSource):
    """Source for LANDFIRE FBFM40 (Fire Behavior Fuel Model 40).

    Returns categorical fuel model codes at 30m resolution. The codes
    correspond to Scott-Burgan 40 fuel model classifications.
    """

    product: Literal["fbfm40"] = "fbfm40"
    description: Literal[
        "LANDFIRE FBFM40 fuel model codes (Scott-Burgan 40 classification)"
    ] = "LANDFIRE FBFM40 fuel model codes (Scott-Burgan 40 classification)"
    remove_non_burnable: list[str] | None = None
    season: str | None = None


class CreateLandfireFbfm40Request(CreateSourceGridRequestBase):
    """Request to create a grid from LANDFIRE FBFM40.

    Returns a single-band grid with categorical fuel model codes.
    To convert codes to fuel parameters, use /grids/lookup/fbfm40.
    """

    version: LandfireFbfm40Version = LandfireFbfm40Version(
        LANDFIRE_VERSIONS["fbfm40"]["default"]
    )
    remove_non_burnable: list[NonBurnableFuelModel] | None = None
    season: LandfireSeason | None = None

    @field_validator("remove_non_burnable")
    @classmethod
    def check_no_duplicates(cls, v):
        return check_no_duplicate_non_burnable(v)

    @model_validator(mode="after")
    def check_version_matches_season(self):
        """Restrict `version` to the correct list depending on `season`.

        Annual and seasonal FBFM40 draw from different LANDFIRE_VERSIONS
        lists (see lib/landfire/config.py) -- this keeps a seasonal
        request from silently defaulting to an annual-only version (and
        vice versa) rather than failing downstream in griddle.
        """

        versions = LANDFIRE_VERSIONS["fbfm40"]
        if self.season is None and self.version not in versions["available"]:
            raise ValueError(
                f"version {self.version} is only available for seasonal "
                f"(season=...) requests. Available annual versions: "
                f"{', '.join(versions['available'])}."
            )

        if self.season is not None and self.version not in versions["lfps_available"]:
            raise ValueError(
                f"version {self.version} is not available for LANDFIRE "
                f"Seasonal Fuels. Available seasonal versions: "
                f"{', '.join(versions['lfps_available'])}."
            )

        return self


FBFM40_BAND = Band(
    key="fbfm",
    name="Scott & Burgan 40 Fuel Model",
    description=(
        "Scott & Burgan 40 fire behavior fuel model code (e.g. GR1, TL3, SH5). "
        "Convert to fuel parameters via /grids/lookup/fbfm40."
    ),
    type=BandType.categorical,
    unit=None,
    index=0,
)
