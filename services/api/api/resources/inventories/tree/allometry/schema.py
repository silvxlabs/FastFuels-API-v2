"""
api/v2/resources/inventories/tree/allometry/schema.py

Crown base height model shared by the allometry routes and the PIM sources.

A model gives a tree's crown base height. The equations compute it from the
tree's species, dbh, FIA compacted crown ratio and height; ``gdam`` predicts it
with GDAM. Discriminated on ``name`` the way ``method`` selects the fusion
algorithm.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class FiaNationalLogit2026Model(BaseModel):
    """National refit to FIA trees with both compacted and uncompacted crown
    ratios recorded. Covers every FIA species, using the California, Oregon and
    Washington coefficients when the domain centroid is in one of those states.
    """

    name: Literal["fia_national_logit_2026"] = "fia_national_logit_2026"


class Monleon2004Eq1Model(BaseModel):
    """Monleon, Azuma and Gedney (2004) Eq. 1, fitted in California, Oregon and
    Washington from crown ratio, dbh and height. Covers 28 species plus other
    pines and oaks; other species get no crown base.
    """

    name: Literal["monleon_2004_eq1"] = "monleon_2004_eq1"


class Monleon2004Eq2Model(BaseModel):
    """Monleon, Azuma and Gedney (2004) Eq. 2, fitted in California, Oregon and
    Washington from crown ratio and dbh. Covers the Eq. 1 species plus other
    pines and oaks; other species get no crown base.
    """

    name: Literal["monleon_2004_eq2"] = "monleon_2004_eq2"


class ToneyReeves2009Model(BaseModel):
    """Toney and Reeves (2009), fitted in the Interior West (ID, MT, WY, NV, UT,
    CO, AZ, NM). Covers 31 species; other species get no crown base.
    """

    name: Literal["toney_reeves_2009"] = "toney_reeves_2009"


class Randolph2010Model(BaseModel):
    """Randolph (2010), fitted in the South (AL, AR, FL, GA, KY, LA, NC, SC, TN,
    TX, VA). Covers 24 species, 10 genera and several junipers; other species get
    no crown base.
    """

    name: Literal["randolph_2010"] = "randolph_2010"


class Westfall2019Model(BaseModel):
    """Westfall, Westfall and Randolph (2019), fitted in the 24 northern US
    states. Covers 27 of FIA's eastern species groups; species outside them get
    no crown base.
    """

    name: Literal["westfall_2019"] = "westfall_2019"


class GdamModel(BaseModel):
    """GDAM's flow-matching model predicts the uncompacted crown jointly with
    dbh, crown ratio and species, conditioned on whichever of them each tree
    has. Needs only position and height; covers the conterminous US.
    """

    name: Literal["gdam"] = "gdam"


# Models that compute the crown base from species, dbh, crown ratio and height.
CROWN_BASE_HEIGHT_EQUATIONS = (
    FiaNationalLogit2026Model,
    Monleon2004Eq1Model,
    Monleon2004Eq2Model,
    ToneyReeves2009Model,
    Randolph2010Model,
    Westfall2019Model,
)

GDAM_CROWN_BASE_HEIGHT_MODEL = GdamModel().name

CrownBaseHeightModel = Annotated[
    Union[(*CROWN_BASE_HEIGHT_EQUATIONS, GdamModel)],
    Field(discriminator="name"),
]


def default_crown_base_height_model() -> FiaNationalLogit2026Model:
    return FiaNationalLogit2026Model()
